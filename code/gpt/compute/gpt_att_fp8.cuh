#pragma once

// possible improvements
// special version for zero alibi slope
// special version for single target attention
// separate kernel to compute max dDot for qk & qv in one pass

constexpr int ATT_BUFFER_COUNT = 4;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp8AttData
{
    union {
        struct {
            T8SMemI8Tile qkFrag[4]; // from 64 samples
        } A;
        struct {
            union {
                T8SMemI8Tile qvFrag[ATT_BUFFER_COUNT]; // prefetch buffering
                T8SMemI8Tile weights[4]; // from 64 to 128 samples
                float OnlineRowScale[64];
            };
            union {
                T8SMemI8Tile vFrag[8]; // 128 dim, to 128 samples
                struct {
                    //float DotProduct[8192]; // keeping floats requires 32kb smem and reduces occupancy
                    half DotProduct[8192]; // could use 24 bit floats or half + row max but model quality seems to be the same
                };
            };
        } B;
        struct {
            // result scaling
            float SumWeight[64];
            float RowScale[64];
            float RowMax[64];
        } C;
    };
};


// requires transposed v
__global__ void Fp8Att(float *qvGlobalScale, TCuda2DPtr<e4m3> qk, TCuda2DPtr<e4m3> qv, TCuda2DPtr<float> qvScale,
    TCuda2DPtr<e4m3> vE4T,
    TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans2, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
    TCuda2DPtr<half> resMatr, TCuda2DPtr<float> sumWeightLog)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    CUDA_STATIC_ASSERT(Q_DIM == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 128);
    TTileCoord tc;

    constexpr int PREFETCH = 2;
    constexpr int TO_TILE_COUNT = 8;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(Q_DIM) * CalcAttentionMult() * *qvGlobalScale;

    int h = threadIdx.x;
    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int fromWarp = warpId * TILE;

    __shared__ TFp8AttData data;

    // load qk to shmem
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        Copy8Tile(&data.A.qkFrag[fromTile], warpId, qk.Fragment(eOffset, fromBase + fromTile * TILE));
    }
    __syncthreads();
    TRegTile<i8> qkFrag[8];
    for (int k = 0; k < 8; ++k) {
        qkFrag[k] = TMmaRowMajor::FragA(data.A.qkFrag[warpId], k);
    }
    __syncthreads();

    // precompute alibi
    float elemAlibi[TTileCoord::num_elements];
    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
        elemAlibi[elem] = GetAttentionDecay(y - x, alibiSlope);
        });

    TRegTile<float> sum[4][2];
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            sum[fromTile][sumX].Clear();
        }
    }

    constexpr float MIN_DOT_PRODUCT = -1e5f;

    TRegTileRow<float> rowMax;
    rowMax.SetMax(MIN_DOT_PRODUCT);
    TRegTileRow<float> sumWeight;
    sumWeight.Clear();
    // to in 128 sample blocks
    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
        TRegTileRow<int> rowSpanStart;
        TRegTileRow<int> rowSpanFinish;
        rowSpanStart.Load(tc, gg.SpanStart + fromWarp);
        rowSpanFinish.Load(tc, gg.SpanFinish + fromWarp);
        rowSpanStart.Add(-gg.Start);
        rowSpanFinish.Add(-gg.Start);
        int alibiFromMinusTo = fromBase + fromWarp - gg.Start;
        for (int toBase = gg.Start, ggFinish = gg.Finish; toBase <= ggFinish; toBase += MM_TILE) {

            // compute qk qv dot products, writes data.DotProduct
            TRegTileRow<float> onlineRowScale;
            {
                TRegTileRow<float> newRowMax = rowMax;
                TAsyncLoader<TO_TILE_COUNT, ATT_BUFFER_COUNT, PREFETCH> qvIter(data.B.qvFrag, warpId, &qv[toBase][eOffset], qv.GetStrideInBytes());
                for (; qvIter.IsValid(); qvIter.Next()) {
                    int toTile = qvIter.GetIndex();

                    // load qv
                    qvIter.Load(warpId);

                    // qqTile[from][to] = dot(qk[from], qv[to])
                    TRegTile<float> qqTile;
                    qqTile.Clear();
                    for (int k = 0; k < 8; k += 2) {
                        TRegTile<i8> b1 = TMmaColMajor::FragB(qvIter.GetBuf(), k);
                        TRegTile<i8> b2 = TMmaColMajor::FragB(qvIter.GetBuf(), k + 1);
                        MMAe4e4(&qqTile, qkFrag[k], qkFrag[k + 1], b1, b2);
                    }

                    float tileAlibi = GetAttentionDecay(alibiFromMinusTo, alibiSlope);
                    TRegTileColumn<float> qqScaleColumn;
                    qqScaleColumn.Load(tc, &qvScale[head][toBase + toTile * TILE]);
                    qqScaleColumn.Scale(QK_VEC_SCALE * attDotScale); // qkScale is constant VEC_SCALE

                    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                        float dp = MIN_DOT_PRODUCT;
                        if (x >= rowSpanStart.x[rowIndex] && x <= rowSpanFinish.x[rowIndex]) {
                            //int glFrom = fromBase + fromWarp + y;
                            //int glTo = toBase + toTile * TILE + x;
                            //float dp = qqTile.x[elem] * qkScale[head][glFrom] * qvScale[head][glTo] * attDotScale;
                            //dp += GetAttentionDecay(glFrom - glTo, alibiSlope);
                            dp = qqTile.x[elem] * qqScaleColumn.x[columnIndex];
                            dp += tileAlibi + elemAlibi[elem];
                            newRowMax.x[rowIndex] = max(newRowMax.x[rowIndex], dp);
                        }
                        // save dot product
                        data.B.DotProduct[warpId * 2048 + toTile * 256 + elem * WARP_SIZE + h] = dp;
                        });

                    rowSpanStart.Add(-TILE);
                    rowSpanFinish.Add(-TILE);
                    alibiFromMinusTo -= TILE;
                }
                __syncthreads();

                // compute online result scale, write data.OnlineRowScale
                newRowMax.WarpMaxReduce();
                for (int elem = 0; elem < rowMax.GetNumElements(); ++elem) {
                    newRowMax.x[elem] = floor(newRowMax.x[elem]);
                    onlineRowScale.x[elem] = exp2f(rowMax.x[elem] - newRowMax.x[elem]);
                }
                onlineRowScale.StoreOne(tc, data.B.OnlineRowScale + warpId * TILE);
                rowMax = newRowMax;
                __syncthreads();
            }

            // scale sum by online scale results, read data.OnlineRowScale
            for (int elem = 0; elem < onlineRowScale.GetNumElements(); ++elem) {
                sumWeight.x[elem] *= onlineRowScale.x[elem];
            }
            for (int fromTile = 0; fromTile < 4; ++fromTile) {
                TRegTileRow<float> rowScale;
                rowScale.Load(tc, data.B.OnlineRowScale + fromTile * TILE);
                for (int sumX = 0; sumX < 2; ++sumX) {
                    sum[fromTile][sumX].Scale(rowScale);
                }
            }
            __syncthreads();

            // compute weights, read data.DotProduct, write data.weights
            for (int toTile = 0; toTile < 8; ++toTile) {
                tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                    float dp = data.B.DotProduct[warpId * 2048 + toTile * 256 + elem * WARP_SIZE + h];
                    dp -= rowMax.x[rowIndex];
                    i8 weight = CvtToE4(exp2f(dp) * ATT_W_MULT);
                    sumWeight.x[rowIndex] += CvtE4ToFloat(weight);
                    // store to weights
                    SetElement(&data.B.weights[warpId], toTile * TILE, x, y, weight);
                    });
            }
            __syncthreads();

            // load v, write data.vFrag
            for (int k = 0; k < 8; ++k) {
                Copy8Tile(&data.B.vFrag[k], warpId, vE4T.Fragment(toBase, eOffset + k * TILE));
            }
            __syncthreads();

            // add to sum scaled v[], read data.vFrag, data.weights
            for (int k = 0; k < 8; k += 2) {
                TRegTile<i8> b1[2];
                b1[0] = TMmaColMajor::FragB(data.B.vFrag[warpId * 2 + 0], k);
                b1[1] = TMmaColMajor::FragB(data.B.vFrag[warpId * 2 + 1], k);
                TRegTile<i8> b2[2];
                b2[0] = TMmaColMajor::FragB(data.B.vFrag[warpId * 2 + 0], k + 1);
                b2[1] = TMmaColMajor::FragB(data.B.vFrag[warpId * 2 + 1], k + 1);
                for (int fromTile = 0; fromTile < 4; ++fromTile) {
                    TRegTile<i8> a1 = TMmaRowMajor::FragA(data.B.weights[fromTile], k);
                    TRegTile<i8> a2 = TMmaRowMajor::FragA(data.B.weights[fromTile], k + 1);
                    for (int sumX = 0; sumX < 2; ++sumX) {
                        MMAe4e4(&sum[fromTile][sumX], a1, a2, b1[sumX], b2[sumX]);
                    }
                }
            }
            __syncthreads();
        }
    }
    rowMax.StoreMax(tc, data.C.RowMax + fromWarp);
    sumWeight.StoreSum(tc, data.C.SumWeight + fromWarp);
    __syncthreads();
    if (h < TILE) {
        float sumWeight = data.C.SumWeight[fromWarp + h];
        float dpMax = data.C.RowMax[fromWarp + h];
        if (dpMax < -20) {
            data.C.RowScale[fromWarp + h] = 0;
            sumWeightLog[head][fromBase + fromWarp + h] = 0;
        } else {
            sumWeight += exp2f(-dpMax) * ATT_W_MULT; // add zero vector with weight 1
            data.C.RowScale[fromWarp + h] = 1.0f / sumWeight;
            sumWeightLog[head][fromBase + fromWarp + h] = dpMax + log2f(sumWeight / ATT_W_MULT);
        }
    }
    __syncthreads();
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            TRegTile<float> &rsum = sum[fromTile][sumX];
            TRegTileRow<float> rowScale;
            rowScale.Load(tc, data.C.RowScale + fromTile * TILE);
            rsum.Scale(rowScale);
            rsum.Store(tc, resMatr.Fragment(eOffset + (warpId * 2 + sumX) * TILE, fromBase + fromTile * TILE));
        }
    }
}
KERNEL_BLOCK_SIZE(Fp8Att, WARP_SIZE, 4);



///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr float FP8_ATT_DOT_MULT = 256;

struct TFp8AttGradQKData
{
    union {
        struct {
            T8SMemI8Tile qkFrag[4];
            T8SMemI8Tile dValFragE4[4];
        };
        struct {
            T8SMemI8Tile qvFragE4T[8]; // 128 dim, to 128 samples
            T8SMemI8Tile qvFrag[2];
            T8SMemI8Tile vFragE4[2];
            i8 dDotTile[2][4][16][16];
        };
        struct {
            float dDotMax[64];
            float dDot8Scale[64];
        };
        float dQK[64][128];
    };
};


__global__ void Fp8AttGradQK(float *qvGlobalScale, TCuda2DPtr<e4m3> qk, TCuda2DPtr<e4m3> qv, TCuda2DPtr<float> qvScale,
    TCuda2DPtr<e4m3> vE4, TCuda2DPtr<e4m3> qvE4T,
    TCuda2DPtr<e4m3> dValLookupE4, TCuda2DPtr<float> dValLookupE4Scale,
    TCuda2DPtr<float> dScaleArr, TCuda2DPtr<float> sumWeightLog,
    TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans2, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
    float attGradMult,
    TCuda2DPtr<TRopeFloat> ropeBuf,
    TCuda2DPtr<float> qkScaleForNormalizeBackprop, TCuda2DPtr<half> dQKres)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    CUDA_STATIC_ASSERT(Q_DIM == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 128);
    TTileCoord tc;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(Q_DIM) * CalcAttentionMult() * *qvGlobalScale;

    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int fromBase = attBlock * ATT_GROUP;
    int fromWarp = warpId * TILE;

    __shared__ TFp8AttGradQKData data;

    // load qv to shmem
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        Copy8Tile(&data.qkFrag[fromTile], warpId, qk.Fragment(eOffset, fromBase + fromTile * TILE));
        Copy8Tile(&data.dValFragE4[fromTile], warpId, dValLookupE4.Fragment(eOffset, fromBase + fromTile * TILE));
    }
    __syncthreads();
    TRegTile<i8> qkFrag[8];
    TRegTile<i8> dValFragE4[8];
    for (int k = 0; k < 8; ++k) {
        qkFrag[k] = TMmaColMajor::FragB(data.qkFrag[warpId], k);
        dValFragE4[k] = TMmaColMajor::FragB(data.dValFragE4[warpId], k);
    }
    __syncthreads();

    // precompute alibi
    float elemAlibi[TTileCoord::num_elements];
    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
        elemAlibi[elem] = GetAttentionDecay(y - x, alibiSlope) - sumWeightLog[head][fromBase + fromWarp + y];
        });

    TRegTileRow<float> dWmultRow;
    dWmultRow.Load(tc, dValLookupE4Scale[head] + fromBase + fromWarp);
    dWmultRow.Scale(V_VEC_SCALE * attDotScale * LOG2);

    TRegTileRow<float> dScaleRow;
    dScaleRow.Load(tc, dScaleArr[head] + fromBase + fromWarp);
    dScaleRow.Scale(attDotScale * LOG2);

    // to in 128 sample blocks, compute just dDot row max
    TRegTileRow<float> rowMax;
    rowMax.SetMax(1e-10f);
    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
        TRegTileRow<int> rowSpanStart;
        TRegTileRow<int> rowSpanFinish;
        rowSpanStart.Load(tc, gg.SpanStart + fromWarp);
        rowSpanFinish.Load(tc, gg.SpanFinish + fromWarp);
        rowSpanStart.Add(-gg.Start);
        rowSpanFinish.Add(-gg.Start);
        int alibiFromMinusTo = fromBase + fromWarp - gg.Start;
        for (int toBase = gg.Start, ggFinish = gg.Finish; toBase <= ggFinish; toBase += MM_TILE) {
            Copy8TileAsync(&data.qvFrag[0], warpId, qv.Fragment(eOffset, toBase + 0));
            Copy8TileAsync(&data.vFragE4[0], warpId, vE4.Fragment(eOffset, toBase + 0));
            for (int toTile = 0; toTile < 8; ++toTile) {
                // load qk & dValFrag
                WaitAsyncCopy();
                __syncthreads();
                if (toTile < 8 - 1) {
                    Copy8TileAsync(&data.qvFrag[(toTile + 1) & 1], warpId, qv.Fragment(eOffset, toBase + (toTile + 1) * TILE));
                    Copy8TileAsync(&data.vFragE4[(toTile + 1) & 1], warpId, vE4.Fragment(eOffset, toBase + (toTile + 1) * TILE));
                }

                // qqTile[from][to] = dot(qk[from], qv[to])
                TRegTile<float> qqTile;
                qqTile.Clear();
                for (int k = 0; k < 8; k += 2) {
                    TRegTile<i8> b1 = TMmaColMajor::FragB(data.qvFrag[toTile & 1], k);
                    TRegTile<i8> b2 = TMmaColMajor::FragB(data.qvFrag[toTile & 1], k + 1);
                    MMAe4e4(&qqTile, qkFrag[k], qkFrag[k + 1], b1, b2);
                }

                // dW[from][to] = dot(dValLookup[from], v[to])
                TRegTile<float> dWTile;
                dWTile.Clear();
                for (int k = 0; k < 8; k += 2) {
                    TRegTile<i8> b1 = TMmaColMajor::FragB(data.vFragE4[toTile & 1], k);
                    TRegTile<i8> b2 = TMmaColMajor::FragB(data.vFragE4[toTile & 1], k + 1);
                    MMAe4e4(&dWTile, dValFragE4[k], dValFragE4[k + 1], b1, b2);
                }

                TRegTileColumn<float> qvScaleColumn;
                qvScaleColumn.Load(tc, qvScale[head] + toBase + toTile * TILE);

                float tileAlibi = GetAttentionDecay(alibiFromMinusTo, alibiSlope);
                TRegTileColumn<float> qqScaleColumn = qvScaleColumn;
                qqScaleColumn.Scale(QK_VEC_SCALE * attDotScale); // qkScale is constant VEC_SCALE

                tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                    //int glFrom = fromBase + fromWarp + y;
                    //int glTo = toBase + toTile * TILE + x;
                    //if (glTo >= gg.SpanStart[from] && glTo <= gg.SpanFinish[from]) {
                    //    float dp = qqTile.x[elem] * qkScale[head][glFrom] * qvScale[head][glTo] * attDotScale;
                    //    dp += GetAttentionDecay(glFrom - glTo, alibiSlope);
                    //    float w = exp2f(dp - sumWeightLog[head][glFrom]);
                    //    float dWmult = VEC_SCALE *dValLookupE4Scale[head][glFrom];
                    //    float dDot = w * (dWTile.x[elem] * dWmult - dScaleArr[head][glFrom]) * attDotScale * LOG2; // log(2) from using exp2() instread of exp() 
                    //    dDot *= qvScale[head][glTo]; // apply scale
                    //}
                    if (x >= rowSpanStart.x[rowIndex] && x <= rowSpanFinish.x[rowIndex]) {
                        float dp = qqTile.x[elem] * qqScaleColumn.x[columnIndex];
                        dp += tileAlibi + elemAlibi[elem];
                        float w = exp2f(dp);
                        float dDot = w * (dWTile.x[elem] * dWmultRow.x[rowIndex] - dScaleRow.x[rowIndex]);
                        dDot *= qvScaleColumn.x[columnIndex];
                        rowMax.x[rowIndex] = max(rowMax.x[rowIndex], fabs(dDot));
                    }
                    });

                rowSpanStart.Add(-TILE);
                rowSpanFinish.Add(-TILE);
                alibiFromMinusTo -= TILE;
            }
        }
    }
    rowMax.WarpMaxReduce();

    TRegTile<float> dQK[4][2];
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dQK[fromTile][sumX].Clear();
        }
    }

    // to in 128 sample blocks
    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
        TRegTileRow<int> rowSpanStart;
        TRegTileRow<int> rowSpanFinish;
        rowSpanStart.Load(tc, gg.SpanStart + fromWarp);
        rowSpanFinish.Load(tc, gg.SpanFinish + fromWarp);
        rowSpanStart.Add(-gg.Start);
        rowSpanFinish.Add(-gg.Start);
        int alibiFromMinusTo = fromBase + fromWarp - gg.Start;
        for (int toBase = gg.Start, ggFinish = gg.Finish; toBase <= ggFinish; toBase += MM_TILE) {
            // load transposed qk / dVal for sum
            __syncthreads();
            for (int k = 0; k < 8; ++k) {
                Copy8Tile(&data.qvFragE4T[k], warpId, qvE4T.Fragment(toBase, eOffset + k * TILE));
            }
            __syncthreads();

            Copy8TileAsync(&data.qvFrag[0], warpId, qv.Fragment(eOffset, toBase + 0));
            Copy8TileAsync(&data.vFragE4[0], warpId, vE4.Fragment(eOffset, toBase + 0));
            for (int toTileBlkStart = 0; toTileBlkStart < 8; toTileBlkStart += 2) {
                for (int blkPos = 0; blkPos < 2; ++blkPos) {
                    int toTile = toTileBlkStart + blkPos;
                    // load qk & dValFrag
                    WaitAsyncCopy();
                    __syncthreads();
                    if (toTile < 8 - 1) {
                        Copy8TileAsync(&data.qvFrag[(toTile + 1) & 1], warpId, qv.Fragment(eOffset, toBase + (toTile + 1) * 16));
                        Copy8TileAsync(&data.vFragE4[(toTile + 1) & 1], warpId, vE4.Fragment(eOffset, toBase + (toTile + 1) * 16));
                    }

                    // qqTile[from][to] = dot(qk[from], qv[to])
                    TRegTile<float> qqTile;
                    qqTile.Clear();
                    for (int k = 0; k < 8; k += 2) {
                        TRegTile<i8> b1 = TMmaColMajor::FragB(data.qvFrag[toTile & 1], k);
                        TRegTile<i8> b2 = TMmaColMajor::FragB(data.qvFrag[toTile & 1], k + 1);
                        MMAe4e4(&qqTile, qkFrag[k], qkFrag[k + 1], b1, b2);
                    }

                    // dW[from][to] = dot(dValLookup[from], v[to])
                    TRegTile<float> dWTile;
                    dWTile.Clear();
                    for (int k = 0; k < 8; k += 2) {
                        TRegTile<i8> b1 = TMmaColMajor::FragB(data.vFragE4[toTile & 1], k);
                        TRegTile<i8> b2 = TMmaColMajor::FragB(data.vFragE4[toTile & 1], k + 1);
                        MMAe4e4(&dWTile, dValFragE4[k], dValFragE4[k + 1], b1, b2);
                    }

                    TRegTileColumn<float> qvScaleColumn;
                    qvScaleColumn.Load(tc, qvScale[head] + toBase + toTile * TILE);

                    float tileAlibi = GetAttentionDecay(alibiFromMinusTo, alibiSlope);
                    TRegTileColumn<float> qqScaleColumn = qvScaleColumn;
                    qqScaleColumn.Scale(QK_VEC_SCALE *attDotScale); // qkScale is constant VEC_SCALE

                    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                        if (x >= rowSpanStart.x[rowIndex] && x <= rowSpanFinish.x[rowIndex]) {
                            float dp = qqTile.x[elem] * qqScaleColumn.x[columnIndex];
                            dp += tileAlibi + elemAlibi[elem];
                            float w = exp2f(dp);
                            float dDot = w * (dWTile.x[elem] * dWmultRow.x[rowIndex] - dScaleRow.x[rowIndex]);
                            dDot *= qvScaleColumn.x[columnIndex];
                            float dDot8Scale = (rowMax.x[rowIndex] / FP8_ATT_DOT_MULT);
                            data.dDotTile[blkPos][warpId][y][x] = CvtToE4(dDot / dDot8Scale);
                        } else {
                            data.dDotTile[blkPos][warpId][y][x] = 0;
                        }
                        });

                    rowSpanStart.Add(-TILE);
                    rowSpanFinish.Add(-TILE);
                    alibiFromMinusTo -= TILE;
                }
                __syncthreads();

                // add to dQK 
                // dQK[from][x] += dDot[from][to] @ qv[to][x];
                TRegTile<i8> b1[2];
                b1[0] = TMmaColMajor::FragB(data.qvFragE4T[warpId * 2 + 0], toTileBlkStart);
                b1[1] = TMmaColMajor::FragB(data.qvFragE4T[warpId * 2 + 1], toTileBlkStart);
                TRegTile<i8> b2[2];
                b2[0] = TMmaColMajor::FragB(data.qvFragE4T[warpId * 2 + 0], toTileBlkStart + 1);
                b2[1] = TMmaColMajor::FragB(data.qvFragE4T[warpId * 2 + 1], toTileBlkStart + 1);
                for (int fromTile = 0; fromTile < 4; ++fromTile) {
                    TRegTile<i8> dDotTile1;
                    LoadFromSmem(&dDotTile1, TCuda2DPtr<i8>(data.dDotTile[0][fromTile], 16, 16, 16));
                    TRegTile<i8> dDotTile2;
                    LoadFromSmem(&dDotTile2, TCuda2DPtr<i8>(data.dDotTile[1][fromTile], 16, 16, 16));
                    for (int sumX = 0; sumX < 2; ++sumX) {
                        MMAe4e4(&dQK[fromTile][sumX],
                            TMmaRowMajor::FragA(dDotTile1), TMmaRowMajor::FragA(dDotTile2),
                            b1[sumX], b2[sumX]);
                    }
                }
            }
        }
    }
    __syncthreads();
    rowMax.StoreMax(tc, data.dDotMax + fromWarp);
    __syncthreads();
    int h = threadIdx.x;
    if (h < TILE) {
        float dDot8Scale = (data.dDotMax[fromWarp + h] / FP8_ATT_DOT_MULT);
        data.dDot8Scale[fromWarp + h] = dDot8Scale;
    }
    __syncthreads();

    // scale result
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            TRegTile<float> &rsum = dQK[fromTile][sumX];
            TRegTileRow<float> dot8Scale;
            dot8Scale.Load(tc, data.dDot8Scale + fromTile * TILE);
            dot8Scale.Scale(attGradMult);
            rsum.Scale(dot8Scale);
            //rsum.Store(tc, dQKres.Fragment(eOffset + (warpId * 2 + sumX) * TILE, fromBase + fromTile * TILE));
        }
    }

    // backprop qk rope and normalize
    __syncthreads();
    for (int fromTile = 0; fromTile < 4; ++fromTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dQK[fromTile][sumX].Store(tc, TCuda2DPtr<float>(data.dQK[fromTile * TILE] + (warpId * 2 + sumX) * TILE, 128 * sizeof(float), 16, 16));
        }
    }
    __syncthreads();
    for (int base = 0; base < 64; base += 4) {
        int y = base + warpId;
        constexpr int WSZ = MM_TILE / WARP_SIZE;

        float dQKvec[WSZ];
        LoadWarpVec<WSZ>(dQKvec, data.dQK[y]);

        // backprop normalize (reverse backprop order to get correct results since only rotated qk is kept)
        float qkVec[WSZ];
        LoadWarpVec<WSZ>(qkVec, qk[fromBase + y] + eOffset);
        ScaleWarpVec<WSZ>(qkVec, qkScaleForNormalizeBackprop[head][fromBase + y]);
        StateNormalizeBackpropWarpVec<WSZ>(qkVec, dQKvec, dQKvec);

        // backprop rope
        float rope[WSZ];
        LoadWarpVec<WSZ>(rope, ropeBuf[fromBase + y]);
        ApplyWarpRopeImpl<WSZ>(rope, -1, dQKvec);

        StoreWarpVec<WSZ>(dQKres[fromBase + y] + eOffset, dQKvec);
    }
}
KERNEL_BLOCK_SIZE(Fp8AttGradQK, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TFp8AttGradQVData
{
    union {
        struct {
            T8SMemI8Tile qvFrag[4];
            T8SMemI8Tile vFragE4[4];
        };
        struct {
            T8SMemI8Tile dValFragE4T[8]; // 128 dim, to 128 samples
            T8SMemI8Tile qkFragE4T[8]; // 128 dim, to 128 samples
            T8SMemI8Tile qkFrag[2];
            T8SMemI8Tile dValFragE4[2];
            i8 dValMultT[2][4][16][16];
            i8 dDotTileT[2][4][16][16];
        };
        struct {
            float dDotMax[64];
            float dDot8Scale[64];
            float dValMax[64];
            float dVal8Scale[64];
        };
        float dV[64][128];
        float dQV[64][128];
    };
};


__global__ void Fp8AttGradQV(float *qvGlobalScale, TCuda2DPtr<e4m3> qk, TCuda2DPtr<e4m3> qv, TCuda2DPtr<float> qvScale,
    TCuda2DPtr<e4m3> vE4, TCuda2DPtr<e4m3> qkE4T,
    TCuda2DPtr<e4m3> dValLookupE4, TCuda2DPtr<e4m3> dValLookupE4T, TCuda2DPtr<float> dValLookupE4Scale,
    TCuda2DPtr<float> dScaleArr, TCuda2DPtr<float> sumWeightLog,
    TCuda1DPtr<TAttentionSpanGroup<ATT_GROUP>> attSpans2, TCuda1DPtr<int> attSpanPtr, float alibiSlope,
    float attGradMult,
    TCuda2DPtr<TRopeFloat> ropeBuf,
    TCuda2DPtr<float> vScaleForNormalizeBackprop, TCuda2DPtr<half> dVres, TCuda2DPtr<half> dQVres)
{
    CUDA_STATIC_ASSERT(MM_TILE == 128);
    CUDA_STATIC_ASSERT(Q_DIM == 128);
    CUDA_ASSERT(ATT_GROUP == 64);
    CUDA_ASSERT(ATT_ALIGN == 128);
    TTileCoord tc;

    // blockIdx.x - head group
    // blockIdx.y - len

    int head = blockIdx.x;
    float attDotScale = CalcDotScale(Q_DIM) * CalcAttentionMult() * *qvGlobalScale;

    int warpId = threadIdx.y;
    int eOffset = head * MM_TILE;
    int attBlock = blockIdx.y;
    int toBase = attBlock * ATT_GROUP;
    int toWarp = warpId * TILE;

    __shared__ TFp8AttGradQVData data;

    // load qv to shmem
    for (int toTile = 0; toTile < 4; ++toTile) {
        Copy8Tile(&data.qvFrag[toTile], warpId, qv.Fragment(eOffset, toBase + toTile * TILE));
        Copy8Tile(&data.vFragE4[toTile], warpId, vE4.Fragment(eOffset, toBase + toTile * TILE));
    }
    __syncthreads();
    TRegTile<i8> qvFrag[8];
    TRegTile<i8> vFragE4[8];
    for (int k = 0; k < 8; ++k) {
        qvFrag[k] = TMmaColMajor::FragB(data.qvFrag[warpId], k);
        vFragE4[k] = TMmaColMajor::FragB(data.vFragE4[warpId], k);
    }
    __syncthreads();

    TRegTileColumn<float> qqScaleColumn;
    qqScaleColumn.Load(tc, qvScale[head] + toBase + toWarp);
    qqScaleColumn.Scale(QK_VEC_SCALE * attDotScale); // qkScale is constant VEC_SCALE

    // from in 128 sample blocks, compute just dDot row max
    TRegTileColumn<float> columnDDotMax;
    columnDDotMax.SetMax(1e-10f);
    TRegTileColumn<float> columnDValMultMax;
    columnDValMultMax.SetMax(1e-10f);
    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
        TRegTileColumn<int> columnSpanStart;
        TRegTileColumn<int> columnSpanFinish;
        columnSpanStart.Load(tc, gg.SpanStart + toWarp);
        columnSpanFinish.Load(tc, gg.SpanFinish + toWarp);
        columnSpanStart.Add(-gg.Start);
        columnSpanFinish.Add(-gg.Start);
        int alibiFromMinusTo = gg.Start - toBase - toWarp;
        for (int fromBase = gg.Start, ggFinish = gg.Finish; fromBase <= ggFinish; fromBase += MM_TILE) {
            Copy8TileAsync(&data.qkFrag[0], warpId, qk.Fragment(eOffset, fromBase + 0));
            Copy8TileAsync(&data.dValFragE4[0], warpId, dValLookupE4.Fragment(eOffset, fromBase + 0));
            for (int fromTile = 0; fromTile < 8; ++fromTile) {
                // load qk & dValFrag
                WaitAsyncCopy();
                __syncthreads();
                if (fromTile < 8 - 1) {
                    Copy8TileAsync(&data.qkFrag[(fromTile + 1) & 1], warpId, qk.Fragment(eOffset, fromBase + (fromTile + 1) * TILE));
                    Copy8TileAsync(&data.dValFragE4[(fromTile + 1) & 1], warpId, dValLookupE4.Fragment(eOffset, fromBase + (fromTile + 1) * TILE));
                }

                // qqTile[from][to] = dot(qk[from], qv[to])
                TRegTile<float> qqTile;
                qqTile.Clear();
                for (int k = 0; k < 8; k += 2) {
                    TRegTile<i8> a1 = TMmaRowMajor::FragA(data.qkFrag[fromTile & 1], k);
                    TRegTile<i8> a2 = TMmaRowMajor::FragA(data.qkFrag[fromTile & 1], k + 1);
                    MMAe4e4(&qqTile, a1, a2, qvFrag[k], qvFrag[k + 1]);
                }

                // dW[from][to] = dot(dValLookup[from], v[to])
                TRegTile<float> dWTile;
                dWTile.Clear();
                for (int k = 0; k < 8; k += 2) {
                    TRegTile<i8> a1 = TMmaRowMajor::FragA(data.dValFragE4[fromTile & 1], k);
                    TRegTile<i8> a2 = TMmaRowMajor::FragA(data.dValFragE4[fromTile & 1], k + 1);
                    MMAe4e4(&dWTile, a1, a2, vFragE4[k], vFragE4[k + 1]);
                }

                float tileAlibi = GetAttentionDecay(alibiFromMinusTo, alibiSlope);

                TRegTileRow<float> sumWeightLogRow;
                sumWeightLogRow.Load(tc, sumWeightLog[head] + fromBase + fromTile * TILE);

                TRegTileRow<float> dValLookupScaleRow;
                dValLookupScaleRow.Load(tc, dValLookupE4Scale[head] + fromBase + fromTile * TILE);

                TRegTileRow<float> dWmultRow = dValLookupScaleRow;
                dWmultRow.Scale(V_VEC_SCALE * attDotScale * LOG2);

                TRegTileRow<float> dScaleRow;
                dScaleRow.Load(tc, dScaleArr[head] + fromBase + fromTile * TILE);
                dScaleRow.Scale(attDotScale * LOG2);

                TRegTileRow<float> qkScaleRow;
                //qkScaleRow.Load(tc, qkScale[head] + fromBase + fromTile * TILE);
                qkScaleRow.FillEvery(QK_VEC_SCALE); // qkScale is constant VEC_SCALE
                dWmultRow.Scale(qkScaleRow);
                dScaleRow.Scale(qkScaleRow);

                float elemAlibi[TTileCoord::num_elements];
                tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                    elemAlibi[elem] = GetAttentionDecay(y - x, alibiSlope) + tileAlibi - sumWeightLogRow.x[rowIndex];
                    });

                tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                    if (y >= columnSpanStart.x[columnIndex] && y <= columnSpanFinish.x[columnIndex]) {
                        float dp = qqTile.x[elem] * qqScaleColumn.x[columnIndex];
                        dp += elemAlibi[elem];
                        float w = exp2f(dp);
                        float dDot = w * (dWTile.x[elem] * dWmultRow.x[rowIndex] - dScaleRow.x[rowIndex]);
                        columnDDotMax.x[columnIndex] = max(columnDDotMax.x[columnIndex], fabs(dDot));
                        columnDValMultMax.x[columnIndex] = fmaxf(columnDValMultMax.x[columnIndex], fabs(w * dValLookupScaleRow.x[rowIndex]));
                    }
                    //int glFrom = fromBase + fromTile * TILE + y;
                    //int glTo = toBase + toWarp + x;
                    //int to = toWarp + x;
                    //if (glFrom >= gg.SpanStart[to] && glFrom <= gg.SpanFinish[to]) {
                    //     lookup
                    //    float dp = qqTile.x[elem] * qkScale[head][glFrom] * qvScale[head][glTo] * attDotScale;
                    //    dp += GetAttentionDecay(glFrom - glTo, alibiSlope);
                    //    float w = exp2f(dp - sumWeightLog[head][glFrom]);
                    //    float dWmult = VEC_SCALE * dValLookupE4Scale[head][glFrom];
                    //    float dDot = w * (dWTile.x[elem] * dWmult - dScaleArr[head][glFrom]) * attDotScale * LOG2; // log(2) from using exp2() instread of exp()
                    //    dDot *= qkScale[head][glFrom]; // apply scale
                    //    columnDDotMax.x[columnIndex] = max(columnDDotMax.x[columnIndex], fabs(dDot));
                    //    columnDValMultMax.x[columnIndex] = fmaxf(columnDValMultMax.x[columnIndex], fabs(w * dValLookupE4Scale[head][glFrom]));
                    //}
                    });

                columnSpanStart.Add(-TILE);
                columnSpanFinish.Add(-TILE);
                alibiFromMinusTo += TILE;
            }
        }
    }
    columnDDotMax.WarpMaxReduce();
    columnDValMultMax.WarpMaxReduce();

    TRegTile<float> dV[4][2];
    for (int toTile = 0; toTile < 4; ++toTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dV[toTile][sumX].Clear();
        }
    }

    TRegTile<float> dQV[4][2];
    for (int toTile = 0; toTile < 4; ++toTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dQV[toTile][sumX].Clear();
        }
    }

    // from in 128 sample blocks
    for (int attIndex = attSpanPtr[attBlock]; attIndex < attSpanPtr[attBlock + 1]; ++attIndex) {
        const TAttentionSpanGroup<ATT_GROUP> &gg = attSpans2[attIndex];
        TRegTileColumn<int> columnSpanStart;
        TRegTileColumn<int> columnSpanFinish;
        columnSpanStart.Load(tc, gg.SpanStart + toWarp);
        columnSpanFinish.Load(tc, gg.SpanFinish + toWarp);
        columnSpanStart.Add(-gg.Start);
        columnSpanFinish.Add(-gg.Start);
        int alibiFromMinusTo = gg.Start - toBase - toWarp;
        for (int fromBase = gg.Start, ggFinish = gg.Finish; fromBase <= ggFinish; fromBase += MM_TILE) {
            // load transposed qk / dVal for sum
            __syncthreads();
            for (int k = 0; k < 8; ++k) {
                Copy8Tile(&data.qkFragE4T[k], warpId, qkE4T.Fragment(fromBase, eOffset + k * TILE));
                Copy8Tile(&data.dValFragE4T[k], warpId, dValLookupE4T.Fragment(fromBase, eOffset + k * TILE));
            }
            __syncthreads();

            Copy8TileAsync(&data.qkFrag[0], warpId, qk.Fragment(eOffset, fromBase + 0));
            Copy8TileAsync(&data.dValFragE4[0], warpId, dValLookupE4.Fragment(eOffset, fromBase + 0));
            for (int fromTileBlkStart = 0; fromTileBlkStart < 8; fromTileBlkStart += 2) {
                for (int blkPos = 0; blkPos < 2; ++blkPos) {
                    int fromTile = fromTileBlkStart + blkPos;
                    // load qk & dValFrag
                    WaitAsyncCopy();
                    __syncthreads();
                    if (fromTile < 8 - 1) {
                        Copy8TileAsync(&data.qkFrag[(fromTile + 1) & 1], warpId, qk.Fragment(eOffset, fromBase + (fromTile + 1) * TILE));
                        Copy8TileAsync(&data.dValFragE4[(fromTile + 1) & 1], warpId, dValLookupE4.Fragment(eOffset, fromBase + (fromTile + 1) * TILE));
                    }

                    // qqTile[from][to] = dot(qk[from], qv[to])
                    TRegTile<float> qqTile;
                    qqTile.Clear();
                    for (int k = 0; k < 8; k += 2) {
                        TRegTile<i8> a1 = TMmaRowMajor::FragA(data.qkFrag[fromTile & 1], k);
                        TRegTile<i8> a2 = TMmaRowMajor::FragA(data.qkFrag[fromTile & 1], k + 1);
                        MMAe4e4(&qqTile, a1, a2, qvFrag[k], qvFrag[k + 1]);
                    }

                    // dW[from][to] = dot(dValLookup[from], v[to])
                    TRegTile<float> dWTile;
                    dWTile.Clear();
                    for (int k = 0; k < 8; k += 2) {
                        TRegTile<i8> a1 = TMmaRowMajor::FragA(data.dValFragE4[fromTile & 1], k);
                        TRegTile<i8> a2 = TMmaRowMajor::FragA(data.dValFragE4[fromTile & 1], k + 1);
                        MMAe4e4(&dWTile, a1, a2, vFragE4[k], vFragE4[k + 1]);
                    }

                    float tileAlibi = GetAttentionDecay(alibiFromMinusTo, alibiSlope);

                    TRegTileRow<float> sumWeightLogRow;
                    sumWeightLogRow.Load(tc, sumWeightLog[head] + fromBase + fromTile * TILE);

                    TRegTileRow<float> dValLookupScaleRow;
                    dValLookupScaleRow.Load(tc, dValLookupE4Scale[head] + fromBase + fromTile * TILE);

                    TRegTileRow<float> dWmultRow = dValLookupScaleRow;
                    dWmultRow.Scale(V_VEC_SCALE *attDotScale *LOG2);

                    TRegTileRow<float> dScaleRow;
                    dScaleRow.Load(tc, dScaleArr[head] + fromBase + fromTile * TILE);
                    dScaleRow.Scale(attDotScale *LOG2);

                    TRegTileRow<float> qkScaleRow;
                    //qkScaleRow.Load(tc, qkScale[head] + fromBase + fromTile * TILE);
                    qkScaleRow.FillEvery(QK_VEC_SCALE); // qkScale is constant VEC_SCALE
                    dWmultRow.Scale(qkScaleRow);
                    dScaleRow.Scale(qkScaleRow);

                    float elemAlibi[TTileCoord::num_elements];
                    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                        elemAlibi[elem] = GetAttentionDecay(y - x, alibiSlope) + tileAlibi - sumWeightLogRow.x[rowIndex];
                        });

                    tc.EnumElements([&](int elem, int x, int y, int rowIndex, int columnIndex) {
                        if (y >= columnSpanStart.x[columnIndex] && y <= columnSpanFinish.x[columnIndex]) {
                            float dp = qqTile.x[elem] * qqScaleColumn.x[columnIndex];
                            dp += elemAlibi[elem];
                            float w = exp2f(dp);
                            float dDot = w * (dWTile.x[elem] * dWmultRow.x[rowIndex] - dScaleRow.x[rowIndex]);

                            float dDot8Scale = (columnDDotMax.x[columnIndex] / FP8_ATT_DOT_MULT);
                            data.dDotTileT[blkPos][warpId][x][y] = CvtToE4(dDot / dDot8Scale);

                            float dVal8Scale = (columnDValMultMax.x[columnIndex] / FP8_ATT_DOT_MULT);
                            data.dValMultT[blkPos][warpId][x][y] = CvtToE4((w * dValLookupScaleRow.x[rowIndex]) / dVal8Scale);
                        } else {
                            data.dDotTileT[blkPos][warpId][x][y] = 0;
                            data.dValMultT[blkPos][warpId][x][y] = 0;
                        }
                        });
                    columnSpanStart.Add(-TILE);
                    columnSpanFinish.Add(-TILE);
                    alibiFromMinusTo += TILE;
                }
                __syncthreads();

                // add to dV
                // dV[to][x] += w[from][to] @ dValLookup[from][x];
                TRegTile<i8> b1[2];
                TRegTile<i8> b2[2];
                b1[0] = TMmaColMajor::FragB(data.dValFragE4T[warpId * 2 + 0], fromTileBlkStart);
                b1[1] = TMmaColMajor::FragB(data.dValFragE4T[warpId * 2 + 1], fromTileBlkStart);
                b2[0] = TMmaColMajor::FragB(data.dValFragE4T[warpId * 2 + 0], fromTileBlkStart + 1);
                b2[1] = TMmaColMajor::FragB(data.dValFragE4T[warpId * 2 + 1], fromTileBlkStart + 1);
                for (int toTile = 0; toTile < 4; ++toTile) {
                    TRegTile<i8> dValMultT1;
                    LoadFromSmem(&dValMultT1, TCuda2DPtr<i8>(data.dValMultT[0][toTile], 16, 16, 16));
                    TRegTile<i8> dValMultT2;
                    LoadFromSmem(&dValMultT2, TCuda2DPtr<i8>(data.dValMultT[1][toTile], 16, 16, 16));
                    for (int sumX = 0; sumX < 2; ++sumX) {
                        MMAe4e4(&dV[toTile][sumX],
                            TMmaRowMajor::FragA(dValMultT1), TMmaRowMajor::FragA(dValMultT2),
                            b1[sumX], b2[sumX]);
                    }
                }

                // add to dQV 
                // dQV[to][x] += dDot[from][to] @ qk[from][x];
                b1[0] = TMmaColMajor::FragB(data.qkFragE4T[warpId * 2 + 0], fromTileBlkStart);
                b1[1] = TMmaColMajor::FragB(data.qkFragE4T[warpId * 2 + 1], fromTileBlkStart);
                b2[0] = TMmaColMajor::FragB(data.qkFragE4T[warpId * 2 + 0], fromTileBlkStart + 1);
                b2[1] = TMmaColMajor::FragB(data.qkFragE4T[warpId * 2 + 1], fromTileBlkStart + 1);
                for (int toTile = 0; toTile < 4; ++toTile) {
                    TRegTile<i8> dDotTileT1;
                    LoadFromSmem(&dDotTileT1, TCuda2DPtr<i8>(data.dDotTileT[0][toTile], 16, 16, 16));
                    TRegTile<i8> dDotTileT2;
                    LoadFromSmem(&dDotTileT2, TCuda2DPtr<i8>(data.dDotTileT[1][toTile], 16, 16, 16));
                    for (int sumX = 0; sumX < 2; ++sumX) {
                        MMAe4e4(&dQV[toTile][sumX],
                            TMmaRowMajor::FragA(dDotTileT1), TMmaRowMajor::FragA(dDotTileT2),
                            b1[sumX], b2[sumX]);
                    }
                }
            }
        }
    }
    __syncthreads();
    columnDDotMax.StoreMax(tc, data.dDotMax + toWarp);
    columnDValMultMax.StoreMax(tc, data.dValMax + toWarp);
    __syncthreads();
    int h = threadIdx.x;
    if (h < TILE) {
        int to = toWarp + h;
        float dDot8Scale = (data.dDotMax[to] / FP8_ATT_DOT_MULT);
        data.dDot8Scale[to] = dDot8Scale;
        float dVal8Scale = (data.dValMax[to] / FP8_ATT_DOT_MULT);
        data.dVal8Scale[to] = dVal8Scale;
    }
    __syncthreads();

    // save dQV and scale qV
    for (int toTile = 0; toTile < 4; ++toTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            //int xOffset = eOffset + (warpId * 2 + sumX) * TILE;
            //int yOffset = toBase + toTile * TILE;
            {
                TRegTile<float> &rsum = dQV[toTile][sumX];
                TRegTileRow<float> dot8Scale;
                dot8Scale.Load(tc, data.dDot8Scale + toTile * TILE);
                dot8Scale.Scale(attGradMult);
                rsum.Scale(dot8Scale);
                //rsum.Store(tc, dQVres.Fragment(xOffset, yOffset));
            }
            {
                TRegTile<float> &rsum = dV[toTile][sumX];
                TRegTileRow<float> val8Scale;
                val8Scale.Load(tc, data.dVal8Scale + toTile * TILE);
                val8Scale.Scale(attGradMult);
                rsum.Scale(val8Scale);
                //rsum.Store(tc, dVres.Fragment(xOffset, yOffset));
            }
        }
    }

    // backprop qv rope
    __syncthreads();
    for (int toTile = 0; toTile < 4; ++toTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dQV[toTile][sumX].Store(tc, TCuda2DPtr<float>(data.dQV[toTile * TILE] + (warpId * 2 + sumX) * TILE, 128 * sizeof(float), 16, 16));
        }
    }
    __syncthreads();
    for (int base = 0; base < 64; base += 4) {
        int y = base + warpId;
        constexpr int WSZ = MM_TILE / WARP_SIZE;

        float dQVvec[WSZ];
        LoadWarpVec<WSZ>(dQVvec, data.dQV[y]);
        float rope[WSZ];
        LoadWarpVec<WSZ>(rope, ropeBuf[toBase + y]);
        ApplyWarpRopeImpl<WSZ>(rope, -1, dQVvec);
        StoreWarpVec<WSZ>(dQVres[toBase + y] + eOffset, dQVvec);
    }

    // backprop v normalize
    __syncthreads();
    for (int toTile = 0; toTile < 4; ++toTile) {
        for (int sumX = 0; sumX < 2; ++sumX) {
            dV[toTile][sumX].Store(tc, TCuda2DPtr<float>(data.dV[toTile * TILE] + (warpId * 2 + sumX) * TILE, 128 * sizeof(float), 16, 16));
        }
    }
    __syncthreads();
    for (int base = 0; base < 64; base += 4) {
        int y = base + warpId;
        constexpr int WSZ = MM_TILE / WARP_SIZE;

        float vVec[WSZ];
        LoadWarpVec<WSZ>(vVec, vE4[toBase + y] + eOffset);
        ScaleWarpVec<WSZ>(vVec, vScaleForNormalizeBackprop[head][toBase + y]);
        float dVvec[WSZ];
        LoadWarpVec<WSZ>(dVvec, data.dV[y]);
        float dVresult[WSZ];
        StateNormalizeBackpropWarpVec<WSZ>(vVec, dVvec, dVresult);
        StoreWarpVec<WSZ>(dVres[toBase + y] + eOffset, dVresult);
    }
}
KERNEL_BLOCK_SIZE(Fp8AttGradQV, WARP_SIZE, 4);


///////////////////////////////////////////////////////////////////////////////////////////////////
// dScale = dot(valLookup, dValLookup), computation is fused into other kernel
