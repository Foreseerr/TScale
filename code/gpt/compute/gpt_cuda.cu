#include "stdafx.h"
#define KERNEL_UNIT "gpt_cuda/"
#include "gpt_cuda.cuh"
#include "cfg_precision.h"
#include "matmul_fwdbwd.cuh"
#include "row_scale.cuh"
#include "row_tile_scale.cuh"
#include <lib/cuda/cuda_util.cuh>
#include <lib/cuda/cuda_graph.cuh>
#include <lib/cuda/cuda_matmul.cuh>
#include <lib/cuda/cuda_mma.cuh>
#include <lib/cuda/cuda_i8.cuh>
#include <lib/cuda/cuda_fp8.cuh>
#include <lib/cuda/cuda_fp16.cuh>
#include <lib/cuda/vec_util.cuh>
#include "par_matrix_cuda.cuh"
#include <gpt/data/data.h>
#include <gpt/att/nodes_batch.h>
#include <gpt/att/rope.h>
#include <lib/random/rand_utils.h>
#include <lib/math/matrix_utils.h>
#include <util/radix_sort.h>
#include <emmintrin.h>

using namespace NCuda;

constexpr bool LOW_VRAM_CONSUMPTION = false;

constexpr int PREDICTION_ARR_SZ = 4096;
constexpr int LAYER_CTX_COUNT = LOW_VRAM_CONSUMPTION ? 1 : 2;
constexpr int ATT_CTX_COUNT = LOW_VRAM_CONSUMPTION ? 1 : 2;
constexpr int Q_DIM = 128;
constexpr int TT_DIM = 128;

namespace NCUDA_GPT
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// config
constexpr float STATE_VEC_SCALE = 1 / 24.0f;

constexpr float QK_VEC_SCALE = 1 / 24.0f;
constexpr float QV_VEC_SCALE = 1 / 24.0f;
constexpr float K_VEC_SCALE = 1 / 24.0f;
constexpr float V_VEC_SCALE = 1 / 24.0f;
constexpr float RELU_VEC_SCALE = 1 / 24.0f;

// element type of state vector
typedef float TStateFloat;
//typedef half TStateFloat;

typedef float TStateGradFloat;
//typedef half TStateGradFloat; // worse models, on the order of 2.576 -> 2.579


///////////////////////////////////////////////////////////////////////////////////////////////////
// utils

// gradient scale
__forceinline __device__ float GetGradScale(float gradMaxNorm)
{
    const float TARGET_MAX_NORM = 128;
    if (gradMaxNorm < TARGET_MAX_NORM / 4 || gradMaxNorm > TARGET_MAX_NORM * 4) {
        if (gradMaxNorm == 0) {
            return 1; // zero gradients can be multiplied by any number
        }
        return TruncateToPow2(TARGET_MAX_NORM / gradMaxNorm);
    }
    return 1;
}


template <class T>
TCuda2DArray<T> &RecastBuf(TCuda2DArray<i8> &buf)
{
    CUDA_STATIC_ASSERT(sizeof(T) == 1);
    return *(TCuda2DArray<T>*) &buf;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// kernels
#include "gpt_rope.cuh"
#include "gpt_att_fp8.cuh"
#include "gpt_att_fp16.cuh"
#include "gpt_combiner.cuh"
#include "gpt_embedding.cuh"
#include "gpt_final.cuh"
#include "gpt_layer_norm.cuh"


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TComputeParams
{
    TOpParameter<int> InvLabelCount;

    int LenBufferSize = 0;
    TOpParameter<int> Len;
    TOpParameter<int> LenRound;
    TOpParameter<int> LenAttTiles;
    TOpParameter<int> LenTiles;
    TOpParameter<int> UsedLabelCount;
    TCudaVector<ui32> DropTable;
    TCudaVector<int> SampleIndex;
    TCuda2DArray<TRopeFloat> RopeBuf;

    void Allocate(TStream &stream, const TModelDescr &modelDescr, int maxLen)
    {
        DropTable.AllocateWC(CalcDropTableSize(modelDescr));
        SampleIndex.AllocateWC(maxLen);
        FillRopeBuf(stream, &RopeBuf, modelDescr.Dims.QDim, maxLen);
    }

    void Init(TStream &stream, yint len, const TVector<int> &sampleIndex, const TVector<ui32> &dropTable, int usedLabelCount)
    {
        Y_ASSERT(MM_TILE >= TILE);
        Len.Set(len);
        yint nTiles = DivCeil(len, MM_TILE);
        LenBufferSize = nTiles * MM_TILE;
        LenRound.Set(LenBufferSize);
        LenAttTiles.Set(nTiles * MM_TILE / ATT_GROUP);
        LenTiles.Set(nTiles);
        UsedLabelCount.Set(usedLabelCount);
        // dropout
        Y_VERIFY(YSize(dropTable) <= DropTable.GetSize());
        Put(stream, &DropTable, dropTable);
        //
        Put(stream, &SampleIndex, sampleIndex);
    }

    void SetInvLabelCount(yint invLabelCount)
    {
        InvLabelCount.Set(invLabelCount);
    }

    yint GetLenBufferSize() const { return LenBufferSize; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// AddProduct implementation

struct TCudaAttentionParams : public TThrRefBase
{
    TVector<TIntrusivePtr<TCudaModelMatrix<TFastModelFloat>>> MatrArr;
    float AlibiSlope = 0;
    yint AttentionWidthId = 0;

    TCudaAttentionParams(yint deviceId, TIntrusivePtr<TCudaModelMatrixScale> cudaMatrixScale, const TAttentionParams &att)
    {
        yint count = YSize(att.MatrArr);
        MatrArr.resize(count);
        for (yint k = 0; k < count; ++k) {
            MatrArr[k] = new TCudaModelMatrix<TFastModelFloat>(deviceId, cudaMatrixScale, att.MatrArr[k], MM_MEM_DEVICE);
        }
        AlibiSlope = att.AlibiSlope;
        AttentionWidthId = att.AttentionWidthId;
    }
    void CopyToDevice(TIntrusivePtr<NCuda::TGraph> c)
    {
        for (auto &mm : MatrArr) {
            mm->CopyToDevice(c);
        }
    }
};


struct TAttentionComputeCtx
{
    TModelDims CurDims;
    // QK
    TCuda2DArray<float> QKscale;
    TCuda2DArray<TAttVecFloat> QK;
    TCuda2DArray<TAttVecFloat> QKT;
    // QV
    TCuda2DArray<float> QVscale;
    TCuda2DArray<TAttVecFloat> QV;
    TCuda2DArray<TAttVecFloat> QVT;
    // K
    TCuda2DArray<half> K16;
    TCuda2DArray<float> Kscale;
    // V
    TCuda2DArray<float> Vscale;
    TCuda2DArray<TAttVecFloat> V;
    TCuda2DArray<TAttVecFloat> VT;
    // rest
    TCuda2DArray<TNormStateFloat> KV;
    TCuda2DArray<half> ValLookup;
    // relu
    TCuda2DArray<TNormStateFloat> Relu;
    TCuda2DArray<TNormStateFloat> ReluSrc;
    TCuda2DArray<float> ReluScale;
    // grad
    TCuda2DArray<half> DQK16;
    TCuda2DArray<half> DQV16;
    TCuda2DArray<half> DK16;
    TCuda2DArray<half> DV16;
    TCuda2DArray<half> DKV16;
    TCuda2DArray<half> DValLookup16;
    TCuda2DArray<e4m3> DValLookupE4;
    TCuda2DArray<e4m3> DValLookupE4T;
    TCuda2DArray<float> DValLookupE4Scale;
    TCuda2DArray<half> DRelu;
    // model delta
    TPackedDeltaMatrix DeltaQK;
    TPackedDeltaMatrix DeltaQV;
    TPackedDeltaMatrix DeltaK;
    TPackedDeltaMatrix DeltaV;
    TPackedDeltaMatrix DeltaCombiner;
    TPackedDeltaMatrix DeltaExpand;
    TPackedDeltaMatrix DeltaContract;
    // sum rank one
    TInt8MatMulBwdBuffers Int8bwdQK;
    TInt8MatMulBwdBuffers Int8bwdQV;
    TInt8MatMulBwdBuffers Int8bwdK;
    TInt8MatMulBwdBuffers Int8bwdV;
    TInt8MatMulBwdBuffers Int8bwdCombiner;
    TInt8MatMulBwdBuffers Int8bwdExpand;
    TInt8MatMulBwdBuffers Int8bwdContract;
    TFp8MatMulBwdBuffers Fp8bwdQK;
    TFp8MatMulBwdBuffers Fp8bwdQV;
    TFp8MatMulBwdBuffers Fp8bwdK;
    TFp8MatMulBwdBuffers Fp8bwdV;
    TFp8MatMulBwdBuffers Fp8bwdCombiner;
    TFp8MatMulBwdBuffers Fp8bwdExpand;
    TFp8MatMulBwdBuffers Fp8bwdContract;
    //
    TCuda2DArray<float> SumWeightLog;
    TCuda2DArray<float> DScale;

    void AllocateCuda(const TModelDims &dims, yint len, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        if (QK.GetYSize() != len || CurDims != dims) {
            CurDims = dims;
            yint dim = dims.Dim;
            yint qSum = dims.GetQSum();
            yint ttSum = dims.GetTTSum();
            yint reluDim = dims.GetReluDim();
            yint combinerDim = dims.GetCombinerDim();
            yint headCount = dims.HeadCount;
            QKscale.AllocateCuda(len, headCount, pool);
            QK.AllocateCuda(qSum, len, pool);
            QVscale.AllocateCuda(len, headCount, pool);
            QV.AllocateCuda(qSum, len, pool);
            K16.AllocateCuda(ttSum, len, pool);
            Kscale.AllocateCuda(len, headCount, pool);
            Vscale.AllocateCuda(len, headCount, pool);
            V.AllocateCuda(ttSum, len, pool);
            KV.AllocateCuda(combinerDim, len, pool);
            ValLookup.AllocateCuda(ttSum, len, pool);
            Relu.AllocateCuda(reluDim, len, pool);
            ReluSrc.AllocateCuda(reluDim, len, pool);
            ReluScale.AllocateCuda(len, reluDim / STATE_NORM_TILE, pool);
            DQK16.AllocateCuda(qSum, len, pool);
            DQV16.AllocateCuda(qSum, len, pool);
            DK16.AllocateCuda(ttSum, len, pool);
            DV16.AllocateCuda(ttSum, len, pool);
            DKV16.AllocateCuda(combinerDim, len, pool);
            DRelu.AllocateCuda(reluDim, len, pool);
            // fp8 att buffers
            if (ATT_TYPE == ATT_FP8) {
                QKT.AllocateCuda(len, qSum, pool);
                QVT.AllocateCuda(len, qSum, pool);
                VT.AllocateCuda(len, ttSum, pool);
                DValLookupE4.AllocateCuda(ttSum, len, pool);
                DValLookupE4T.AllocateCuda(len, ttSum, pool);
                DValLookupE4Scale.AllocateCuda(len, headCount, pool);
            } else if (ATT_TYPE == ATT_FP16) {
                DValLookup16.AllocateCuda(ttSum, len, pool);
            }
            //
            DeltaQK.AllocateCuda(dim, qSum, pool);
            DeltaQV.AllocateCuda(dim, qSum, pool);
            DeltaK.AllocateCuda(dim, ttSum, pool);
            DeltaV.AllocateCuda(dim, ttSum, pool);
            DeltaCombiner.AllocateCuda(combinerDim, dim, pool);
            DeltaExpand.AllocateCuda(dim, reluDim, pool);
            DeltaContract.AllocateCuda(reluDim, dim, pool);
            //
            if (BWD_MATMUL_TYPE == MATMUL_INT8) {
                Int8bwdQK.AllocateCuda(dim, qSum, len, pool);
                Int8bwdQV.AllocateCuda(dim, qSum, len, pool);
                Int8bwdK.AllocateCuda(dim, ttSum, len, pool);
                Int8bwdV.AllocateCuda(dim, ttSum, len, pool);
                Int8bwdCombiner.AllocateCuda(combinerDim, dim, len, pool);
                Int8bwdExpand.AllocateCuda(dim, reluDim, len, pool);
                Int8bwdContract.AllocateCuda(reluDim, dim, len, pool);
            } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
                Fp8bwdQK.AllocateCuda(dim, qSum, len, pool);
                Fp8bwdQV.AllocateCuda(dim, qSum, len, pool);
                Fp8bwdK.AllocateCuda(dim, ttSum, len, pool);
                Fp8bwdV.AllocateCuda(dim, ttSum, len, pool);
                Fp8bwdCombiner.AllocateCuda(combinerDim, dim, len, pool);
                Fp8bwdExpand.AllocateCuda(dim, reluDim, len, pool);
                Fp8bwdContract.AllocateCuda(reluDim, dim, len, pool);
            }
            //
            SumWeightLog.AllocateCuda(len, dims.HeadCount, pool);
            DScale.AllocateCuda(len, dims.HeadCount, pool);
        }
    }
};


struct TLayerGradComputeCtx
{
    yint StateDim = 0;
    TCuda2DArray<float> DNormState;

    void AllocateCuda(const TModelDims &dims, yint len, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        if (DNormState.GetYSize() < len || StateDim != dims.Dim) {
            StateDim = dims.Dim;
            DNormState.AllocateCuda(dims.Dim, len, pool);
        }
    }
};


template <class T, int COUNT>
struct TComputeCtxSet
{
    T CtxArr[COUNT];
    yint CurCtx = 0;

    void AllocateCuda(const TModelDims &dims, yint len, TIntrusivePtr<TCudaMemoryPool> pool)
    {
        for (yint k = 0; k < COUNT; ++k) {
            CtxArr[k].AllocateCuda(dims, len, pool);
        }
    }
    T &GetCtx()
    {
        CurCtx = (CurCtx + 1) % COUNT;
        return CtxArr[CurCtx];
    }
};


struct TAttentionGroupData : public TThrRefBase
{
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> AttSpans;
    TCudaVector<int> AttSpanPtr;
    TCudaVector<TAttentionSpanGroup<ATT_GROUP>> RevAttSpans;
    TCudaVector<int> RevAttSpanPtr;

    void Allocate(const TModelDescr &modelDescr, int maxLen, int groupPerBlock)
    {
        int spanGroups = DivCeil(maxLen, ATT_GROUP);
        AttSpans.Allocate(spanGroups * groupPerBlock);
        RevAttSpans.Allocate(spanGroups * groupPerBlock);
        AttSpanPtr.Allocate(spanGroups + 1);
        RevAttSpanPtr.Allocate(spanGroups + 1);
    }

    template <int N>
    void AssignAttentionGroups(TStream &stream, TAttentionInfoGrouped<N> *pGroups, yint attGroupCount, TCudaVector<TAttentionSpanGroup<N>> *pSpans, TCudaVector<int> *pSpanPtr)
    {
        while (pGroups->GetGroupCount() < attGroupCount) {
            pGroups->AddEmptySpanGroup();
        }
        Put(stream, pSpans, pGroups->SpanGroups);
        Put(stream, pSpanPtr, pGroups->SpanGroupPtr);
    }

    void Init(TStream &stream, yint lenBufferSize, TAttentionInfoGrouped<ATT_GROUP> *pAttGroups, TAttentionInfoGrouped<ATT_GROUP> *pRevAttGroups)
    {
        int attGroupCount = lenBufferSize / ATT_GROUP;
        AssignAttentionGroups(stream, pAttGroups, attGroupCount, &AttSpans, &AttSpanPtr);
        AssignAttentionGroups(stream, pRevAttGroups, attGroupCount, &RevAttSpans, &RevAttSpanPtr);
    }
};


class TFinalLayerWindows
{
public:
    struct TWin : public TThrRefBase
    {
        int Offset = 0;
        TOpParameter<int> Len;
        TOpParameter<int> LenTiles;
        TOpParameter<int> LenRound;

        TWin() {}
        TWin(int offset) : Offset(offset) {}
        void SetLen(yint totalLen)
        {
            yint len = Min<yint>(totalLen - Offset, PREDICTION_ARR_SZ);
            int lenRound = DivCeil(len, MM_TILE) * MM_TILE;
            Len.Set(len);
            LenTiles.Set(lenRound / MM_TILE);
            LenRound.Set(lenRound);
        }
    };

private:
    TVector<TIntrusivePtr<TWin>> WindowArr;
    yint CurrentWindowCount = 0;

public:
    void Create(yint maxWindowCount)
    {
        WindowArr.resize(maxWindowCount);
        for (yint w = 0; w < maxWindowCount; ++w) {
            WindowArr[w] = new TWin(w * PREDICTION_ARR_SZ);
        }
    }

    TWin &GetWindow(yint w)
    {
        return *WindowArr[w];
    }

    void Init(yint len)
    {
        CurrentWindowCount = DivCeil(len, PREDICTION_ARR_SZ);
        for (yint w = 0; w < CurrentWindowCount; ++w) {
            WindowArr[w]->SetLen(len);
        }
    }

    yint GetCurrentWindowCount() const
    {
        return CurrentWindowCount;
    }
};


struct TFragmentStates : public TThrRefBase
{
    TCuda2DArray<TNormStateFloat> NormState;
    TCuda2DArray<float> StateScale;

    void AllocateCuda(yint stateDim, yint len)
    {
        NormState.AllocateCuda(stateDim, len);
        StateScale.AllocateCuda(len, stateDim / STATE_NORM_TILE);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute product of two attention lookups
static void AddLookupProduct(
    TIntrusivePtr<TGraph> c, const TModelDims &dims, TComputeParams *pParams,
    bool copyModelToDevice,
    TVector<TIntrusivePtr<TAttentionGroupData>> *pAttGDArr,
    TComputeCtxSet<TAttentionComputeCtx, ATT_CTX_COUNT> *pAttCtxSet,
    TCudaAttentionParams &att,
    TCuda2DArray<TStateFloat> *pState, TFragmentStates *pKeepState, TFragmentStates *pNextKeepState, TFragmentStates *pNextKeepStateRelu)
{
    int stateDim = dims.Dim;
    int headCount = dims.HeadCount;
    int reluDim = dims.GetReluDim();
    int combinerDim = dims.GetCombinerDim();
    int qSum = Q_DIM * headCount;
    int ttSum = TT_DIM * headCount;
    int ttTiles = ttSum / MM_TILE;
    int TT_GROUPS = (TT_DIM > TILE_GROUP_SIZE) ? TT_DIM / TILE_GROUP_SIZE : 1;

    TAttentionComputeCtx &attCtx = pAttCtxSet->GetCtx();
    float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

    if (copyModelToDevice) {
        att.CopyToDevice(c);
    }

    {
        TAttentionGroupData &attGD = *(*pAttGDArr)[att.AttentionWidthId];
        auto &attCombiner = att.MatrArr[MP_ATT_COMBINER];
        auto &attQK = att.MatrArr[MP_ATT_QK];
        auto &attQV = att.MatrArr[MP_ATT_QV];
        auto &attK = att.MatrArr[MP_ATT_K];
        auto &attV = att.MatrArr[MP_ATT_V];
        TCudaPOD<float> scaleQV = attQV->GetScale();
        TCudaPOD<float> scaleCombiner = attCombiner->GetScale();

        TCuda2DArray<TNormStateFloat> &normState = pKeepState->NormState;

        // mul forward
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            MulForwardFp16<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardFp16<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);
            MulForwardFp16<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardFp16<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            MulForwardFp8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardFp8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);;
            MulForwardFp8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardFp8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        } else {
            MulForwardInt8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardInt8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);;
            MulForwardInt8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardInt8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        }

        if (ATT_TYPE == ATT_FP8) {
            // fp8 attention
            Transpose(c, attCtx.V, ttTiles, pParams->LenTiles, &attCtx.VT);

            CudaCall(c, Fp8Att).Grid(headCount, pParams->LenAttTiles)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.VT)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.ValLookup, &attCtx.SumWeightLog);

        } else if (ATT_TYPE == ATT_FP16) {
            // fp16 attention
            CudaCall(c, Fp16Att<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_LOOKUP_BATCH + TT_GROUPS).Grid(pParams->LenAttTiles, headCount)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.SumWeightLog, &attCtx.ValLookup);
        }

        // KV
        float kvMult = 1 / STATE_VEC_SCALE;
        CudaCall(c, KVProduct<TNormStateFloat>).Grid(headCount, pParams->Len)
            (attCtx.K16, attCtx.ValLookup)
            (kvMult)
            .Write(&attCtx.KV);

        float combinerNormScale = CalcDotScale(combinerDim) * MODEL_DISCR_SCALE / kvMult;
        TKernelOp *addLayer = 0;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            addLayer = &MulForwardFp16<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, combinerDim, stateDim, attCtx.KV, attCombiner, pState);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            addLayer = &MulForwardFp8<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, combinerDim, stateDim, attCtx.KV, attCombiner, pState);
        } else {
            addLayer = &MulForwardInt8<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, combinerDim, stateDim, attCtx.KV, attCombiner, pState);
        }
        (*addLayer)(combinerNormScale, scaleCombiner, STATE_VEC_SCALE).Write(&pNextKeepState->NormState, &pNextKeepState->StateScale);
    }

    // relu
    {
        auto &attExpand = att.MatrArr[MP_RELU_EXPAND];
        auto &attContract = att.MatrArr[MP_RELU_CONTRACT];
        TCudaPOD<float> scaleContract = attContract->GetScale();

        TCuda2DArray<TNormStateFloat> &reluNormState = pNextKeepState->NormState;

        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            MulForwardFp16<TStoreRowTileNormalizeRelu>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.Relu)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            MulForwardFp8<TStoreRowTileNormalizeRelu>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.Relu)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        } else {
            MulForwardInt8<TStoreRowTileNormalizeRelu>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.Relu)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        }

        float contractNormScale = CalcDotScale(reluDim) * MODEL_DISCR_SCALE * RELU_VEC_SCALE;
        TKernelOp *addLayer = 0;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            addLayer = &MulForwardFp16<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, reluDim, stateDim, attCtx.Relu, attContract, pState);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            addLayer = &MulForwardFp8<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, reluDim, stateDim, attCtx.Relu, attContract, pState);
        } else {
            addLayer = &MulForwardInt8<TStoreLayerAddDelta<TNormStateFloat>>(c, pParams, reluDim, stateDim, attCtx.Relu, attContract, pState);
        }
        (*addLayer)(contractNormScale, scaleContract, STATE_VEC_SCALE).Write(&pNextKeepStateRelu->NormState, &pNextKeepStateRelu->StateScale);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// backprop kernels

__global__ void ScaleGrad(float *gradMaxNorm, float *prevGradScale, TCuda2DPtr<TStateGradFloat> grad1, float *gradScale)
{
    int tile = blockIdx.x;
    int t = blockIdx.y;
    int offset = tile * MM_TILE;
    constexpr int WSZ = MM_TILE / WARP_SIZE;

    float gradScaleMult = GetGradScale(*gradMaxNorm);
    if (gradScaleMult != 1) {
        float v[WSZ];
        LoadWarpVec<WSZ>(v, grad1[t] + offset);
        ScaleWarpVec<WSZ>(v, gradScaleMult);
        StoreWarpVec<WSZ>(grad1[t] + offset, v);
    }
    if (t == 0 && tile == 0 && threadIdx.x == 0) {
        CUDA_ASSERT(gradScaleMult != 0);
        *gradScale = *prevGradScale / gradScaleMult;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStateGradData
{
    TCuda2DArray<TStateGradFloat> StateGrad;
    TCuda2DArray<half> StateGrad16;
    TCudaVector<float> LayerGradScale;
    TCudaVector<float> LayerGradMaxNorm;

    void InitGradScale(TStream &stream)
    {
        TVector<float> gradScale;
        ClearPodArray(&gradScale, LayerGradScale.GetSize());
        gradScale[0] = 1;
        Put(stream, &LayerGradScale, gradScale);
    }
    void AllocateCuda(TStream &stream, int dim, int maxLen, int maxStepId)
    {
        StateGrad.AllocateCuda(dim, maxLen);
        StateGrad16.AllocateCuda(dim, maxLen);
        LayerGradScale.Allocate(maxStepId);
        LayerGradMaxNorm.AllocateCuda(maxStepId);
        InitGradScale(stream);
    }
};


// add gradient of product of two attention lookups
static void AddLookupProductBackprop(
    TIntrusivePtr<TGraph> c, const TModelDims &dims, TComputeParams *pParams,
    int stepId,
    TVector<TIntrusivePtr<TAttentionGroupData>> *pAttGDArr,
    TLayerGradComputeCtx *pGradCtx, TComputeCtxSet<TAttentionComputeCtx, ATT_CTX_COUNT> *pAttCtxSet,
    TCudaAttentionParams &att,
    TCudaVector<int> &iterCounter,
    TFragmentStates &prevState, TFragmentStates &prevStateRelu,
    TStateGradData *pGrad
)
{
    TLayerGradComputeCtx &ctx = *pGradCtx;

    int stateDim = dims.Dim;
    int headCount = dims.HeadCount;
    int reluDim = dims.GetReluDim();
    int combinerDim = dims.GetCombinerDim();
    int qSum = Q_DIM * headCount;
    int ttSum = TT_DIM * headCount;
    int qTiles = qSum / MM_TILE;
    int ttTiles = ttSum / MM_TILE;
    int stateTiles = stateDim / MM_TILE;

    int TT_GROUPS = (TT_DIM > TILE_GROUP_SIZE) ? TT_DIM / TILE_GROUP_SIZE : 1;
    int Q_GROUPS = (Q_DIM > TILE_GROUP_SIZE) ? Q_DIM / TILE_GROUP_SIZE : 1;

    TAttentionComputeCtx &attCtx = pAttCtxSet->GetCtx();
    float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;

    // backprop relu
    {
        auto &attExpand = att.MatrArr[MP_RELU_EXPAND];
        auto &attContract = att.MatrArr[MP_RELU_CONTRACT];

        TCudaPOD<float> gradMaxNorm = pGrad->LayerGradMaxNorm.GetElement(stepId);
        TCudaPOD<float> prevGradScale = pGrad->LayerGradScale.GetElement(stepId);
        TCudaPOD<float> gradScale = pGrad->LayerGradScale.GetElement(stepId + 1);
        CudaCall(c, ScaleGrad).Grid(stateTiles, pParams->Len)
            (gradMaxNorm, prevGradScale).Write(&pGrad->StateGrad, &gradScale);

        TCuda2DArray<TNormStateFloat> &reluNormState = prevStateRelu.NormState;

        // expand mul forward
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            MulForwardFp16<TStoreRowTileNormalize>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.ReluSrc)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            MulForwardFp8<TStoreRowTileNormalize>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.ReluSrc)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        } else {
            MulForwardInt8<TStoreRowTileNormalize>(c, pParams, stateDim, reluDim, reluNormState, attExpand, &attCtx.ReluSrc)(stateNormScale, RELU_VEC_SCALE).Write(&attCtx.ReluScale);
        }
        CudaCall(c, MatrixRelu<TNormStateFloat>).Grid(reluDim / MM_TILE, pParams->Len)(attCtx.ReluSrc).Write(&attCtx.Relu);

        // contract mulback
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            CudaCall(c, ConvertMatrixScaled<TStateGradFloat, half>).Grid(stateTiles, pParams->LenRound)(pParams->Len, pGrad->StateGrad, 1.0f).Write(&pGrad->StateGrad16);
            BackpropMatMulFp16(c, pParams, reluDim, stateDim, attCtx.Relu, attContract, pGrad->StateGrad16, gradScale, RESULT_GRAD_MOV, &attCtx.DRelu, &attCtx.DeltaContract);
        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            attCtx.Fp8bwdContract.BackpropMatMul(c, pParams, attCtx.Relu, attContract, pGrad->StateGrad, gradScale, RESULT_GRAD_MOV, &attCtx.DRelu, &attCtx.DeltaContract);
        } else {
            attCtx.Int8bwdContract.BackpropMatMul(c, pParams, attCtx.Relu, attContract, pGrad->StateGrad, gradScale, RESULT_GRAD_MOV, &attCtx.DRelu, &attCtx.DeltaContract);
        }

        CudaCall(c, BackpropRowTileNormalizeRelu<TNormStateFloat, half>).Grid(reluDim / MM_TILE, pParams->Len)(attCtx.ReluSrc, attCtx.ReluScale).Write(&attCtx.DRelu);

        // expand mulback
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            BackpropMatMulFp16(c, pParams, stateDim, reluDim, reluNormState, attExpand, attCtx.DRelu, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaExpand);
        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            attCtx.Fp8bwdExpand.BackpropMatMul(c, pParams, reluNormState, attExpand, attCtx.DRelu, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaExpand);
        } else {
            attCtx.Int8bwdExpand.BackpropMatMul(c, pParams, reluNormState, attExpand, attCtx.DRelu, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaExpand);
        }

        attContract->CopyDeltaToHostAndApply(c, attCtx.DeltaContract, iterCounter);
        attExpand->CopyDeltaToHostAndApply(c, attCtx.DeltaExpand, iterCounter);

        TCudaPOD<float> nextGradMaxNorm = pGrad->LayerGradMaxNorm.GetElement(stepId + 1);
        TCudaPOD<float> attContractScale = attContract->GetScale(); // apply combiner scale to the gradient
        CudaCall(c, BackpropLayerNormalize<TNormStateFloat, TStateGradFloat>).Shmem(STATE_NORM_TILE * 2 * sizeof(float)).Grid(stateDim / STATE_NORM_TILE, pParams->Len)
            (STATE_NORM_TILE, pParams->DropTable, reluNormState, prevStateRelu.StateScale, ctx.DNormState)
            (attContractScale)
            .Write(&pGrad->StateGrad).AtomicWrite(&nextGradMaxNorm);
    }

    // backprop attention
    yint stepId2 = stepId + 1;
    {
        TCudaPOD<float> gradMaxNorm = pGrad->LayerGradMaxNorm.GetElement(stepId2);
        TCudaPOD<float> prevGradScale = pGrad->LayerGradScale.GetElement(stepId2);
        TCudaPOD<float> gradScale = pGrad->LayerGradScale.GetElement(stepId2 + 1);
        CudaCall(c, ScaleGrad).Grid(stateTiles, pParams->Len)
            (gradMaxNorm, prevGradScale).Write(&pGrad->StateGrad, &gradScale);

        // attention derivatives
        TCuda2DArray<TNormStateFloat> &normState = prevState.NormState;

        TAttentionGroupData &attGD = *(*pAttGDArr)[att.AttentionWidthId];
        auto &attCombiner = att.MatrArr[MP_ATT_COMBINER];
        auto &attQK = att.MatrArr[MP_ATT_QK];
        auto &attQV = att.MatrArr[MP_ATT_QV];
        auto &attK = att.MatrArr[MP_ATT_K];
        auto &attV = att.MatrArr[MP_ATT_V];
        TCudaPOD<float> scaleQV = attQV->GetScale();

        // mul forward
        //float stateNormScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            MulForwardFp16<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardFp16<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);
            MulForwardFp16<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardFp16<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            MulForwardFp8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardFp8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);;
            MulForwardFp8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardFp8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        } else {
            MulForwardInt8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQK, &attCtx.QK)(pParams->RopeBuf, stateNormScale, QK_VEC_SCALE).Write(&attCtx.QKscale);
            MulForwardInt8<TStoreRowTileNormalizeRope>(c, pParams, stateDim, qSum, normState, attQV, &attCtx.QV)(pParams->RopeBuf, stateNormScale, QV_VEC_SCALE).Write(&attCtx.QVscale);;
            MulForwardInt8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attK, &attCtx.K16)(stateNormScale, K_VEC_SCALE).Write(&attCtx.Kscale);
            MulForwardInt8<TStoreRowTileNormalize>(c, pParams, stateDim, ttSum, normState, attV, &attCtx.V)(stateNormScale, V_VEC_SCALE).Write(&attCtx.Vscale);
        }

        if (ATT_TYPE == ATT_FP8) {
            // fp8 attention
            Transpose(c, attCtx.V, ttTiles, pParams->LenTiles, &attCtx.VT);
            Transpose(c, attCtx.QK, qTiles, pParams->LenTiles, &attCtx.QKT);
            Transpose(c, attCtx.QV, qTiles, pParams->LenTiles, &attCtx.QVT);

            CudaCall(c, Fp8Att).Grid(headCount, pParams->LenAttTiles)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.VT)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.ValLookup, &attCtx.SumWeightLog);

        } else if (ATT_TYPE == ATT_FP16) {
            // fp16 attention
            CudaCall(c, Fp16Att<Q_DIM, TT_DIM>).Block(WARP_SIZE, ATT_LOOKUP_BATCH + TT_GROUPS).Grid(pParams->LenAttTiles, headCount)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.SumWeightLog, &attCtx.ValLookup);
        }

        // KV
        CudaCall(c, KVProduct<TNormStateFloat>).Grid(headCount, pParams->Len)
            (attCtx.K16, attCtx.ValLookup)
            (1 / STATE_VEC_SCALE)
            .Write(&attCtx.KV);

        // combiner derivatives
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            CudaCall(c, ConvertMatrixScaled<TStateGradFloat, half>).Grid(stateTiles, pParams->LenRound)(pParams->Len, pGrad->StateGrad, 1.0f).Write(&pGrad->StateGrad16);
            BackpropMatMulFp16(c, pParams, combinerDim, stateDim, attCtx.KV, attCombiner, pGrad->StateGrad16, gradScale, RESULT_GRAD_MOV, &attCtx.DKV16, &attCtx.DeltaCombiner);

        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            attCtx.Fp8bwdCombiner.BackpropMatMul(c, pParams, attCtx.KV, attCombiner, pGrad->StateGrad, gradScale, RESULT_GRAD_MOV, &attCtx.DKV16, &attCtx.DeltaCombiner);

        } else {
            attCtx.Int8bwdCombiner.BackpropMatMul(c, pParams, attCtx.KV, attCombiner, pGrad->StateGrad, gradScale, RESULT_GRAD_MOV, &attCtx.DKV16, &attCtx.DeltaCombiner);
        }

        float attGradMult = 1; // can be used to solve overflow issues

        // attention derivative
        if (ATT_TYPE == ATT_FP8) {
            // fp8 attiention derivatives

            // KV backprop + backprop normalize(K)
            CudaCall(c, KVProductBackpropE4fused<e4m3>).Grid(headCount, pParams->LenRound)
                (pParams->Len)
                (attCtx.K16, attCtx.Kscale, attCtx.ValLookup, attCtx.DKV16)
                (attGradMult)
                .Write(&attCtx.DK16, &attCtx.DValLookupE4, &attCtx.DValLookupE4Scale, &attCtx.DScale);

            Transpose(c, attCtx.DValLookupE4, ttTiles, pParams->LenTiles, &attCtx.DValLookupE4T);

            // grad QK + backprop normalize(QK)
            CudaCall(c, Fp8AttGradQK).Grid(headCount, pParams->LenAttTiles)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V, attCtx.QVT)
                (attCtx.DValLookupE4, attCtx.DValLookupE4Scale)
                (attCtx.DScale, attCtx.SumWeightLog)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                (attGradMult)
                (pParams->RopeBuf)
                (attCtx.QKscale).Write(&attCtx.DQK16);

            // grad QV + backprop normalize(V)
            CudaCall(c, Fp8AttGradQV).Grid(headCount, pParams->LenAttTiles)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V, attCtx.QKT)
                (attCtx.DValLookupE4, attCtx.DValLookupE4T, attCtx.DValLookupE4Scale)
                (attCtx.DScale, attCtx.SumWeightLog)
                (attGD.RevAttSpans, attGD.RevAttSpanPtr, att.AlibiSlope)
                (attGradMult)
                (pParams->RopeBuf)
                (attCtx.Vscale).Write(&attCtx.DV16, &attCtx.DQV16);

        } else if (ATT_TYPE == ATT_FP16) {
            // fp16 attention derivatives
            constexpr int ATT_GRAD_BATCH = 4;

            // KV backprop
            CudaCall(c, KVProductBackprop).Grid(headCount, pParams->LenRound)
                (pParams->Len)
                (attCtx.K16, attCtx.ValLookup, attCtx.DKV16)
                (attGradMult)
                .Write(&attCtx.DK16, &attCtx.DValLookup16);

            CudaCall(c, CalcDScale<TT_DIM>).Grid(headCount, pParams->LenRound)(attCtx.ValLookup, attCtx.DValLookup16).Write(&attCtx.DScale);

            CudaCall(c, Fp16AttGradQK<Q_DIM, TT_DIM, ATT_GRAD_BATCH>).Block(WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS).Grid(pParams->LenAttTiles, headCount)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V)
                (attCtx.DValLookup16)
                (attCtx.DScale, attCtx.SumWeightLog)
                (attGD.AttSpans, attGD.AttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.DQK16);
            //CudaCall(c, PrintVec<128, float>)(40, attCtx.DQKFloat);

            CudaCall(c, Fp16AttGradQV<Q_DIM, TT_DIM, ATT_GRAD_BATCH>).Block(WARP_SIZE, ATT_GRAD_BATCH + Q_GROUPS + TT_GROUPS).Grid(pParams->LenAttTiles, headCount)
                (scaleQV, attCtx.QK, attCtx.QV, attCtx.QVscale)
                (attCtx.V)
                (attCtx.DValLookup16)
                (attCtx.DScale, attCtx.SumWeightLog)
                (attGD.RevAttSpans, attGD.RevAttSpanPtr, att.AlibiSlope)
                .Write(&attCtx.DQV16, &attCtx.DV16);

            // normally we should backprop rope then backprop tile normalization but we kept rotated QK/QV only, so we perform backprop in reverse order to get correct results
            CudaCall(c, BackpropRowTileNormalize<TAttVecFloat, half>).Grid(headCount, pParams->Len)(attCtx.QK, attCtx.QKscale).Write(&attCtx.DQK16);
            //CudaCall(c, BackpropRowTileNormalize<TAttVecFloat,half>).Grid(headCount, pParams->Len)(attCtx.QV, attCtx.QVscale).Write(&attCtx.DQV16);

            CudaCall(c, ApplyRope<half>).Grid(headCount, pParams->Len)(pParams->RopeBuf, -1.0f).Write(&attCtx.DQK16);
            CudaCall(c, ApplyRope<half>).Grid(headCount, pParams->Len)(pParams->RopeBuf, -1.0f).Write(&attCtx.DQV16);

            //CudaCall(c, BackpropRowTileNormalize<TAttVecFloat, half>).Grid(headCount, pParams->Len)(attCtx.QK, attCtx.QKscale).Write(&attCtx.DQK16);
            ////CudaCall(c, BackpropRowTileNormalize<TAttVecFloat,half>).Grid(headCount, pParams->Len)(attCtx.QV, attCtx.QVscale).Write(&attCtx.DQV16);
            CudaCall(c, BackpropRowTileNormalize<half, half>).Grid(headCount, pParams->Len)(attCtx.K16, attCtx.Kscale).Write(&attCtx.DK16);
            CudaCall(c, BackpropRowTileNormalize<TAttVecFloat, half>).Grid(headCount, pParams->Len)(attCtx.V, attCtx.Vscale).Write(&attCtx.DV16);
        }

        // backprop mul forward
        if (BWD_MATMUL_TYPE == MATMUL_FP16) {
            BackpropMatMulFp16(c, pParams, stateDim, qSum, normState, attQK, attCtx.DQK16, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaQK);
            BackpropMatMulFp16(c, pParams, stateDim, qSum, normState, attQV, attCtx.DQV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaQV);
            BackpropMatMulFp16(c, pParams, stateDim, ttSum, normState, attK, attCtx.DK16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaK);
            BackpropMatMulFp16(c, pParams, stateDim, ttSum, normState, attV, attCtx.DV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaV);

        } else if (BWD_MATMUL_TYPE == MATMUL_FP8) {
            attCtx.Fp8bwdQK.BackpropMatMul(c, pParams, normState, attQK, attCtx.DQK16, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaQK);
            attCtx.Fp8bwdQV.BackpropMatMul(c, pParams, normState, attQV, attCtx.DQV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaQV);
            attCtx.Fp8bwdK.BackpropMatMul(c, pParams, normState, attK, attCtx.DK16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaK);
            attCtx.Fp8bwdV.BackpropMatMul(c, pParams, normState, attV, attCtx.DV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaV);

        } else {
            attCtx.Int8bwdQK.BackpropMatMul(c, pParams, normState, attQK, attCtx.DQK16, gradScale, RESULT_GRAD_MOV, &ctx.DNormState, &attCtx.DeltaQK);
            attCtx.Int8bwdQV.BackpropMatMul(c, pParams, normState, attQV, attCtx.DQV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaQV);
            attCtx.Int8bwdK.BackpropMatMul(c, pParams, normState, attK, attCtx.DK16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaK);
            attCtx.Int8bwdV.BackpropMatMul(c, pParams, normState, attV, attCtx.DV16, gradScale, RESULT_GRAD_ADD, &ctx.DNormState, &attCtx.DeltaV);
        }

        // accumulate delta on host
        attCombiner->CopyDeltaToHostAndApply(c, attCtx.DeltaCombiner, iterCounter);
        attQK->CopyDeltaToHostAndApply(c, attCtx.DeltaQK, iterCounter);
        attQV->CopyDeltaToHostAndApply(c, attCtx.DeltaQV, iterCounter);
        attK->CopyDeltaToHostAndApply(c, attCtx.DeltaK, iterCounter);
        attV->CopyDeltaToHostAndApply(c, attCtx.DeltaV, iterCounter);

        TCudaPOD<float> nextGradMaxNorm = pGrad->LayerGradMaxNorm.GetElement(stepId2 + 1);
        TCudaPOD<float> attCombinerScale = attCombiner->GetScale(); // apply combiner scale to the gradient
        CudaCall(c, BackpropLayerNormalize<TNormStateFloat, TStateGradFloat>).Shmem(STATE_NORM_TILE * 2 * sizeof(float)).Grid(stateDim / STATE_NORM_TILE, pParams->Len)
            (STATE_NORM_TILE, pParams->DropTable, normState, prevState.StateScale, ctx.DNormState)
            (attCombinerScale)
            .Write(&pGrad->StateGrad).AtomicWrite(&nextGradMaxNorm);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// embedding & final layer

///////////////////////////////////////////////////////////////////////////////////////////////////
class TSinlgeComputeContext : public TThrRefBase
{
    TStream Stream;
    TIntrusivePtr<TCudaMemoryAllocator> CudaMem;
    TIntrusivePtr<TCudaMemoryPool> PoolEmbed;
    TIntrusivePtr<TCudaMemoryPool> PoolLayers;
    TIntrusivePtr<TCudaMemoryPool> PoolFinal;
    TModelDescr ModelDescr;
    TIntrusivePtr<IModel> Model;
    TIntrusivePtr<TCudaModelMatrixScale> CudaMatrixScale;
    TIntrusivePtr<TCudaModelMatrix<TFastModelFloat>> FinalLayer;
    TIntrusivePtr<TCudaModelMatrix<TEmbedFloat>> Embedding;
    TVector<TIntrusivePtr<TCudaAttentionParams>> LayerArr;
    // model params
    TCudaVector<float> Bias;
    // gradients
    TCuda2DArray<float> DeltaFinalLayer;
    // compute params
    TCudaVector<TLabelIndex> LabelArr;
    TCudaVector<TLabelIndex> UsedLabelArr;
    TCudaVector<ui32> LabelPtr;
    yint TargetNodeCount = 0;
    TCudaVector<TLabelIndex> InvLabelArr;
    TCudaVector<ui32> InvLabelPos;
    TCudaVector<ui32> InvLabelPosPtr;
    TVector<TIntrusivePtr<TFragmentStates>> AllStates;
    TCuda2DArray<TStateFloat> State; // after forward pass contains state after all layers applied
    TCuda2DArray<TEmbedFloat> UsedEmbedBuffer; // temporary use, can reuse memory from other buffers (like gradient) for this
    TStateGradData GradData;
    TCuda2DArray<half> LogitBuf;
    TCuda2DArray<float> LogitBufRowTileLogSum;
    TCuda2DArray<half> LogitBufHost;
    TCudaVector<int> TargetArr;
    TCudaVector<float> TargetResProbArr;
    TCudaVector<int> IterCounter;
    TCudaVector<float> SumScore;
    TCudaVector<float> SumTrainErr;

    TComputeParams ComputeParams;
    TFinalLayerWindows FinalWindows;
    TLabelInverseIndex LabelInverseIndex;
    TLabelForwardIndex LabelForwardIndex;
    TVector<TIntrusivePtr<TAttentionGroupData>> AttGDArr;
    TComputeCtxSet<TLayerGradComputeCtx, LAYER_CTX_COUNT> LayerCtxSet;
    TComputeCtxSet<TAttentionComputeCtx, ATT_CTX_COUNT> AttCtxSet;

    bool ComputeInFly = false;
    bool NeedCopyToDevice = true;

public:
    // public to create with external template by size class
    TIntrusivePtr<TGraph> ForwardComputer;
    TIntrusivePtr<TGraph> CopyModelForwardComputer;
    TVector<TIntrusivePtr<TGraph>> BackpropComputerArr;
    TVector<TIntrusivePtr<TGraph>> CopyModelBackpropComputerArr;
    TVector<TIntrusivePtr<TGraph>> SumScoreComputerArr;
    TVector<TIntrusivePtr<TGraph>> FinalLayerWindowComputerArr;
    TVector<TIntrusivePtr<TGraph>> FinalLayerWindowComputerFullArr;

private:
    yint GetVocabSizeRounded() const
    {
        return DivCeil(ModelDescr.VocabSize, MM_TILE) * MM_TILE;
    }

    yint GetFinalLayerSizeRounded() const
    {
        return DivCeil(ModelDescr.VocabSize, MM_TILE) * MM_TILE;
    }

public:
    void WaitCudaCompute()
    {
        if (ComputeInFly) {
            Stream.Sync();
            ComputeInFly = false;
        }
    }

public:
    TSinlgeComputeContext(yint deviceId, TIntrusivePtr<IModel> pModel, yint nodeCount) : Model(pModel)
    {
        ModelDescr = Model->GetModelDescr();
        // dimension restrictions
        yint dim = ModelDescr.Dims.Dim;
        Y_VERIFY((dim % WARP_SIZE) == 0); // some kernels process states with warps
        Y_VERIFY((dim % I8_TILE_GROUP_SIZE) == 0);
        Y_VERIFY((ModelDescr.Dims.GetTTSum() % MM_TILE) == 0);
        // tt restrictions
        Y_VERIFY((ModelDescr.Dims.GetTTSum() % TILE_GROUP_SIZE) == 0);
        // params
        yint maxLen = DivCeil(nodeCount, MM_TILE) * MM_TILE;
        yint vocabRoundSize = GetVocabSizeRounded();
        yint finalLayerRoundSize = GetFinalLayerSizeRounded();
        yint labelCount = ModelDescr.LabelCount;
        yint maxLabels = maxLen * 8; // upper cap
        yint maxUsedLabels = maxLen * 2; // upper cap
        yint finalMaxLen = Min<yint>(maxLen, PREDICTION_ARR_SZ);
        yint maxWindowCount = DivCeil(maxLen, PREDICTION_ARR_SZ);
        yint maxStepId = YSize(ModelDescr.Layers) * 2 + 100; // upper cap
        yint depth = YSize(ModelDescr.Layers);
        //
        CudaMem = new TCudaMemoryAllocator();
        PoolEmbed = CudaMem->CreatePool();
        PoolLayers = CudaMem->CreatePool();
        PoolFinal = CudaMem->CreatePool();
        //
        CudaMatrixScale = new TCudaModelMatrixScale(Model->GetMatrixScale(), Stream);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            LayerArr[d] = new TCudaAttentionParams(deviceId, CudaMatrixScale, Model->GetAttention(d));
        }
        Embedding = new TCudaModelMatrix<TEmbedFloat>(deviceId, CudaMatrixScale, Model->GetEmbedding(), MM_MEM_HOST);
        FinalLayer = new TCudaModelMatrix<TFastModelFloat>(deviceId, CudaMatrixScale, Model->GetFinalLayer(), MM_MEM_DEVICE);
        //
        Bias.Allocate(ModelDescr.VocabSize);
        // gradients
        if (LOW_VRAM_CONSUMPTION) {
            DeltaFinalLayer.AllocateCuda(dim, finalLayerRoundSize, PoolFinal);
        } else {
            DeltaFinalLayer.AllocateCuda(dim, finalLayerRoundSize);
        }
        //
        LabelArr.AllocateWC(maxLabels); // upper cap
        UsedLabelArr.AllocateWC(maxUsedLabels); // upper cap
        LabelPtr.AllocateWC(maxLen + 1);
        InvLabelArr.AllocateWC(labelCount); // upper cap
        InvLabelPos.AllocateWC(maxLabels); // upper cap
        InvLabelPosPtr.AllocateWC(labelCount + 1);
        AllStates.resize(depth * 2 + 1);
        for (yint k = 0; k < YSize(AllStates); ++k) {
            AllStates[k] = new TFragmentStates;
            AllStates[k]->AllocateCuda(dim, maxLen);
        }
        State.Allocate(dim, maxLen);
        UsedEmbedBuffer.AllocateCuda(dim, maxUsedLabels, PoolEmbed);
        GradData.AllocateCuda(Stream, dim, maxLen, maxStepId);
        LogitBuf.AllocateCuda(finalLayerRoundSize, finalMaxLen, PoolFinal);
        LogitBufRowTileLogSum.AllocateCuda(finalMaxLen, finalLayerRoundSize / MM_TILE, PoolFinal);
        LogitBufHost.AllocateHost(finalLayerRoundSize, finalMaxLen);
        TargetArr.Allocate(maxLen);
        TargetResProbArr.Allocate(maxLen);
        IterCounter.AllocateCuda(1);
        IterCounter.ClearDeviceMem(Stream);
        SumScore.Allocate(1);
        SumTrainErr.Allocate(4);
        SumTrainErr.ClearDeviceMem(Stream);
        //
        LabelForwardIndex.Init(labelCount);
        //
        // compute params & contexts
        ComputeParams.Allocate(Stream, ModelDescr, maxLen);
        FinalWindows.Create(maxWindowCount);
        yint attentionWidthCount = ModelDescr.GetAttentionWidthCount();
        AttGDArr.resize(attentionWidthCount);
        for (yint wa = 0; wa < attentionWidthCount; ++wa) {
            AttGDArr[wa] = new TAttentionGroupData();
            AttGDArr[wa]->Allocate(ModelDescr, maxLen, 4); // upper cap
        }
        LayerCtxSet.AllocateCuda(ModelDescr.Dims, maxLen, PoolLayers);
        AttCtxSet.AllocateCuda(ModelDescr.Dims, maxLen, PoolLayers);
        // create compute graphs
        Y_VERIFY(ModelDescr.Dims.Dim % MM_TILE == 0);
        Y_VERIFY(ModelDescr.Dims.QDim == Q_DIM);
        Y_VERIFY(ModelDescr.Dims.TTDim == TT_DIM);
        Y_VERIFY(Q_DIM == MM_TILE);
        Y_VERIFY(TT_DIM == MM_TILE);
        CudaMem->AllocateMemory();
        CreateGraphs(maxWindowCount);
        // assign model params
        CopyModelParams();
    }

private:
    // create compute graphs
    void AddForwardGraph(TIntrusivePtr<TGraph> c, bool isBackprop, bool copyModelToDevice)
    {
        TModelDims &dims = ModelDescr.Dims;
        TComputeParams *pParams = &ComputeParams;
        int stateDim = dims.Dim;
        int stateTiles = stateDim / MM_TILE;

        if (copyModelToDevice) {
            CudaMatrixScale->CopyToDevice(c);
        }

        // copy used embeddings
        c->SetMemPool(PoolEmbed);
        CudaCall(c, CopyUsedEmbeddings).Grid(stateTiles, pParams->UsedLabelCount)
            (UsedLabelArr, Embedding->GetFast()).Write(&UsedEmbedBuffer);

        // compute token vectors
        TCudaPOD<float> scaleEmbed = Embedding->GetScale();
        CudaCall(c, AddEmbeddings).Grid(stateTiles, pParams->LenRound)
            (pParams->Len)
            (LabelArr, LabelPtr, UsedEmbedBuffer, scaleEmbed, MODEL_DISCR_SCALE)
            .Write(&State);

        // all data from embedding matrix is read, can modify it
        if (isBackprop) {
            Embedding->AllowDelayedUpdates(c, IterCounter);
        }

        CudaCall(c, LayerNormalizeStateVecs<TStateFloat, TNormStateFloat>).Shmem(STATE_NORM_TILE * sizeof(float)).Grid(stateDim / STATE_NORM_TILE, pParams->LenRound)
            (pParams->Len, STATE_NORM_TILE, STATE_VEC_SCALE, State, pParams->DropTable).Write(&AllStates[0]->NormState, &AllStates[0]->StateScale);

        // apply layers
        c->SetMemPool(PoolLayers);
        Y_ASSERT(YSize(LayerArr) == YSize(ModelDescr.Layers));
        for (yint d = 0; d < YSize(LayerArr); ++d) {
            AddLookupProduct(c, ModelDescr.Dims, pParams,
                copyModelToDevice,
                &AttGDArr, &AttCtxSet,
                *LayerArr[d],
                &State, AllStates[d * 2].Get(), AllStates[d * 2 + 1].Get(), AllStates[d * 2 + 2].Get());
        }

        if (copyModelToDevice) {
            FinalLayer->CopyToDevice(c); // have to copy once if we don't update final layer
        }
    }

    TIntrusivePtr<TGraph> CreateForwardGraph(bool copyModelToDevice)
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        AddForwardGraph(c, false, copyModelToDevice);
        return c;
    }


    void AddFinalLayerWindow(TIntrusivePtr<TGraph> c, TFinalLayerWindows::TWin &window)
    {
        TModelDims &dims = ModelDescr.Dims;
        int stateDim = dims.Dim;
        int flSize = GetFinalLayerSizeRounded();
        int vocabSize = ModelDescr.VocabSize;
        TCudaPOD<float> scaleFinalLayer = FinalLayer->GetScale();
        auto finalStateNormalized = AllStates.back()->NormState.MakeFragment(0, window.Offset);

        float normScale = CalcDotScale(stateDim) * STATE_VEC_SCALE * MODEL_DISCR_SCALE;
        float flScale = CalcFinalLayerMult() * normScale;
        if (FWD_MATMUL_TYPE == MATMUL_FP16) {
            // using half precision sum here impairs quality beyond repair
            MulForwardFp16<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, FinalLayer, &LogitBuf)(scaleFinalLayer, flScale, vocabSize, Bias).Write(&LogitBufRowTileLogSum);
        } else if (FWD_MATMUL_TYPE == MATMUL_FP8) {
            MulForwardFp8<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, FinalLayer, &LogitBuf)(scaleFinalLayer, flScale, vocabSize, Bias).Write(&LogitBufRowTileLogSum);
        } else {
            MulForwardInt8<TStoreFinalLayerLogits>(c, &window, stateDim, flSize, finalStateNormalized, FinalLayer, &LogitBuf)(scaleFinalLayer, flScale, vocabSize, Bias).Write(&LogitBufRowTileLogSum);
        }
        CudaCall(c, SumLogWeight).Grid(window.LenTiles)(flSize / MM_TILE).Write(&LogitBufRowTileLogSum);
    }


    void ComputeFinalProb(TIntrusivePtr<TGraph> c, TFinalLayerWindows::TWin &window)
    {
        int flSize = GetFinalLayerSizeRounded();
        int vocabSize = ModelDescr.VocabSize;
        CudaCall(c, ComputeFinalProbKernel).Grid(flSize / MM_TILE, window.Len)(window.Offset, vocabSize, LogitBufRowTileLogSum, TargetArr).Write(&LogitBuf, &TargetResProbArr);
    }


    template <class TStoreDelta>
    void AddBackpropFinalLayerGraph(TIntrusivePtr<TGraph> c, yint windowOffset, TFinalLayerWindows::TWin &window)
    {
        TModelDims &dims = ModelDescr.Dims;
        int stateDim = dims.Dim;
        int finalTiles = GetFinalLayerSizeRounded() / MM_TILE;
        int stateTiles = stateDim / MM_TILE;
        int vocabRoundSize = GetVocabSizeRounded();

        // compute gradient (scale gradient by VEC_SCALE to avoid fp16 sum overflow in DeltaFinalLayer computation)
        CudaCall(c, ComputeGradient).Grid(finalTiles, window.LenRound)
            (window.Len, window.Offset, ModelDescr.VocabSize, vocabRoundSize, LogitBuf, LogitBufRowTileLogSum, TargetArr)
            .Write(&LogitBuf, &SumTrainErr);
        CudaCall(c, CollectSumTrainErr).Write(&SumTrainErr);

        // mul backward, scaling gradient by constant does not change result so we avoid scaling by final layer scale and MODEL_DISCR_SCALE
        yint xSize = GradData.StateGrad.GetXSize();
        yint ySize = Min<yint>(PREDICTION_ARR_SZ, GradData.StateGrad.GetYSize() - windowOffset);
        auto stateGradFrag = GradData.StateGrad.MakeFragment(0, xSize, windowOffset, ySize);
        // using half precision accumulators on this mul backward destroys quality for some reason, so use full precision always
        MatMulXYoYZeXZ<TStore>(c, LogitBuf, FinalLayer->GetFast(), &stateGradFrag, window.LenTiles, finalTiles, stateTiles).Struct();

        auto finalStateNormalized = AllStates.back()->NormState.MakeFragment(0, window.Offset);
        // using half precisio sums is problematic due to sum overflow with typical state vec scale 1/32
        MatMulXYoXZeYZ<TStoreDelta>(c, LogitBuf, finalStateNormalized, &DeltaFinalLayer, window.LenTiles, finalTiles, stateTiles).Struct();
    }


    TIntrusivePtr<TGraph> CreateBackpropGraph(bool copyModelToDevice, yint windowCount)
    {
        TModelDims &dims = ModelDescr.Dims;
        int stateDim = dims.Dim;
        int stateTiles = stateDim / MM_TILE;
        TComputeParams *pParams = &ComputeParams;

        TIntrusivePtr<TGraph> c = new TGraph;

        // new flag value for deltas
        AssignIterCounter(c, Model->GetCurrentIteration(), &IterCounter);

        AddForwardGraph(c, true, copyModelToDevice);

        // backprop final layer
        c->SetMemPool(PoolFinal);
        for (yint w = 0; w < windowCount; ++w) {
            TFinalLayerWindows::TWin &window = FinalWindows.GetWindow(w);
            // apply final layer on fragment
            AddFinalLayerWindow(c, window);
            // compute final layer gradient on fragment
            if (w == 0) {
                AddBackpropFinalLayerGraph<TStore>(c, window.Offset, window);
            } else {
                AddBackpropFinalLayerGraph<TStoreAdd>(c, window.Offset, window);
            }
        }
        FinalLayer->CopyDeltaToHostAndApply(c, DeltaFinalLayer, IterCounter);

        {
            // use State contents from forward pass
            c->ClearMem(GradData.LayerGradMaxNorm);
            TCudaPOD<float> gradMaxNorm = GradData.LayerGradMaxNorm.GetElement(0);
            CudaCall(c, BackpropFinalNormalize<TStateFloat, TStateGradFloat>).Shmem(STATE_NORM_TILE * 2 * sizeof(float)).Grid(stateDim / STATE_NORM_TILE, pParams->LenRound)
                (pParams->Len, STATE_NORM_TILE)
                (pParams->DropTable, State, GradData.StateGrad)
                .Write(&GradData.StateGrad).AtomicWrite(&gradMaxNorm);
        }

        // modify layers
        c->SetMemPool(PoolLayers);
        int stepId = 0;
        for (yint d = YSize(LayerArr) - 1; d >= 0; --d) {
            TLayerGradComputeCtx &layerGradCtx = LayerCtxSet.GetCtx();
            AddLookupProductBackprop(c, ModelDescr.Dims,pParams,
                stepId,
                &AttGDArr, &layerGradCtx, &AttCtxSet,
                *LayerArr[d],
                IterCounter,
                *AllStates[d * 2], *AllStates[d * 2 + 1],
                &GradData);
            stepId += 2;
        }

        {
            // tune embed
            TCudaVector<i8> &deltaLabel = Embedding->GetDelta();
            TCudaVector<float> &deltaLabelTileScale = Embedding->GetDeltaTileScale();
            TCudaPOD<float> gradScale = GradData.LayerGradScale.GetElement(stepId);
            Embedding->ClearTileScale(c);
            CudaCall(c, BackpropEmbeddings<TStateGradFloat>).Grid(stateTiles, pParams->InvLabelCount)
                (stateDim, InvLabelArr, InvLabelPos, InvLabelPosPtr)
                (GradData.StateGrad, gradScale)
                .Write(&deltaLabel, &deltaLabelTileScale);
            Embedding->ApplyHostDelta(c, IterCounter);
        }
        return c;
    }


    TIntrusivePtr<TGraph> CreateSumScoreGraph(yint windowCount)
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        c->SetMemPool(PoolFinal);
        c->ClearMem(SumScore);
        for (yint w = 0; w < windowCount; ++w) {
            TFinalLayerWindows::TWin &window = FinalWindows.GetWindow(w);
            // apply final layer on fragment
            AddFinalLayerWindow(c, window);
            // compute score
            CudaCall(c, ComputeLossKernel)(window.Offset, window.Len, LogitBuf, LogitBufRowTileLogSum, TargetArr)
                .Write(&SumScore);
        }
        return c;
    }

    TIntrusivePtr<TGraph> CreateFinalLayerWindowGraph(TFinalLayerWindows::TWin &window)
    {
        TIntrusivePtr<TGraph> c = new TGraph;
        c->SetMemPool(PoolFinal);
        AddFinalLayerWindow(c, window);
        ComputeFinalProb(c, window);
        return c;
    }

    void CreateGraphs(yint maxWindowCount)
    {
        ForwardComputer = CreateForwardGraph(false);
        CopyModelForwardComputer = CreateForwardGraph(true);
        BackpropComputerArr.resize(maxWindowCount + 1);
        CopyModelBackpropComputerArr.resize(maxWindowCount + 1);
        SumScoreComputerArr.resize(maxWindowCount + 1);
        for (yint wc = 1; wc <= maxWindowCount; ++wc) {
            BackpropComputerArr[wc] = CreateBackpropGraph(false, wc);
            CopyModelBackpropComputerArr[wc] = CreateBackpropGraph(true, wc);
            SumScoreComputerArr[wc] = CreateSumScoreGraph(wc);
        }
        FinalLayerWindowComputerArr.resize(maxWindowCount);
        FinalLayerWindowComputerFullArr.resize(maxWindowCount);
        for (yint w = 0; w < maxWindowCount; ++w) {
            TFinalLayerWindows::TWin &window = FinalWindows.GetWindow(w);
            FinalLayerWindowComputerArr[w] = CreateFinalLayerWindowGraph(window);
            TIntrusivePtr<TGraph> c = CreateFinalLayerWindowGraph(window);
            c->KernelCopy(&LogitBufHost, LogitBuf);
            FinalLayerWindowComputerFullArr[w] = c;
        }
    }

public:
    TModelDescr GetLocalModelDescr()
    {
        return ModelDescr;
    }

    void CopyModelParams()
    {
        Put(Stream, &Bias, Model->GetBias());
        Stream.Sync();
        NeedCopyToDevice = true;
    }

    void SetTarget(yint len, const TVector<TNodeTarget> &target)
    {
        TVector<int> targetArr;
        targetArr.resize(len, -1);
        yint count = 0;
        for (const TNodeTarget &nt : target) {
            Y_VERIFY(nt.TargetId >= 0 && nt.TargetId < ModelDescr.VocabSize);
            Y_VERIFY(targetArr[nt.Node] == -1); // current ComputeGradient() supports single target per node
            targetArr[nt.Node] = nt.TargetId;
            ++count;
        }
        TargetNodeCount = count;
        Put(Stream, &TargetArr, targetArr);
    }

    void Init(const TNodesBatch &nodes, const TVector<ui32> &dropTable, IComputeContext::EInitType initType)
    {
        Y_ASSERT(ModelDescr == Model->GetModelDescr());
        yint len = nodes.GetNodeCount();
        Y_VERIFY(YSize(nodes.LabelPtr) <= LabelPtr.GetSize());
        Y_VERIFY(nodes.LabelPtr.back() <= LabelArr.GetSize());
        for (yint pos : nodes.LabelArr) {
            Y_ASSERT((pos & LABEL_MASK) < ModelDescr.LabelCount);
        }
        yint attentionWidthCount = ModelDescr.GetAttentionWidthCount();
        TVector<TAttentionInfoGrouped<ATT_GROUP>> attGroupsArr;
        TVector < TAttentionInfoGrouped<ATT_GROUP>> revAttGroupsArr;
        attGroupsArr.resize(attentionWidthCount);
        revAttGroupsArr.resize(attentionWidthCount);
        for (yint wa = 0; wa < attentionWidthCount; ++wa) {
            GroupAttention<ATT_GROUP, ATT_ALIGN>(nodes.AttArr[wa], &attGroupsArr[wa]);
            GroupAttention<ATT_GROUP, ATT_ALIGN>(TransposeAttention(nodes.AttArr[wa]), &revAttGroupsArr[wa]);
        }

        // inverse index is not needed for forward computations (inference)
        if (initType == IComputeContext::INIT_TRAIN) {
            LabelInverseIndex.BuildInverseIndex(nodes.LabelArr, nodes.LabelPtr);
        }
        LabelForwardIndex.BuildUsedIndex(nodes.LabelArr);

        WaitCudaCompute();
        Put(Stream, &LabelArr, LabelForwardIndex.RecodedLabelArr);
        Put(Stream, &UsedLabelArr, LabelForwardIndex.UsedLabels);
        Put(Stream, &LabelPtr, nodes.LabelPtr);
        SetTarget(len, nodes.Target);

        yint invLabelCount = YSize(LabelInverseIndex.InvLabelArr);

        ComputeParams.Init(Stream, len, nodes.SampleIndex, dropTable, YSize(LabelForwardIndex.UsedLabels));
        if (initType == IComputeContext::INIT_TRAIN) {
            ComputeParams.SetInvLabelCount(invLabelCount);
        }
        FinalWindows.Init(len);
        for (yint wa = 0; wa < attentionWidthCount; ++wa) {
            AttGDArr[wa]->Init(Stream, ComputeParams.GetLenBufferSize(), &attGroupsArr[wa], &revAttGroupsArr[wa]);
        }
    }

    void RunForward()
    {
        if (NeedCopyToDevice) {
            CopyModelForwardComputer->Run(Stream);
        } else {
            ForwardComputer->Run(Stream);
        }
        NeedCopyToDevice = false;
        ComputeInFly = true;
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors)
    {
        State.CopyToHost(Stream);
        Stream.Sync();
        GetAllData(State, pStateVectors);
        pStateVectors->resize(ComputeParams.Len.Get());
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction)
    {
        // far from optimal, debug purposes only
        yint len = ComputeParams.Len.Get();
        pPrediction->resize(len);
        for (yint w = 0; w < FinalWindows.GetCurrentWindowCount(); ++w) {
            TFinalLayerWindows::TWin &window = FinalWindows.GetWindow(w);
            yint winOffset = window.Offset;
            yint winLen = window.Len.Get();
            FinalLayerWindowComputerFullArr[w]->Run(Stream);
            Stream.Sync();
            TVector<TVector<half>> winPred;
            GetAllData(LogitBufHost, &winPred);
            // scale result
            for (yint t = 0; t < winLen; ++t) {
                TVector<float> &dst = (*pPrediction)[winOffset + t];
                TVector<half> &pred = winPred[t];
                Y_VERIFY(YSize(pred) >= ModelDescr.VocabSize);
                yint width = ModelDescr.VocabSize;
                dst.yresize(width);
                for (yint c = 0; c < width; ++c) {
                    dst[c] = float(pred[c]);
                }
            }
        }
    }

    void ComputeFragmentPredictions(TVector<float> *pPrediction)
    {
        yint len = ComputeParams.Len.Get();
        pPrediction->resize(len);
        for (yint w = 0; w < FinalWindows.GetCurrentWindowCount(); ++w) {
            FinalLayerWindowComputerArr[w]->Run(Stream);
        }
        TargetResProbArr.CopyToHost(Stream);
        Stream.Sync();
        GetData(TargetResProbArr, pPrediction, len);
    }

    float ComputeScore()
    {
        yint windowCount = FinalWindows.GetCurrentWindowCount();
        SumScoreComputerArr[windowCount]->Run(Stream);
        SumScore.CopyToHost(Stream);
        Stream.Sync();
        TVector<float> sumScore;
        GetAllData(SumScore, &sumScore);
        return sumScore[0] / TargetNodeCount;
    }

    float GetAvrgTrainErr()
    {
        SumTrainErr.CopyToHost(Stream);
        Stream.Sync();
        TVector<float> sumTrainErr;
        GetAllData(SumTrainErr, &sumTrainErr);
        SumTrainErr.ClearDeviceMem(Stream);
        return sumTrainErr[3] / sumTrainErr[2];
    }

    void RunBackprop()
    {
        Put(Stream, &InvLabelArr, LabelInverseIndex.InvLabelArr);
        Put(Stream, &InvLabelPos, LabelInverseIndex.InvLabelPos);
        Put(Stream, &InvLabelPosPtr, LabelInverseIndex.InvLabelPosPtr);

        yint windowCount = FinalWindows.GetCurrentWindowCount();
        if (NeedCopyToDevice) {
            CopyModelBackpropComputerArr[windowCount]->Run(Stream);
        } else {
            BackpropComputerArr[windowCount]->Run(Stream);
        }
        NeedCopyToDevice = true;
        ComputeInFly = true;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// multi GPU support (for backprop only so far)
class TMultiComputeContext : public IComputeContext
{
    enum EJob
    {
        JOB_WAIT,
        JOB_COPY_MODEL_PARAMS,
        JOB_INIT_TRAIN,
        JOB_INIT_TEST,
        JOB_BACKPROP_INIT_PARAMS,
        JOB_BACKPROP_RUN,
        JOB_GET_AVRG_TRAIN_ERR,
    };

    struct TDeviceControlThread : public TThrRefBase
    {
        TThread Worker;
        TSingleConsumerJobQueue<EJob> JobQueue;
        std::atomic<int> JobQueueSize;
        TNodesBatch Nodes;
        TVector<ui32> DropTable;
        IComputeContext::EInitType InitType = IComputeContext::INIT_TRAIN;
        float AvrgTrainErr = 0;

        TDeviceControlThread() : JobQueueSize(0)
        {
        }
        void Run(TMultiComputeContext *pThis)
        {
            Worker.Create(pThis);
        }
        void AddOp(EJob op)
        {
            JobQueueSize.fetch_add(1);
            JobQueue.Enqueue(op);
        }
        void WaitDevice()
        {
            while (JobQueueSize.load() > 0) {
                _mm_pause();
            }
        }
    };


private:
    TIntrusivePtr<IModel> Model;
    TVector<TIntrusivePtr<TSinlgeComputeContext>> CtxArr;
    bool ModelDeltaInFly = false;
    std::atomic<int> WorkerId;
    TVector<TIntrusivePtr<TDeviceControlThread>> ThrArr;
    volatile bool Exit = false;


private:
    void SetDevice(yint deviceId) const
    {
        if (YSize(CtxArr) > 1) {
            Y_VERIFY(cudaSetDevice(deviceId) == cudaSuccess);
        }
    }
public:
    void WorkerThread()
    {
        yint deviceId = WorkerId.fetch_add(1);
        SetDevice(deviceId);
        TDeviceControlThread *thr = ThrArr[deviceId].Get();
        while (!Exit) {
            EJob job;
            if (thr->JobQueue.DequeueFirst(&job)) {
                TSinlgeComputeContext *ctx = CtxArr[deviceId].Get();
                switch (job) {
                case JOB_WAIT:
                    ctx->WaitCudaCompute();
                    break;
                case JOB_COPY_MODEL_PARAMS:
                    ctx->CopyModelParams();
                    break;
                case JOB_INIT_TRAIN:
                    ctx->Init(thr->Nodes, thr->DropTable, IComputeContext::INIT_TRAIN);
                    break;
                case JOB_INIT_TEST:
                    ctx->Init(thr->Nodes, thr->DropTable, IComputeContext::INIT_TEST);
                    break;
                case JOB_BACKPROP_RUN:
                    ctx->RunBackprop();
                    break;
                case JOB_GET_AVRG_TRAIN_ERR:
                    thr->AvrgTrainErr = ctx->GetAvrgTrainErr();
                    break;
                }
                thr->JobQueueSize.fetch_add(-1);
            } else {
                _mm_pause();
            }
        }
    }
private:
    void ForeachDevice(EJob func)
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->AddOp(func);
        }
    }

    void WaitDevices()
    {
        for (yint deviceId = 0; deviceId < YSize(ThrArr); ++deviceId) {
            ThrArr[deviceId]->WaitDevice();
        }
    }

    void WaitCompute()
    {
        // correct order is to wait gpu graph completion first, then wait cpu ops (gpu graphs launch cpu compute)
        ForeachDevice(JOB_WAIT);
        WaitDevices();
        Model->WaitCompute();
    }

    void WaitDelayedCompute()
    {
        if (ModelDeltaInFly) {
            WaitCompute();
            Model->WaitDelayedCompute();
            ModelDeltaInFly = false;
        }
    }

private:
    IModel *GetModel() override
    {
        return Model.Get();
    }

    void OnParamsUpdate() override
    {
        Y_VERIFY(CtxArr[0]->GetLocalModelDescr() == Model->GetModelDescr());
        ForeachDevice(JOB_COPY_MODEL_PARAMS);
        WaitDevices();
    }

    yint GetDeviceCount() override
    {
        return YSize(CtxArr);
    }

    TModelDescr GetModelDescr() override
    {
        for (auto &ctx : CtxArr) {
            Y_ASSERT(ctx->GetLocalModelDescr() == Model->GetModelDescr());
        }
        return Model->GetModelDescr();
    }

    float GetAvrgTrainErr() override
    {
        ForeachDevice(JOB_GET_AVRG_TRAIN_ERR);
        WaitDevices();
        yint deviceCount = YSize(ThrArr);
        float sum = 0;
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            sum += ThrArr[deviceId]->AvrgTrainErr;
        }
        return sum / deviceCount;
    }

    void WaitUpdates() override
    {
        WaitDelayedCompute();
    }

    TNodesBatch &GetNodes(yint deviceId) override
    {
        ThrArr[deviceId]->WaitDevice();
        return ThrArr[deviceId]->Nodes;
    }

    TVector<ui32> &GetDropTable(yint deviceId)
    {
        ThrArr[deviceId]->WaitDevice();
        return ThrArr[deviceId]->DropTable;
    }

    void Init(yint deviceId, EInitType initType) override
    {
        if (initType == IComputeContext::INIT_TRAIN) {
            ThrArr[deviceId]->AddOp(JOB_INIT_TRAIN);
        } else {
            ThrArr[deviceId]->AddOp(JOB_INIT_TEST);
        }
    }

    void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) override
    {
        WaitDevices();
        WaitDelayedCompute();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        CtxArr[0]->ComputeFinalStateVectors(pStateVectors);
    }

    void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) override
    {
        WaitDevices();
        WaitDelayedCompute();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        CtxArr[0]->ComputeFragmentPredictions(pPrediction);
    }

    void ComputeFragmentPredictions(TVector<float> *pPrediction) override
    {
        WaitDevices();
        WaitDelayedCompute();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        CtxArr[0]->ComputeFragmentPredictions(pPrediction);
    }

    float ComputeScore() override
    {
        WaitDevices();
        WaitDelayedCompute();
        // setDevice() not needed, device 0 is default
        CtxArr[0]->RunForward();
        return CtxArr[0]->ComputeScore();
    }

    void Backprop(const TTrainingStep &step, EAddToModel addToModel) override
    {
        WaitCompute(); // modify cuda graphs when cuda queue is empty, wait non delayed updates
        Model->StartIteration(step, addToModel); // no pending matrix ops at this point expected
        ForeachDevice(JOB_BACKPROP_RUN);
        ModelDeltaInFly = true;
    }

    ~TMultiComputeContext()
    {
        Exit = true;
    }

public:
    TMultiComputeContext(TIntrusivePtr<IModel> pModel, yint nodeCount) : Model(pModel), WorkerId(0)
    {
        yint deviceCount = pModel->GetDeviceCount();
        CtxArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            SetDevice(deviceId);
            CtxArr[deviceId] = new TSinlgeComputeContext(deviceId, pModel, nodeCount);
        }
        SetDevice(0);

        ThrArr.resize(deviceCount);
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            ThrArr[deviceId] = new TDeviceControlThread();
        }
        for (yint deviceId = 0; deviceId < deviceCount; ++deviceId) {
            ThrArr[deviceId]->Run(this);
        }
    }
};

TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount)
{
    return new TMultiComputeContext(pModel, nodeCount);
}
}
