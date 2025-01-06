#include "stdafx.h"
#include "ppm_lmatch.h"
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void TestMatchLen(TLMatchChunkResult *pRes, yint ptr2, yint sameLen, TBPEToken nextToken)
{
    if (sameLen > pRes->MatchLen[ptr2]) {
        pRes->MatchLen[ptr2] = sameLen;
        pRes->NextToken[ptr2] = nextToken;
    }
}

void IndexChunk(TBPEToken docStartToken, TLMatchChunkIndex *pChunk, TLMatchChunkResult *pRes)
{
    TLMatchChunkIndex &chunk = *pChunk;
    yint len = chunk.GetLength();
    Y_VERIFY(len < 0xffffffffull);
    ClearPodArray(&pRes->MatchLen, len);
    pRes->NextToken.resize(0);
    pRes->NextToken.resize(len, UNDEFINED_TOKEN);
    pChunk->Arr.resize(len);
    for (yint t = 0; t < len; ++t) {
        chunk.Arr[t] = t;
    }
    const TBPEToken *textPtr = chunk.Text.data() + MAX_COMPACT_INDEX_MATCH_LEN;
    for (yint blkSize = 1; blkSize < len; blkSize *= 2) {
        TVector<TLMatchChunkIndex::TPos> newBlock;
        newBlock.yresize(blkSize * 2);
        for (yint offset = 0; offset < len - blkSize; offset += blkSize * 2) {
            yint p1 = offset;
            yint p2 = offset + blkSize;
            yint fin1 = p2;
            yint fin2 = Min(len, p2 + blkSize);
            yint fragLen = fin2 - offset;
            yint resPtr = 0;

            yint prevPtr1 = -1;
            for (;;) {
                yint ptr1 = chunk.Arr[p1];
                yint ptr2 = chunk.Arr[p2];
                yint sameLen;
                bool ptr1IsLower = IsPrefixLower(textPtr + ptr1, textPtr + ptr2, docStartToken, &sameLen);
                TestMatchLen(pRes, ptr2, sameLen, textPtr[ptr1 + 1]);
                if (ptr1IsLower) {
                    prevPtr1 = ptr1;
                    newBlock[resPtr++] = ptr1;
                    ++p1;
                    if (p1 == fin1) {
                        newBlock[resPtr++] = ptr2;
                        ++p2;
                        while (p2 < fin2) {
                            ptr2 = chunk.Arr[p2];
                            IsPrefixLower(textPtr + ptr1, textPtr + ptr2, docStartToken, &sameLen);
                            TestMatchLen(pRes, ptr2, sameLen, textPtr[ptr1 + 1]);
                            newBlock[resPtr++] = ptr2;
                            ++p2;
                        }
                        break;
                    }
                } else {
                    newBlock[resPtr++] = ptr2;
                    ++p2;
                    if (p2 == fin2) {
                        newBlock[resPtr++] = ptr1;
                        ++p1;
                        while (p1 < fin1) {
                            ptr1 = chunk.Arr[p1];
                            IsPrefixLower(textPtr + ptr1, textPtr + ptr2, docStartToken, &sameLen);
                            TestMatchLen(pRes, ptr2, sameLen, textPtr[ptr1 + 1]);
                            newBlock[resPtr++] = ptr1;
                            ++p1;
                        }
                        break;
                    } else {
                        yint newPtr2 = chunk.Arr[p2];
                        if (prevPtr1 >= 0) {
                            IsPrefixLower(textPtr + prevPtr1, textPtr + newPtr2, docStartToken, &sameLen);
                            TestMatchLen(pRes, newPtr2, sameLen, textPtr[prevPtr1 + 1]);
                        }
                    }
                }
            }
            
            for (yint i = 0; i < fragLen; ++i) {
                chunk.Arr[offset + i] = newBlock[i];
            }
        }
    }
}


void LookupPrevChunk(TBPEToken docStartToken, const TLMatchChunkIndex &prev, const TLMatchChunkIndex &cur, TLMatchChunkResult *pRes)
{
    yint p1 = 0; // prev ptr
    yint p2 = 0; // curptr
    yint fin1 = prev.GetLength();
    yint fin2 = cur.GetLength();
    const TBPEToken *textPrev = prev.Text.data() + MAX_COMPACT_INDEX_MATCH_LEN;
    const TBPEToken *textCur = cur.Text.data() + MAX_COMPACT_INDEX_MATCH_LEN;

    yint prevPtr1 = -1;
    for (;;) {
        yint ptr1 = prev.Arr[p1];
        yint ptr2 = cur.Arr[p2];
        yint sameLen;
        bool ptr1IsLower = IsPrefixLower(textPrev + ptr1, textCur + ptr2, docStartToken, &sameLen);
        TestMatchLen(pRes, ptr2, sameLen, textPrev[ptr1 + 1]);
        if (ptr1IsLower) {
            prevPtr1 = ptr1;
            ++p1;
            if (p1 == fin1) {
                ++p2;
                while (p2 < fin2) {
                    ptr2 = cur.Arr[p2];
                    IsPrefixLower(textPrev + ptr1, textCur + ptr2, docStartToken, &sameLen);
                    TestMatchLen(pRes, ptr2, sameLen, textPrev[ptr1 + 1]);
                    ++p2;
                }
                break;
            }
        } else {
            ++p2;
            if (p2 == fin2) {
                ++p1;
                while (p1 < fin1) {
                    ptr1 = prev.Arr[p1];
                    IsPrefixLower(textPrev + ptr1, textCur + ptr2, docStartToken, &sameLen);
                    TestMatchLen(pRes, ptr2, sameLen, textPrev[ptr1 + 1]);
                    ++p1;
                }
                break;
            } else {
                yint newPtr2 = cur.Arr[p2];
                if (prevPtr1 >= 0) {
                    IsPrefixLower(textPrev + prevPtr1, textCur + newPtr2, docStartToken, &sameLen);
                    TestMatchLen(pRes, newPtr2, sameLen, textPrev[prevPtr1 + 1]);
                }
            }
        }
    }
}


void Lookup(TBPEToken docStartToken, const TLMatchChunkIndex &chunk, const TBPEToken *prefix, yint *pBestLen, TBPEToken *pNextToken)
{
    yint beg = 0;
    yint fin = YSize(chunk.Arr);
    yint begSameLen = -1;
    yint finSameLen = -1;
    const TBPEToken *text = chunk.Text.data() + MAX_COMPACT_INDEX_MATCH_LEN;
    while (fin - beg > 1) {
        yint mid = (beg + fin) / 2;
        yint sameLen;
        if (IsPrefixLower(text + chunk.Arr[mid], prefix, docStartToken, &sameLen)) {
            beg = mid;
            begSameLen = sameLen;
        } else {
            fin = mid;
            finSameLen = sameLen;
        }
    }
    if (begSameLen < 0) {
        IsPrefixLower(text + chunk.Arr[beg], prefix, docStartToken, &begSameLen);
    }
    if (begSameLen > *pBestLen) {
        *pBestLen = begSameLen;
        *pNextToken = text[chunk.Arr[beg] + 1];
    }
    if (finSameLen > *pBestLen) {
        *pBestLen = finSameLen;
        *pNextToken = text[chunk.Arr[fin] + 1];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void ReferenceSearchMatch(const TVector<TBPEToken> &text, TVector<yint> *pMatchLen)
{
    yint texLen = YSize(text);
    ClearPodArray(pMatchLen, texLen);
    for (yint t = 0; t < texLen; ++t) {
        yint bestLen = 0;
        for (yint k = t - 1; k >= 0; --k) {
            yint maxLen = Min(MAX_COMPACT_INDEX_MATCH_LEN, k + 1);
            yint resLen = maxLen;
            for (yint len = 0; len < maxLen; ++len) {
                if (text[t - len] != text[k - len]) {
                    resLen = len;
                    break;
                }
            }
            bestLen = Max<yint>(bestLen, resLen);
        }
        (*pMatchLen)[t] = bestLen;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void TestPerf()
{
    TMersenne<ui32> rng(1313);
    const yint LEN = 10 * 1000000;
    TLMatchChunkIndex chunk;
    TLMatchChunkResult lmatch;
    for (yint k = 0; k < LEN; ++k) {
        chunk.Text.push_back(rng.Uniform(8));
    }
    for (;;) {
        NHPTimer::STime tStart;
        NHPTimer::GetTime(&tStart);
        IndexChunk(UNDEFINED_TOKEN, &chunk, &lmatch);
        DebugPrintf("%g secs\n", NHPTimer::GetTimePassed(&tStart));
    }
}


void TestLMatch()
{
    //TestPerf();
    TMersenne<ui32> rng(1313);
    TBPEToken docStartToken = UNDEFINED_TOKEN;
    for (;;) {
        const yint LEN = 2000 + rng.Uniform(2000);

        TVector<TBPEToken> text;
        for (yint i = 0; i < LEN; ++i) {
            text.push_back(rng.Uniform(8));
        }

        // reference
        TVector<yint> refMatchLen;
        ReferenceSearchMatch(text, &refMatchLen);

        // online index
        TLmatchOnlineIndex index(docStartToken);
        for (yint i = 0; i < LEN; ++i) {
            yint bestLen = 0;
            TBPEToken nextToken;
            index.Add(text[i], &bestLen, &nextToken);
            Y_VERIFY(bestLen == refMatchLen[i]);
        }

        // chunked indexing
        TLMatchChunkIndex chunk;
        TLMatchChunkResult lmatch;
        chunk.FillHead();
        chunk.Text.insert(chunk.Text.end(), text.begin(), text.end());
        chunk.Text.push_back(UNDEFINED_TOKEN);
        IndexChunk(docStartToken, &chunk, &lmatch);
        for (yint i = 0; i < LEN; ++i) {
            Y_VERIFY(lmatch.MatchLen[i] == refMatchLen[i]);
        }

        // split into 2 chunks and lookup previous
        TLMatchChunkIndex chunk1;
        TLMatchChunkIndex chunk2;
        chunk1.FillHead();
        yint halfPoint = YSize(text) / 2;
        for (yint t = 0; t <= halfPoint; ++t) {
            chunk1.Text.push_back(text[t]);
        }
        for (yint t = halfPoint - MAX_COMPACT_INDEX_MATCH_LEN; t < YSize(text); ++t) {
            chunk2.Text.push_back(text[t]);
        }
        chunk2.Text.push_back(UNDEFINED_TOKEN);
        TLMatchChunkResult lmatch1;
        TLMatchChunkResult lmatch2;
        IndexChunk(docStartToken, &chunk1, &lmatch1);
        IndexChunk(docStartToken, &chunk2, &lmatch2);
        LookupPrevChunk(docStartToken, chunk1, chunk2, &lmatch2);
        TVector<ui8> xxMatch;
        for (ui8 x : lmatch1.MatchLen) {
            xxMatch.push_back(x);
        }
        for (ui8 x : lmatch2.MatchLen) {
            xxMatch.push_back(x);
        }
        for (yint i = 0; i < LEN; ++i) {
            Y_VERIFY(xxMatch[i] == refMatchLen[i]);
        }

        printf(".");
    }
}
