#pragma once
#include "bpe.h"

constexpr yint MAX_COMPACT_INDEX_MATCH_LEN = 32;


///////////////////////////////////////////////////////////////////////////////////////////////////
inline bool IsPrefixLower(const TBPEToken *p1, const TBPEToken *p2, TBPEToken docStartToken, yint *pSameLen)
{
    // faster then sse version somehow
    for (yint k = 0; k < MAX_COMPACT_INDEX_MATCH_LEN; ++k) {
        TBPEToken c1 = p1[-k];
        TBPEToken c2 = p2[-k];
        if (c1 != c2) {
            *pSameLen = k;
            return c1 < c2;
        } else if (c1 == docStartToken) {
            *pSameLen = k + 1;
            return false;
        }
    }
    *pSameLen = MAX_COMPACT_INDEX_MATCH_LEN;
    return false; // unknown actually
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// sorted array of refs, match length is exact, position not always latest
class TLmatchOnlineIndex
{
    //static const i32 INVALID_TOKEN = -1;
    typedef ui32 TPos;

    struct TMatch
    {
        yint Pos = 0;
        yint Len = 0;

        TMatch() {}
        TMatch(yint pos, yint len) : Pos(pos), Len(len) {}
    };

private:
    TVector<TPos> Arr;
    TVector<TBPEToken> Text;
    TBPEToken DocStartToken;
    TMatch PrevBestMatch;

private:
    bool IsLower(yint pos1, yint pos2, yint *pSameLen) const
    {
        const TBPEToken *tt = Text.data();
        return IsPrefixLower(tt + pos1, tt + pos2, DocStartToken, pSameLen);
    }

    // search in interval [beg;fin)
    TMatch SearchLongestMatch(yint searchPos, yint beg, yint fin)
    {
        yint begSameLen = -1;
        yint finSameLen = -1;
        while (fin - beg > 1) {
            yint mid = (beg + fin) / 2;
            yint sameLen;
            if (IsLower(Arr[mid], searchPos, &sameLen)) {
                beg = mid;
                begSameLen = sameLen;
            } else {
                fin = mid;
                finSameLen = sameLen;
            }
        }
        if (begSameLen < 0) {
            IsLower(Arr[beg], searchPos, &begSameLen);
        }
        if (finSameLen == begSameLen) {
            // does not find latest match of this length, but still better then nothing
            return TMatch(Max<yint>(Arr[beg], Arr[fin]), finSameLen);
        }
        if (finSameLen > begSameLen) {
            return TMatch(Arr[fin], finSameLen);
        }
        return TMatch(Arr[beg], begSameLen);
    }

    TMatch SearchLongestMatch(yint searchPos)
    {
        yint sz = YSize(Arr);
        TMatch bestMatch;
        for (yint blk = 1; blk <= sz; blk *= 2) {
            if (sz & blk) {
                yint offset = sz & ~(2 * blk - 1);
                TMatch match = SearchLongestMatch(searchPos, offset, offset + blk);
                if (match.Len > bestMatch.Len || (match.Len == bestMatch.Len && match.Pos > bestMatch.Pos)) {
                    bestMatch = match;
                }
            }
        }
        return bestMatch;
    }

    void Merge(yint offset, yint blkSize)
    {
        TVector<TPos> newBlock;
        newBlock.yresize(blkSize * 2);
        yint p1 = offset;
        yint p2 = offset + blkSize;
        yint fin1 = p2;
        yint fin2 = p2 + blkSize;
        yint resPtr = 0;
        for (;;) {
            yint sameLen;
            if (IsLower(Arr[p1], Arr[p2], &sameLen)) {
                newBlock[resPtr++] = Arr[p1++];
                if (p1 == fin1) {
                    while (p2 < fin2) {
                        newBlock[resPtr++] = Arr[p2++];
                    }
                    break;
                }
            } else {
                newBlock[resPtr++] = Arr[p2++];
                if (p2 == fin2) {
                    while (p1 < fin1) {
                        newBlock[resPtr++] = Arr[p1++];
                    }
                    break;
                }
            }
        }
        for (yint i = 0; i < blkSize * 2; ++i) {
            Arr[offset + i] = newBlock[i];
        }
    }

    void Add(yint k)
    {
        yint prevSize = YSize(Arr);
        Arr.push_back(k);
        yint newSize = YSize(Arr);
        //printf("%g -> %g\n", prevSize * 1., newSize * 1.);
        for (yint blk = 2; blk <= newSize; blk *= 2) {
            //if ((prevSize & blk) == blk && (newSize & blk) == 0) {
            if ((prevSize ^ newSize) & blk) {
                yint offset = prevSize & ~(blk - 1);
                //printf("  merge [%g %g] [%g %g]\n", offset * 1., offset * 1. + blk / 2 - 1, offset * 1. + blk / 2, offset * 1. + blk - 1);
                Merge(offset, blk / 2);
            }
        }
    }

public:
    TLmatchOnlineIndex(TBPEToken docStartToken) : DocStartToken(docStartToken)
    {
        for (yint k = 0; k < MAX_COMPACT_INDEX_MATCH_LEN; ++k) {
            Text.push_back(UNDEFINED_TOKEN);
        }
    }

    void Clear()
    {
        Arr.resize(0);
        Text.resize(MAX_COMPACT_INDEX_MATCH_LEN);
        PrevBestMatch = TMatch();
    }

    void Add(TBPEToken cc, yint *pBestLen, TBPEToken *pNextToken)
    {
        yint indexPos = YSize(Text);
        Y_VERIFY(indexPos < 0xffffffffll);
        Text.push_back(cc);
        TMatch bestMatch;
        // try to extend previous match
        if (PrevBestMatch.Len > 0) {
            if (cc == Text[PrevBestMatch.Pos + 1]) {
                bestMatch = PrevBestMatch;
                bestMatch.Len += 1;
                bestMatch.Pos += 1;
            }
        }
        // search if extending failed
        if (bestMatch.Len == 0) {
            bestMatch = SearchLongestMatch(indexPos);
        }
        *pBestLen = bestMatch.Len;
        if (bestMatch.Len > 0) {
            *pNextToken = Text[bestMatch.Pos + 1];
        }
        PrevBestMatch = bestMatch;
        Add(indexPos);
    }

    const TBPEToken *GetCurrentPrefix() const { return Text.data() + YSize(Text) - 1; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TLMatchChunkIndex
{
    // text has additional MAX_COMPACT_INDEX_MATCH_LEN tokens of history, so we can look back without buffer overrun
    // text has additional token at the end, so that looking up next is always within borders
    typedef ui32 TPos;
    TVector<TBPEToken> Text;
    TVector<TPos> Arr;
    SAVELOAD(Text, Arr);

    yint GetLength() const
    {
        return YSize(Text) - MAX_COMPACT_INDEX_MATCH_LEN - 1;
    }
    void FillHead()
    {
        for (yint k = 0; k < MAX_COMPACT_INDEX_MATCH_LEN; ++k) {
            Text.push_back(UNDEFINED_TOKEN);
        }
    }
};

struct TLMatchChunkResult
{
    TVector<ui8> MatchLen;
    TVector<TBPEToken> NextToken;
    SAVELOAD(MatchLen, NextToken);
};


void IndexChunk(TBPEToken docStartToken, TLMatchChunkIndex *pChunk, TLMatchChunkResult *pRes);
void LookupPrevChunk(TBPEToken docStartToken, const TLMatchChunkIndex &prev, const TLMatchChunkIndex &cur, TLMatchChunkResult *pRes);
void Lookup(TBPEToken docStartToken, const TLMatchChunkIndex &chunk, const TBPEToken *prefix, yint *pBestLen, TBPEToken *pNextToken);
