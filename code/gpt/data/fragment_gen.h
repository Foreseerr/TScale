#pragma once
#include "bpe.h"
#include "ppm_lmatch.h"
#include "ppm_window.h"
#include "data.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
class TLMatchSearch : public TThrRefBase
{
    TVector<TLMatchChunkIndex> ChunkArr;
    TBPEToken DocStartToken = UNDEFINED_TOKEN;
public:
    TLMatchSearch(const TString &lmIndexDir, TBPEToken docStartToken);
    void Search(const TBPEToken *prefix, yint *bestLen, TBPEToken *pNextToken) const;
};

void LookupLMatch(const TString &lmIndexDir, TBPEToken docStartToken, const TLMatchChunkIndex &chunk, TLMatchChunkResult *pRes);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TFragmentGen
{
    bool UsePPM = false;
    bool UseLMatch = false;
    TVector<TBPEToken> Text;
    TVector<TBPEToken> PPM;
    TVector<TBPEToken> LMatch;
    TVector<yint> LMatchLen;
    TWindowPPMIndex WindowIndex;
    TLmatchOnlineIndex LMatchIndex;
    TIntrusivePtr<TLMatchSearch> LMatchSearch;
public:
    TFragmentGen(bool usePPM, bool useLMatch, yint docStartToken, TIntrusivePtr<TLMatchSearch> lmSearch) : UsePPM(usePPM), UseLMatch(useLMatch), LMatchIndex(docStartToken), LMatchSearch(lmSearch)
    {
    }

    void Clear()
    {
        Text.clear();
        PPM.clear();
        LMatch.clear();
        LMatchLen.clear();
        WindowIndex.Clear();
        LMatchIndex.Clear();
    }

    void AddToken(TBPEToken token)
    {
        Text.push_back(token);
        if (UsePPM) {
            yint bestLen = 0;
            yint bestPos = 0;
            WindowIndex.IndexPos(Text, YSize(Text) - 1, &bestLen, &bestPos);
            if (bestLen > 0) {
                PPM.push_back(Text[bestPos + 1]);
            } else {
                PPM.push_back(UNDEFINED_TOKEN);
            }
        }
        if (UseLMatch) {
            yint bestLen = 0;
            TBPEToken nextToken = UNDEFINED_TOKEN;
            LMatchIndex.Add(token, &bestLen, &nextToken);
            if (LMatchSearch.Get()) {
                LMatchSearch->Search(LMatchIndex.GetCurrentPrefix(), &bestLen, &nextToken);
            }
            LMatch.push_back(nextToken);
            LMatchLen.push_back(bestLen);
        }
    }

    void FillFragment(TFragment *pFrag, yint maxLen, yint fragmentStartToken) const
    {
        *pFrag = TFragment();
        yint start = Max<yint>(0, YSize(Text) - maxLen);
        yint fin = YSize(Text);
        for (yint t = start; t < fin; ++t) {
            pFrag->Text.push_back(Text[t]);
            if (UsePPM) {
                pFrag->PPM.push_back(PPM[t]);
            }
            if (UseLMatch) {
                pFrag->LMatch.push_back(LMatch[t]);
            }
        }
        if (fragmentStartToken >= 0 && YSize(pFrag->Text) > 0) {
            pFrag->Text[0] = fragmentStartToken;
        }
    }

    const TVector<yint> &GetLMatchBestLen() const
    {
        return LMatchLen;
    }
};
