#pragma once
#include "data.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDatasetWeightedSpan
{
    double Weight = 0;
    yint DocsetId = 0;
    yint SpanStart = 0;
    yint SpanFinish = 0;

    TDatasetWeightedSpan() {}
    TDatasetWeightedSpan(double w, yint id, yint start, yint finish) : Weight(w), DocsetId(id), SpanStart(start), SpanFinish(finish) {}
};


class TDataset : public IDataSource
{
    struct TDocumentSet
    {
        TVector<TBPEToken> Text;
        TVector<TBPEToken> PPM;
        TVector<TBPEToken> LMatch;
        TString IndexFilename;
        TString PPMIndexFilename;
        TString LMatchFilename;
        yint BytesPerToken = 0;
        yint TotalTokens = 0;
        yint LMatchOffset = 0;
        SAVELOAD(Text, PPM, IndexFilename, PPMIndexFilename, LMatchFilename, BytesPerToken, TotalTokens, LMatchOffset);
        TIntrusivePtr<TPackedBPETokenReader> Reader;
        TIntrusivePtr<TPackedBPETokenReader> PPMReader;
        TIntrusivePtr<TPackedBPETokenReader> LMatchReader;

        void FillFragment(bool usePPM, bool useLMatch, yint offset, yint fragLen, TFragment *p)
        {
            *p = TFragment();
            p->Offset = offset;
            if (IsInMemory()) {
                for (yint t = 0; t < fragLen; ++t) {
                    p->Text.push_back(Text[offset + t]);
                    if (usePPM) {
                        p->PPM.push_back(PPM[offset + t]);
                    }
                    if (useLMatch) {
                        p->LMatch.push_back(LMatch[offset + t]);
                    }
                    p->Target.push_back(Text[offset + t + 1]);
                }
            } else {
                if (Reader.Get() == 0) {
                    Reader = new TPackedBPETokenReader(IndexFilename, BytesPerToken);
                    if (!PPMIndexFilename.empty()) {
                        PPMReader = new TPackedBPETokenReader(PPMIndexFilename, BytesPerToken);
                    }
                    if (!LMatchFilename.empty()) {
                        LMatchReader = new TPackedBPETokenReader(LMatchFilename, BytesPerToken);
                    }
                }
                TVector<TBPEToken> buf;
                Reader->Read(offset, fragLen + 1, &buf);
                for (yint t = 0; t < fragLen; ++t) {
                    p->Text.push_back(buf[t]);
                    p->Target.push_back(buf[t + 1]);
                }
                if (usePPM) {
                    PPMReader->Read(offset, fragLen, &p->PPM);
                }
                if (useLMatch) {
                    LMatchReader->Read(LMatchOffset + offset, fragLen, &p->LMatch);
                }
            }
        }

        bool IsInMemory() const
        {
            return IndexFilename.empty();
        }
    };

    TVector<TDocumentSet> DocsetArr;
    TDataStats Stats;
    TVector<TDatasetWeightedSpan> TrainSpans;
    TVector<TDatasetWeightedSpan> TestSpans;
public:
    SAVELOAD(DocsetArr, Stats, TrainSpans, TestSpans);

private:
    template <class TRng>
    void MakeRandomFragment(TRng &rng,
        yint docsetId, yint spanStart, yint spanFinish,
        yint fragLen, TFragment *p)
    {
        TDocumentSet &docset = DocsetArr[docsetId];
        if (spanFinish - spanStart <= fragLen - 1) {
            docset.FillFragment(Stats.UsePPM, Stats.UseLMatch, spanStart, spanFinish - spanStart - 1, p);
        } else {
            yint offset = spanStart + rng.Uniform(spanFinish - spanStart - fragLen - 1);
            docset.FillFragment(Stats.UsePPM, Stats.UseLMatch, offset, fragLen, p);
        }
        if (Stats.FragmentStartToken >= 0 && YSize(p->Text) > 0) {
            p->Text[0] = Stats.FragmentStartToken;
            // do not train on first token, it differs from subsequent tokens, should predict bias[] anyway
            p->Target[0] = UNDEFINED_TOKEN;
        }
    }

public:
    TDataset() {}

    TDataset(bool usePPM, bool useLMatch, yint vocabSize, yint docStartToken, yint fragmentStartToken)
    {
        Stats.UsePPM = usePPM;
        Stats.UseLMatch = useLMatch;
        Stats.VocabSize = vocabSize;
        Stats.DocStartToken = docStartToken;
        Stats.FragmentStartToken = fragmentStartToken;
        DocsetArr.reserve(10000);
    }

    const TDataStats &GetStats() const override
    {
        return Stats;
    }

    void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) override
    {
        TXRng rng(rngSeed);
        for (yint k = 0; k < 17; ++k) {
            rng.GenRand();
        }
        const TVector<TDatasetWeightedSpan> &spanArr = (trt == TRAIN) ? TrainSpans : TestSpans;
        Y_VERIFY(!spanArr.empty());
        for (yint k = 0; k < fragCount; ++k) {
            // use gumbel max trick
            float best = -1e38f;
            const TDatasetWeightedSpan *bestSpan = &spanArr[0];
            for (yint k = 0; k < YSize(spanArr); ++k) {
                float score = spanArr[k].Weight / -log(rng.GenRandReal3());
                if (score > best) {
                    best = score;
                    bestSpan = &spanArr[k];
                }
            }
            TFragment &frag = *pFragArr->insert(pFragArr->end());
            MakeRandomFragment(rng, bestSpan->DocsetId, bestSpan->SpanStart, bestSpan->SpanFinish, len, &frag);
        }
    }

    bool IsInMemory() const
    {
        bool res = true;
        for (const TDocumentSet &ds : DocsetArr) {
            res &= ds.IsInMemory();
        }
        return res;
    }

    bool IsOnDisk() const
    {
        bool res = true;
        for (const TDocumentSet &ds : DocsetArr) {
            res &= !ds.IsInMemory();
        }
        return res;
    }

    friend class TDatasetBuilder;
};
