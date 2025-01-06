#pragma once
#include "bpe.h"
#include <gpt/rng/xrng.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
inline void TruncateArray(TVector<TBPEToken> *p, yint len)
{
    if (YSize(*p) > len) {
        p->resize(len);
    }
}


struct TFragment
{
    yint Offset = 0;
    TVector<TBPEToken> Text;
    TVector<TBPEToken> PPM;
    TVector<TBPEToken> LMatch;
    TVector<TBPEToken> Target;
    SAVELOAD(Offset, Text, PPM, LMatch, Target);

    yint GetLength() const { return YSize(Text); }
    void Truncate(yint len)
    {
        TruncateArray(&Text, len);
        TruncateArray(&PPM, len);
        TruncateArray(&LMatch, len);
        TruncateArray(&Target, len);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IDataSource : public TThrRefBase
{
    struct TDataStats
    {
        bool UsePPM = false;
        bool UseLMatch = false;
        float Compression = 0;
        yint VocabSize = 0;
        yint DocStartToken = -1;
        yint FragmentStartToken = -1;
        TVector<float> Bias;
        bool HasTest = false;

        SAVELOAD(UsePPM, UseLMatch, Compression, VocabSize, DocStartToken, FragmentStartToken, Bias, HasTest);
    };

    enum ETrainTest
    {
        TRAIN,
        TEST,
    };

    virtual const TDataStats &GetStats() const = 0;
    virtual void SampleFragments(ETrainTest trt, yint rngSeed, yint fragCount, yint len, TVector<TFragment> *pFragArr) = 0;
};

