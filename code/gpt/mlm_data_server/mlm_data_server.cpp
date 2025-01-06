#include "stdafx.h"
#include <lib/net/tcp_net.h>
#include <gpt/data/net_data.h>
#include <lib/file/dir.h>


const float TRAIN_DROP_RATE = 0.4f;


static void LoadBatches(yint vocabSize, const TString &dir, TVector<TFragment> *pRes)
{
    TVector<TFindFileResult> allFiles;
    FindAllFiles(dir, &allFiles);
    for (TFindFileResult &ff : allFiles) {
        if (ff.IsDir) {
            continue;
        }
        TVector<char> data;
        ReadWholeFile(Sprintf("%s/%s", dir.c_str(), ff.Name.c_str()), &data);
        int *x = (int*)data.data();
        int id = x[0];
        Y_VERIFY(id == 1000000);
        constexpr yint BATCH = 512;
        constexpr yint LEN = 512;
        //int nb_q = x[1] / BATCH;
        //int nb_tgt = x[2] / BATCH;
        yint text_len = x[1];
        yint target_len = x[2];
        Y_VERIFY(text_len == target_len);
        yint len = text_len / BATCH;
        x += 3;
        for (yint n = 0; n < BATCH; ++n) {
            //int *text = x + n * (nb_q + nb_tgt);
            //int *target = x + n * (nb_q + nb_tgt) + nb_q;
            int *text = x + n * len;
            int *target = x + text_len + n * len;
            TFragment frag;
            for (int t = 0; t < len; ++t) {
                frag.Text.push_back(text[t] >= 0 ? text[t] : UNDEFINED_TOKEN);
                frag.Target.push_back(target[t] >= 0 ? target[t] : UNDEFINED_TOKEN);
            }
            for (TBPEToken x : frag.Text) {
                Y_VERIFY(x == UNDEFINED_TOKEN || x < vocabSize);
            }
            for (TBPEToken x : frag.Target) {
                Y_VERIFY(x == UNDEFINED_TOKEN || x < vocabSize);
            }
            Y_VERIFY(YSize(frag.Text) <= LEN);
            while (YSize(frag.Text) < LEN) {
                frag.Text.push_back(UNDEFINED_TOKEN);
                frag.Target.push_back(UNDEFINED_TOKEN);
            }
            while (!frag.Text.empty() && frag.Text.back() == 0) {
                frag.Text.pop_back();
                frag.Target.pop_back();
            }
            pRes->push_back(frag);
        }
    }
}


static void ReconstructBatches(TVector<TFragment> *pRes)
{
    for (TFragment &frag : *pRes) {
        for (yint t = 0; t < YSize(frag.Text); ++t) {
            if (frag.Target[t] != UNDEFINED_TOKEN) {
                frag.Text[t] = frag.Target[t];
                frag.Target[t] = UNDEFINED_TOKEN;
            }
        }
    }
}


class TFragmentLoader : public IDataSource
{
    TDataStats Stats;
    TVector<TFragment> TrainFragments, TestFragments;

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
        const TVector<TFragment> &src = (trt == TRAIN) ? TrainFragments : TestFragments;
        for (yint k = 0; k < fragCount; ++k) {
            TFragment frag = src[rng.Uniform(YSize(src))];
            frag.Truncate(len);
            if (trt == TRAIN) {
                for (yint t = 0; t < YSize(frag.Text); ++t) {
                    if (rng.GenRandReal3() < TRAIN_DROP_RATE) {
                        frag.Target[t] = frag.Text[t];
                        frag.Text[t] = 1; // UNDEFINED_TOKEN is more reasonable but keep compatibility
                    }
                }
            }
            pFragArr->push_back(frag);
        }
    }

public:
    TFragmentLoader(const TString &trainPath, const TString &testPath)
    {
        Stats.Compression = 1;
        Stats.VocabSize = 70000;
        ClearPodArray(&Stats.Bias, Stats.VocabSize);
        LoadBatches(Stats.VocabSize, trainPath, &TrainFragments);
        ReconstructBatches(&TrainFragments);
        LoadBatches(Stats.VocabSize, testPath, &TestFragments);
        Stats.HasTest = !TestFragments.empty();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace NNet;
int main(int argc, char **argv)
{
    TIntrusivePtr<IDataSource> data = new TFragmentLoader("D:/mlm/train_batches", "D:/mlm/test_batches");
    //TIntrusivePtr<IDataSource> data = new TFragmentLoader("D:/mlm/test_batches", "D:/mlm/test_batches");
    TIntrusivePtr<ITcpSendRecv> net = CreateTcpSendRecv();
    DebugPrintf("Start serving queries\n"); fflush(0);
    RunDataServer(net, data);
    return 0;
}
