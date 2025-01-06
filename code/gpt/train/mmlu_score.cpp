#include "stdafx.h"
#include "mmlu_score.h"
#include <gpt/data/data.h>
#include <gpt/data/fragment_gen.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/att/sliding_window.h>


static bool ReadVec(TFileStream &f, TVector<ui16> *p)
{
    ui16 len = 0;
    if (f.Read(&len, 2) != 2) {
        return false;
    }
    p->resize(len);
    f.Read(p->data(), len * 2);
    return true;
}


struct TChoiceSample
{
    TVector<TFragment> FragArr;
    TVector<TVector<yint>> FragLMatchLen;
    yint Correct = 0;
    yint CtxSize = 0;

    yint GetNodesRequired() const
    {
        yint res = 0;
        for (const TFragment &frag : FragArr) {
            res += YSize(frag.Text);
        }
        return res;
    }
};


void ComputeChoiceScore(const TModelParams &params, const TString &queryFile, yint docStartToken, yint fragmentStartToken, const TString &lmIndexDir)
{
    TFileStream f(IO_READ, queryFile);
    Y_VERIFY(f.IsValid() && "file not found");

    constexpr yint MAX_NODE_COUNT = 16384 / 2;
    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, MAX_NODE_COUNT);
    pCtx->SetParams(params);

    bool usePPM = params.GetModelDescr().HasFlag(MPF_PPM);
    bool useLMatch = params.ModelDescr.HasFlag(MPF_LMATCH);
    TIntrusivePtr<TLMatchSearch> lmSearch; // new TLMatchSearch(lmIndexDir, docStartToken);  // do not use online lmatch index
    TFragmentGen fgen(usePPM, useLMatch, docStartToken, lmSearch);

    TVector<TChoiceSample> allSamples;
    DebugPrintf("Generate fragments for samples\n"); fflush(0);
    for (;;) {
        TVector<ui16> ctx;
        if (!ReadVec(f, &ctx)) {
            break;
        }
        TChoiceSample sample;
        sample.CtxSize = YSize(ctx) + 1; // + 1 for fragmentStartToken
        ui16 num_cont = 0;
        f.Read(&num_cont, 2);
        for (yint k = 0; k < num_cont; ++k) {
            TVector<ui16> cont;
            ReadVec(f, &cont);
            // make fragment
            fgen.Clear();
            fgen.AddToken(fragmentStartToken);
            for (TBPEToken x : ctx) {
                fgen.AddToken(x);
            }
            for (TBPEToken x : cont) {
                fgen.AddToken(x);
            }
            TFragment frag;
            fgen.FillFragment(&frag, MAX_NODE_COUNT, fragmentStartToken);
            Y_VERIFY(YSize(frag.Text) > 1);
            // fill fragment target tokens
            for (yint t = 1; t < YSize(frag.Text); ++t) {
                frag.Target.push_back(frag.Text[t]);
            }
            frag.Target.push_back(UNDEFINED_TOKEN);
            Y_ASSERT(YSize(frag.Target) == YSize(frag.Text));
            sample.FragArr.push_back(frag);
            sample.FragLMatchLen.push_back(fgen.GetLMatchBestLen());
        }
        ui16 correct = 0;
        f.Read(&correct, 2);
        sample.Correct = correct;
        Y_VERIFY(!f.IsFailed() && "file corrupted");
        allSamples.push_back(sample);
        //printf("."); fflush(0);
    }
    //printf("\n"); fflush(0);

    // ondisk lmatch
    if (useLMatch) {
        if (!lmIndexDir.empty()) {
            printf("Prepare lmatch lookup\n"); fflush(0);
            yint dst = 0;
            TLMatchChunkIndex chunk;
            chunk.FillHead();
            for (TChoiceSample &s : allSamples) {
                for (yint k = 0; k < YSize(s.FragArr); ++k) {
                    TFragment &frag = s.FragArr[k];
                    chunk.Text.push_back(docStartToken);
                    for (yint t = 1; t < frag.GetLength(); ++t) {
                        chunk.Text.push_back(frag.Text[t]);
                    }
                }
            }
            chunk.Text.push_back(UNDEFINED_TOKEN);

            TLMatchChunkResult res;
            IndexChunk(docStartToken, &chunk, &res);

            // fill correct lmatch without cross query reuse
            dst = 0;
            for (TChoiceSample &s : allSamples) {
                for (yint k = 0; k < YSize(s.FragArr); ++k) {
                    TFragment &frag = s.FragArr[k];
                    for (yint t = 0; t < frag.GetLength(); ++t) {
                        res.MatchLen[dst] = s.FragLMatchLen[k][t];
                        res.NextToken[dst] = frag.LMatch[t];
                        ++dst;
                    }
                }
            }

            // lookup history
            LookupLMatch(lmIndexDir, docStartToken, chunk, &res);

            // fill looked up next tokens
            dst = 0;
            for (TChoiceSample &s : allSamples) {
                for (yint k = 0; k < YSize(s.FragArr); ++k) {
                    TFragment &frag = s.FragArr[k];
                    for (yint t = 0; t < frag.GetLength(); ++t) {
                        frag.LMatch[t] = res.NextToken[dst];
                        ++dst;
                    }
                }
            }
        } else {
            DebugPrintf("no lmatch index specified for useLMatch model, results are suboptimal\n");
        }
    }

    double totalSamples = 0;
    double correctSamples = 0;
    yint sampleId = 0;
    DebugPrintf("Compute preferences\n"); fflush(0);
    while (sampleId < YSize(allSamples)) {
        TVector<TFragment> batchFragArr;
        yint begSampleId = sampleId;
        yint nodeCount = 0;
        while (sampleId < YSize(allSamples)) {
            const TChoiceSample &s = allSamples[sampleId];
            nodeCount += s.GetNodesRequired();
            if (nodeCount > MAX_NODE_COUNT) {
                break;
            }
            batchFragArr.insert(batchFragArr.end(), s.FragArr.begin(), s.FragArr.end());
            ++sampleId;
            //break;
        }

        MakeTest(batchFragArr, pCtx.Get(), MAIN_DEVICE);
        TVector<float> predArr;
        pCtx->ComputeFragmentPredictions(&predArr);

        yint ptr = 0;
        for (yint x = begSampleId; x < sampleId; ++x) {
            const TChoiceSample &s = allSamples[x];

            yint topChoice = 0;
            double topScore = -1e38;
            for (yint k = 0; k < YSize(s.FragArr); ++k) {
                const TFragment &frag = s.FragArr[k];
                double sumLoss = 0;
                double count = 0;
                for (yint t = s.CtxSize; t < YSize(frag.Text); ++t) {
                    sumLoss += log(predArr[ptr + t - 1]);
                    count += 1;
                }
                double score = sumLoss / count;
                if (score > topScore) {
                    topScore = score;
                    topChoice = k;
                }
                ptr += YSize(frag.Text);
            }

            if (topChoice == s.Correct) {
                correctSamples += 1;
                printf("+"); fflush(0);
            } else {
                printf("."); fflush(0);
            }
            totalSamples += 1;
        }
    }
    printf("\n");
    DebugPrintf("%g%% correct\n", correctSamples * 100. / totalSamples);
    fflush(0);
}
