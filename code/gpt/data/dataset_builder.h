#pragma once
#include "dataset.h"
#include "ppm_window.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
const TString DATASET_INDEX_FNAME = "/index.bin";
const TString DATASET_HDR_FNAME = "/index_hdr.bin";
const TString DATASET_PPM_FNAME = "/index_ppm.bin";
const TString LMATCH_INDEX_FNAME = "/lmatch.bin";


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TDatasetParams
{
    TVector<double> FreqArr;
    yint TotalUtf8Chars = 0;
    yint TotalTokens = 0;
    yint BytesPerToken = 0;
    TVector<TDatasetWeightedSpan> TrainSpans;
    TVector<TDatasetWeightedSpan> TestSpans;
    SAVELOAD(FreqArr, TotalUtf8Chars, TotalTokens, BytesPerToken, TrainSpans, TestSpans);

    TDatasetParams() {}
    TDatasetParams(yint vocabSize)
    {
        ClearPodArray(&FreqArr, vocabSize);
        BytesPerToken = (vocabSize > 65530) ? 3 : 2;
    }
    void CountDocset(const TVector<TBPEToken> &data, yint offset, yint utf8charCount, float testFraction)
    {
        yint vocabSize = YSize(FreqArr);
        for (ui64 x : data) {
            Y_VERIFY(x < vocabSize);
            FreqArr[x] += 1;
        }
        yint len = YSize(data);
        if (testFraction == 0) {
            TrainSpans.push_back(TDatasetWeightedSpan(len, -1, offset, offset + len));
        } else if (testFraction == 1) {
            TestSpans.push_back(TDatasetWeightedSpan(len, -1, offset, offset + len));
        } else {
            // test is always last part of docset
            yint testLen = len * testFraction;
            yint trainLen = len - testLen;
            TrainSpans.push_back(TDatasetWeightedSpan(trainLen, -1, offset, offset + trainLen));
            TestSpans.push_back(TDatasetWeightedSpan(testLen, -1, offset + trainLen, offset + len));
        }
        TotalUtf8Chars += utf8charCount;
        TotalTokens += len;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TDatasetBuilder : public TThrRefBase
{
    TIntrusivePtr<TDataset> Dataset;
    TVector<double> FreqArr;
    yint TotalUtf8Chars = 0;
    yint TotalTokens = 0;

private:
    TBPEToken GetDocStartToken()
    {
        yint tkn = Dataset->Stats.DocStartToken;
        if (tkn < 0) {
            return UNDEFINED_TOKEN;
        } else {
            return tkn;
        }
    }

    void AddParams(yint docsetId, const TDatasetParams &params, float weight)
    {
        Y_VERIFY(YSize(FreqArr) == YSize(params.FreqArr));
        for (yint x = 0; x < YSize(params.FreqArr); ++x) {
            FreqArr[x] += params.FreqArr[x];
        }
        for (const TDatasetWeightedSpan &span : params.TrainSpans) {
            Dataset->TrainSpans.push_back(TDatasetWeightedSpan(span.Weight * weight, docsetId, span.SpanStart, span.SpanFinish));
        }
        for (const TDatasetWeightedSpan &span : params.TestSpans) {
            Dataset->TestSpans.push_back(TDatasetWeightedSpan(span.Weight * weight, docsetId, span.SpanStart, span.SpanFinish));
        }
        TotalUtf8Chars += params.TotalUtf8Chars;
        TotalTokens += params.TotalTokens;
    }

    void ComputeOnDiskLMatch(const TString &lmIndexDir);
    void ComputeInMemoryLMatch();

public:
    TDatasetBuilder(bool usePPM, bool useLMatch, yint vocabSize, yint docStartToken, yint fragmentStartToken)
    {
        Dataset = new TDataset(usePPM, useLMatch, vocabSize, docStartToken, fragmentStartToken);
        ClearPodArray(&FreqArr, vocabSize);
    }

    TDatasetBuilder(bool usePPM, bool useLMatch, const TTokenizer &tokenizer)
    {
        yint vocabSize = tokenizer.GetVocabSize();
        yint docStartToken = tokenizer.HasDocStartToken() ? tokenizer.GetDocStartToken() : -1;
        yint fragmentStartToken = tokenizer.GetFragmentStartToken();
        Dataset = new TDataset(usePPM, useLMatch, vocabSize, docStartToken, fragmentStartToken);
        ClearPodArray(&FreqArr, vocabSize);
    }

    void AddTokenizedDocset(const TVector<TBPEToken> &data, const TDatasetParams &params, float weight)
    {
        yint docsetId = YSize(Dataset->DocsetArr);
        TDataset::TDocumentSet &docset = *Dataset->DocsetArr.insert(Dataset->DocsetArr.end());
        docset.Text = data;
        if (Dataset->Stats.UsePPM) {
            ComputeWindowPPM(docset.Text, &docset.PPM, GetDocStartToken());
        }
        docset.TotalTokens = YSize(docset.Text);
        AddParams(docsetId, params, weight);
    }

    void AddIndexedDocset(const TString &dir, const TDatasetParams &params, yint vocabSize, float weight)
    {
        Y_VERIFY(vocabSize == Dataset->Stats.VocabSize);
        yint docsetId = YSize(Dataset->DocsetArr);
        TDataset::TDocumentSet &docset = *Dataset->DocsetArr.insert(Dataset->DocsetArr.end());
        docset.IndexFilename = dir + DATASET_INDEX_FNAME;
        if (Dataset->Stats.UsePPM) {
            docset.PPMIndexFilename = dir + DATASET_PPM_FNAME;
        }
        docset.BytesPerToken = params.BytesPerToken;
        docset.TotalTokens = params.TotalTokens;
        AddParams(docsetId, params, weight);
    }

    TIntrusivePtr<TDataset> MakeDataset(const TString &lmIndexDir)
    {
        yint vocabSize = Dataset->Stats.VocabSize;
        Dataset->Stats.Bias.resize(vocabSize);
        for (yint c = 0; c < vocabSize; ++c) {
            Dataset->Stats.Bias[c] = log2(FreqArr[c] + 0.5);
        }
        Dataset->Stats.Compression = TotalTokens / (TotalUtf8Chars + 0.);
        Dataset->Stats.HasTest = !Dataset->TestSpans.empty();
        if (Dataset->Stats.UseLMatch) {
            if (Dataset->IsInMemory()) {
                ComputeInMemoryLMatch();
            } else if (Dataset->IsOnDisk()) {
                ComputeOnDiskLMatch(lmIndexDir);
            } else {
                Y_VERIFY(0 && "can not use lmatch for mixed in memory and on disk docsets");
            }
        }
        return Dataset.Release();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TDataset> MakeCharDataset(TTokenizer *pTokenizer, const TVector<char> &text, float testFraction, bool usePPM, bool useLMatch);
void AddDocset(TIntrusivePtr<TDatasetBuilder> pBuilder, const TTokenizer &tokenizer, const TVector<TVector<char>> &docSet, float weight, float testFraction);

void AddIndexedDocset(TDatasetBuilder *pBuilder, const TString &dir, float weight);
void IndexDocsetDir(const TString &dir, const TTokenizer &tokenizer, bool usePPM, float testFraction);
void IndexTokenizedDir(const TString &dir, yint vocabSize, yint docStartToken, bool usePPM, float testFraction, yint tokenWidth, yint headerSize);
