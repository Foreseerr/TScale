#include "stdafx.h"
#include "make_bin_features.h"
#include "doc_info.h"
#include <algorithm>


static bool IsBinaryFeature(const TVector<TDocInfo> &docInfos, int idx, float *minValue, float *maxValue)
{
    *minValue = 1e38f;
    *maxValue = -1e38f;
    bool isBinary = true;
    for (int i = 0; i < YSize(docInfos); ++i) {
        float f = docInfos[i].Factors[idx];
        if (f != 0 && f != 1)
            isBinary = false;
        *maxValue = Max(*maxValue, f);
        *minValue = Min(*minValue, f);
    }
    return isBinary;
}

static void AddReason(TVector<TVector<bool> > *res, const TVector<TDocInfo> &docInfos, int idx, float border)
{
    res->resize(res->size() + 1);
    TVector<bool> &dst = (*res)[res->size() - 1];
    dst.resize(docInfos.size());
    for (int i = 0; i < YSize(docInfos); ++i) {
        const TDocInfo &di = docInfos[i];
        dst[i] = di.Factors[idx] > border;
    }
}

struct TBinaryFeature
{
    yint Start;
    yint End;
    float RightLimit;

    yint Size() const
    {
        return End - Start;
    }

    bool operator>(const TBinaryFeature &bf) const
    {
        return Size() > bf.Size();
    }

    bool Split(const TVector<float> &feat, TBinaryFeature *res)
    {
        yint mid = Start + Size() / 2;
        float median = feat[mid];
        yint lb = std::lower_bound(feat.begin() + Start, feat.begin() + mid, median) - feat.begin();
        yint ub = std::upper_bound(feat.begin() + mid, feat.begin() + End, median) - feat.begin();
        yint split = (((double)(End - lb)) * ((double)(lb - Start)) <
            ((double)(End - ub)) * ((double)(ub - Start))) ? ub : lb;
        //yint split = lb != Start ? lb : ub;
        bool splitted = split != Start && split != End;
        if (splitted) {
            TBinaryFeature s = { Start, split, feat[split - 1] };
            *res = s;
            Start = split;
        }
        return splitted;
    }

    static bool CompareSizeGreater(const TBinaryFeature &a, const TBinaryFeature &b)
    {
        return a > b;
    }
    static bool CompareLimitLess(const TBinaryFeature &a, const TBinaryFeature &b)
    {
        return a.RightLimit < b.RightLimit;
    }
};


static void MedianGrid(TVector<float> &feat, int binCount, TVector<float> *conditions)
{
    float maxVal = Max(1.0f, feat.back());
    TBinaryFeature current = { 0, YSize(feat), maxVal };
    TVector<TBinaryFeature> row;

    row.push_back(current);
    while (YSize(row) < (yint)binCount) {
        Sort(row.begin(), row.end(), TBinaryFeature::CompareSizeGreater);
        yint bf = 0;
        while (bf < YSize(row)) {
            TBinaryFeature split;
            if (row[bf].Split(feat, &split)) {
                row.push_back(split);
                break;
            }
            bf++;
        }
        if (bf >= YSize(row))
            break;
    }
    Sort(row.begin(), row.end(), TBinaryFeature::CompareLimitLess);
    conditions->resize(0);
    for (int i = 0; i < YSize(row); ++i) {
        conditions->push_back(row[i].RightLimit);
    }
}


void ExtractBoolsFromDocInfo(const TVector<TDocInfo> &docInfos, TVector<TVector<bool> > *pRes)
{
    pRes->clear();
    if (docInfos.empty())
        return;
    int reasonCount = docInfos[0].Factors.size();
    for (int nReason = 0; nReason < reasonCount; ++nReason) {
        float minValue = 0, maxValue = 0;
        if (IsBinaryFeature(docInfos, nReason, &minValue, &maxValue)) {
            if (minValue != maxValue)
                AddReason(pRes, docInfos, nReason, 0);
        } else {
            TVector<float> vals;
            for (int i = 0; i < YSize(docInfos); ++i) {
                vals.push_back(docInfos[i].Factors[nReason]);
            }
            Sort(vals.begin(), vals.end());

            TVector<float> borders;

            //// FAST
            //const int N_STEPS = 16;
            ////for (int i = 0; i < N_STEPS; ++i) {
            ////    borders.push_back(minValue + i * (maxValue - minValue) / N_STEPS);
            ////}
            //for (int i = 0; i < N_STEPS; ++i) {
            //    int idx = i * 1. * vals.ysize() / N_STEPS;
            //    borders.push_back(vals[Clamp(idx, 0, vals.ysize() - 1)]);
            //}

            // ROCKS
            MedianGrid(vals, 32, &borders);

            Sort(borders.begin(), borders.end());
            {
                int dst = 0;
                for (int i = 1; i < YSize(borders); ++i) {
                    if (borders[i] != borders[dst]) {
                        borders[++dst] = borders[i];
                    }
                }
                borders.resize(dst);
            }
            for (int i = 0; i < YSize(borders); ++i) {
                AddReason(pRes, docInfos, nReason, borders[i]);
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////
void PrintFeatures(const char *fileName, const vector<vector<bool> > &features,
    const vector<float> &target, const vector<int> &queryId, const vector<int> &groupId,
    int start, int finish)
{
    int featureCount = features.size();
    std::ofstream fout(fileName);
    for (int i = start; i < finish; ++i) {
        fout << queryId[i] << "\t" << target[i] << "\turl\t" << groupId[i];
        for (int f = 0; f < featureCount; ++f) {
            fout << "\t" << (int)features[f][i];
        }
        fout << "\n";
    }
}


void PrintFeaturs(const char *fname,
    const TVector<TDocInfo> docs,
    const vector<float> &target, const vector<vector<bool> > &features,
    int beg, int fin)
{
    std::ofstream fo(fname);
    for (int i = beg; i < fin; ++i) {
        const TDocInfo &d = docs[i];
        fo << d.QueryId << "\t" << target[i] << "\tUrl\t0";
        for (int f = 0; f < YSize(features); ++f) {
            fo << "\t" << (features[f][i] ? 1. : 0);
        }
        fo << "\n";
    }
}

