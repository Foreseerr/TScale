#pragma once

struct TDocInfo;

void ExtractBoolsFromDocInfo(const vector<TDocInfo> &docInfos, vector<vector<bool> > *pRes);

void PrintFeatures(const char *fileName, const vector<vector<bool> > &features,
    const vector<float> &target, const vector<int> &queryId, const vector<int> &groupId,
    int start, int finish);

void PrintFeaturs(const char *fname,
    const TVector<TDocInfo> docs,
    const vector<float> &target, const vector<vector<bool> > &features,
    int beg, int fin);
