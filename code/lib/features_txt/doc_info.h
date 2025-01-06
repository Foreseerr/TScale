#pragma once


struct TDocInfo
{
    int QueryId, GroupId;
    TString szUrl;
    float fRelev;
    TVector<double> Factors;
};

void LoadData(TVector<TDocInfo> *pRes, const char *pszFileName);
