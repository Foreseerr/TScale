#include "stdafx.h"
#include "doc_info.h"
#include <fstream>

#define MAX_LINE_SIZE 120000

void LoadData(TVector<TDocInfo> *pRes, const char *pszFileName)
{
    TVector<TDocInfo> &docInfos = *pRes;
    //docInfos.clear();

    std::ifstream fIn(pszFileName);
    char szBuf[MAX_LINE_SIZE + 10];
    int nDims = -1;
    TVector<const char*> words;
    TVector<double> plane;
    while (!fIn.eof()) {
        fIn.getline(szBuf, MAX_LINE_SIZE);
        Split(szBuf, &words);
        if (nDims == -1)
            nDims = words.size() - 4;
        if (words.size() != nDims + 4)
            break;

        TDocInfo dInfo;
        dInfo.QueryId = atoi(words[0]);
        dInfo.fRelev = atof(words[1]);
        dInfo.szUrl = words[2];
        dInfo.GroupId = atoi(words[3]);
        dInfo.Factors.resize(nDims);
        for (int i = 0; i < nDims; ++i)
            dInfo.Factors[i] = atof(words[i + 4]);
        docInfos.push_back(dInfo);
    }
}
