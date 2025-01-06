#pragma once
#include "bpe.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
void GenerateArithmetic();
void GenerateArithmetic97();

///////////////////////////////////////////////////////////////////////////////////////////////////
void LoadDocument(TVector<char> *pRes, const TString &fileName);
void LoadDocumentSetFromFiles(TVector<TVector<char>> *pRes, const TString &dir);
void LoadDocumentSetFromBin(TVector<TVector<char>> *pRes, const TString &fileName);
void LoadTokenized(const TString &fileName, yint tokenWidth, yint headerSize, TVector<TBPEToken> *p);
void SaveDocumentSetToBin(const TVector<TVector<char>> &textArr, const TString &fileName);


