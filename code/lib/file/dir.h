#pragma once

struct TFindFileResult
{
    TString Name;
    bool IsDir = false;

    TFindFileResult() {}
    TFindFileResult(const TString &n, bool isDir) : Name(n), IsDir(isDir) {}
};

void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res);
void MakeDirectory(const TString &dir);
void ChDir(const TString &dir);
bool DoesFileExist(const TString &fileName);
void EraseFile(const TString &fileName);
void RenameFile(const TString &fileName, const TString &newName);
