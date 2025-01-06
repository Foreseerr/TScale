#include "stdafx.h"
#include "dir.h"

#ifdef _MSC_VER

void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((prefix + "/*.*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    if (fd.cFileName[0] != '.') {
        res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
    }
    while (FindNextFileA(h, &fd)) {
        if (fd.cFileName[0] != '.') {
            res->push_back(TFindFileResult(fd.cFileName, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)));
        }
    }
    FindClose(h);
}

void MakeDirectory(const TString &dir)
{
    CreateDirectoryA(dir.c_str(), 0);
}

void ChDir(const TString &dir)
{
    SetCurrentDirectoryA(dir.c_str());
}

bool DoesFileExist(const TString &fileName)
{
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(fileName.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE)
        return false;
    FindClose(h);
    return true;
}

void EraseFile(const TString &fileName)
{
    DeleteFileA(fileName.c_str());
}

void RenameFile(const TString &fileName, const TString &newName)
{
    Y_VERIFY(MoveFileA(fileName.c_str(), newName.c_str()));
}

#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

void FindAllFiles(const TString &prefix, TVector<TFindFileResult> *res)
{
    res->resize(0);
    DIR *dir = opendir(prefix.c_str());
    if (dir == NULL) {
        Y_ASSERT(0); // directory does not exist?
        return;
    }
    for (;;) {
        struct dirent *dp = readdir(dir);
        if (dp == 0) {
            break;
        }
        if (dp->d_name[0] == '.') {
            continue;
        }
        struct stat fprop;
        TString fname(dp->d_name);
        int rv = stat((prefix + "/" + fname).c_str(), &fprop);
        if (S_ISDIR(fprop.st_mode)) {
            res->push_back(TFindFileResult(fname, true));
        } else {
            res->push_back(TFindFileResult(fname, false));
        }
    }
    closedir(dir);
}

void MakeDirectory(const TString &dir)
{
    mkdir(dir.c_str(), 0777);
}

void ChDir(const TString &dir)
{
    Y_VERIFY(chdir(dir.c_str()) == 0);
}

bool DoesFileExist(const TString &fileName)
{
    struct stat buffer;
    return (stat(fileName.c_str(), &buffer) == 0);
}

void EraseFile(const TString &fileName)
{
    unlink(fileName.c_str());
}

void RenameFile(const TString &fileName, const TString &newName)
{
    Y_VERIFY(rename(fileName.c_str(), newName.c_str()) == 0);
}

#endif
