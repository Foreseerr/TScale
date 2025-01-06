#pragma once

enum {
    CFG_OP_ASSIGNMENT,
    CFG_OP_CALL,
};

struct TConfigFile
{
    struct TOp
    {
        yint Op = CFG_OP_ASSIGNMENT;
        TString Dst;
        TVector<TString> Args;
    };
    TVector<TOp> OpArr;
};

void ParseConfig(TConfigFile *pCfg, const TString &sz);

bool IsYes(const TString &str);
