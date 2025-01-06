#include "stdafx.h"
#include "config.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
TStringParams::TStringParams(const TString &cfg)
{
    TString name;
    TString value;
    bool isName = true;
    for (char c : cfg) {
        if (isName) {
            if (isalpha((ui8)c)) {
                name += c;
            } else {
                isName = false;
                value += c;
            }
        } else {
            if (isalpha((ui8)c)) {
                Params.push_back(TParam(name, atof(value.c_str())));
                name.resize(0);
                value.resize(0);
                name.push_back(c);
                isName = true;
            } else {
                value.push_back(c);
            }
        }
    }
    if (!value.empty()) {
        Params.push_back(TParam(name, atof(value.c_str())));
    }
}


TString TStringParams::GetString()
{
    TString res;
    for (const TParam &param : Params) {
        res += Sprintf("%s%g", param.Name.c_str(), param.Value);
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TOpt::TOpt(const TString &cfg, int argc, char **argv)
{
    THashMap<TString, yint> argCount;
    TString name;
    for (char c : cfg) {
        if (c == ':') {
            ++argCount[name];
        } else {
            name.clear();
            name += c;
            argCount[name];
        }
    }
    for (yint k = 1; k < argc;) {
        Y_VERIFY(*argv[k] != 0);
        if (argv[k][0] == '-') {
            auto it = argCount.find(argv[k] + 1);
            if (it != argCount.end()) {
                TParam param;
                param.Name = it->first;
                ++k;
                for (yint x = 0; x < it->second; ++x) {
                    if (k == argc) {
                        DebugPrintf("Insufficient number of argument of command line parameter %s\n", param.Name.c_str());
                        exit(-1);
                    }
                    param.Args.push_back(argv[k++]);
                }
                Params.push_back(param);
                continue;
            }
        }
        Args.push_back(argv[k++]);
    }
}
