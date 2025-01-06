#pragma once

namespace NNet
{
struct THttpRequest
{
    TString Req;
    THashMap<TString, TString> Params;
    TVector<char> Data;

    bool HasParam(const char *pszParam) const
    {
        THashMap<TString, TString>::const_iterator i = Params.find(pszParam);
        return (i != Params.end());
    }

    TString GetParam(const char *pszParam) const
    {
        THashMap<TString, TString>::const_iterator i = Params.find(pszParam);
        if (i == Params.end())
            return "";
        return i->second;
    }

    yint GetIntParam(const char *pszParam) const
    {
        TString sz = GetParam(pszParam);
        return atoll(sz.c_str());
    }

    bool GetBoolParam(const char *pszParam) const
    {
        TString sz = GetParam(pszParam);
        return sz == "yes" || sz == "true" || atoi(sz.c_str()) == 1;
    }

    TString GetUrl() const
    {
        TString res = "/" + Req;
        bool first = true;
        for (auto it = Params.begin(); it != Params.end(); ++it) {
            if (first) {
                res += "?" + it->first + "=" + it->second;
            } else {
                res += "&" + it->first + "=" + it->second;
            }
            first = false;
        }
        return res;
    }
};

bool ParseRequest(THttpRequest *pRes, const char *pszReq);

TString EncodeCGI(const TString &arg);
TString DecodeCGI(const TString &x);
}
