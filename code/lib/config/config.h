#pragma once

struct TStringParams
{
    struct TParam
    {
        TString Name;
        double Value = 0;

        TParam() {}
        TParam(const TString &name, double val) : Name(name), Value(val) {}
    };
    TVector<TParam> Params;
    TVector<TString> Args;

public:
    TStringParams() {}
    TStringParams(const TString &cfg);
    TString GetString();
    void AddParam(const TString &name, double value)
    {
        Params.push_back(TParam(name, value));
    }
    double GetParam(const TString &name, double defaultValue) const
    {
        for (const TParam &param : Params) {
            if (param.Name == name) {
                return param.Value;
            }
        }
        return defaultValue;
    }
};


struct TOpt
{
    struct TParam
    {
        TString Name;
        TVector<TString> Args;
    };
    TVector<TParam> Params;
    TVector<TString> Args;

public:
    TOpt(const TString &cfg, int argc, char **argv);
    bool HasParam(const TString &name)
    {
        for (auto &param : Params) {
            if (param.Name == name) {
                return true;
            }
        }
        return false;
    }
};
