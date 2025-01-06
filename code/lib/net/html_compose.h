#pragma once

namespace NNet
{
struct THtmlPage
{
    TVector<TString> Parts;

    THtmlPage(const TString &title, const TString &cssStyles, const TString &bodyProps);
    void AddPart(const TString &x)
    {
        Parts.push_back(x);
    }
    void MakeHtml(TString *pRes);
    THtmlPage &operator+=(const TString &x) { AddPart(x); return *this; }
};

TString DefaultTableCSS(const TString &textAlign);
TString EncodeXML(const TString &src);
}
