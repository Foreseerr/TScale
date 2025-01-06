#include "stdafx.h"
#include "html_compose.h"

namespace NNet
{
THtmlPage::THtmlPage(const TString &title, const TString &cssStyles, const TString &bodyProps)
{
    Parts.resize(1000);
    Parts.resize(1);
    Parts[0] = 
        "<html>\n"
        "<head>\n"
        "    <meta charset = \"UTF-8\">\n"
        //"    <meta name=\"viewport\" content=\"initial-scale=1,shrink-to-fit=no\">\n"
        "    <title>" + title + "</title>\n"
        "    <style>\n"
        + cssStyles +
        "    </style>\n"
        "</head>\n"
        //"<body onload = \"ReloadData('0')\">\n";
        "<body " + bodyProps + ">\n";
}


void THtmlPage::MakeHtml(TString *pRes)
{
    TString fin = "</body></html>";
    yint totalSize = YSize(fin);
    for (const TString &sz : Parts) {
        totalSize += YSize(sz);
    }
    pRes->resize(totalSize);
    yint dst = 0;
    for (const TString &sz : Parts) {
        yint len = YSize(sz);
        memcpy(&(*pRes)[dst], sz.data(), len);
        dst += len;
    }
    memcpy(&(*pRes)[dst], fin.data(), YSize(fin));
}


TString DefaultTableCSS(const TString &textAlign)
{
    return
        "table {border: 1px solid;border-collapse: collapse;}\n"
        "th {text-align:left;padding-left:0.5rem;padding-right:0.5rem;background-color:lightgrey;}\n"
        "td {padding:0.5rem;}\n"
        "tr,td{text-align:" + textAlign + ";vertical-align:middle;}\n";
}


TString EncodeXML(const TString &src)
{
    TString res;
    for (char c : src) {
        if (c == '\"') {
            res += "&quot;";
        } else if (c == '&') {
            res += "&amp;";
        } else if (c == '\'') {
            res += "&apos;";
        } else if (c == '<') {
            res += "&lt;";
        } else if (c == '>') {
            res += "&gt;";
        } else {
            res += c;
        }
    }
    return res;
}

}
