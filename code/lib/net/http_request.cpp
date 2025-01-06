#include "stdafx.h"
#include "http_request.h"
#include <string>

namespace NNet
{
bool ParseRequest(THttpRequest *pRes, const char *pszReq)
{
    if (pszReq[0] != '/')
        return false;
    ++pszReq;
    while (*pszReq && *pszReq != '?' && *pszReq != '&')
        pRes->Req += *pszReq++;
    while (*pszReq) {
        ASSERT(*pszReq == '&' || *pszReq == '?');
        ++pszReq;
        string szParam, szVal;
        while (*pszReq && *pszReq != '=' && *pszReq != '&')
            szParam += *pszReq++;
        if (*pszReq == '=') {
            ++pszReq;
            while (*pszReq && *pszReq != '&')
                szVal += *pszReq++;
        }
        pRes->Params[szParam] = szVal;
    }
    return true;
}


TString EncodeCGI(const TString &arg)
{
	//TString res;
	//TUtf16String str = UTF8ToWide(arg);
	//for (size_t i = 0; i < str.size(); ++i) {
	//    unsigned int x = str[i];
	//    if (x < 128) {
	//        res += (char)x;
	//    } else {
	//        res += Sprintf("&#%d;", (int)x);
	//    }
	//}
	TString res = arg;
	TString res2;
	for (size_t i = 0; i < res.size(); ++i) {
		unsigned char c = res[i];
		if (isalnum(c)) {
			res2 += c;
		} else if (c == ' ') {
			res2 += '+';
		} else {
			const char hex[] = "0123456789ABCDEF";
			res2 += '%';
			res2 += hex[(c >> 4) & 15];
			res2 += hex[c & 15];
		}
	}
	return res2;
}


TString DecodeCGI(const TString &x)
{
	TString res;
	for (yint i = 0; i < YSize(x); ++i) {
		if (x[i] == '+') {
			res += " ";
		} else if (x[i] == '%' && i < YSize(x) - 2) {
			char ll[3] = { x[i + 1], x[i + 2], 0 };
			res += (unsigned char)std::stoi(ll, 0, 16);
			i += 2;
		} else {
			res += x[i];
		}
	}
	TString res2;
	for (yint i = 0; i < YSize(res);) {
		if (res[i] == '&' && i < YSize(res) - 1 && res[i + 1] == '#') {
			i += 2;
			TString let;
			while (i < YSize(res) && res[i] != ';') {
				let += res[i++];
			}
			++i;
			wchar_t wc = std::stoi(let.c_str());
			char buf[100];
			int utf8Len = wctomb(buf, wc);
			for (yint k = 0; k < utf8Len; ++k) {
				res2 += buf[k];
			}
		} else {
			res2 += res[i++];
		}
	}
	return res2;
}
}
