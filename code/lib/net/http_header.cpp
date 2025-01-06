#include "stdafx.h"
#include "http_header.h"


namespace NNet
{
////////////////////////////////////////////////////////////////////////////////
TString THttpPacket::GetRequest()
{
    if (HeaderLines.empty()) {
        return "";
    }
    const char *pszReq = HeaderLines[0].data();
    if (strncmp(pszReq, "GET", 3) == 0) {
        pszReq += 3;
    } else if (strncmp(pszReq, "POST", 4) == 0) {
        pszReq += 4;
    } else {
        return "";
    }
    while (*pszReq && isspace((unsigned char)*pszReq)) {
        ++pszReq;
    }
    const char *resStart = pszReq;
    while (*pszReq && !isspace((unsigned char)*pszReq)) {
        ++pszReq;
    }
    return TString(resStart, pszReq);
}


yint THttpPacket::GetHttpStatus()
{
    if (HeaderLines.empty()) {
        return -1;
    }
    const char *hdr = HeaderLines[0].c_str();
    if (strncmp(hdr, "HTTP", 4) != 0) {
        return -1;
    }
    for (; *hdr; ++hdr) {
        if (*hdr == ' ') {
            return atoi(++hdr);
        }
    }
    return -1;
}


bool THttpPacket::IsKeepalive()
{
    bool isKeepAlive = false;
    if (!HeaderLines.empty() && (StartsWith(HeaderLines[0], "HTTP/1.1") ||EndsWith(HeaderLines[0], "HTTP/1.1"))) {
        isKeepAlive = true; // keep alive is default for http 1.1
    }
    for (const TString &q : HeaderLines) {
        if (q == "Connection: close") { // other connection options mean keep alive
            isKeepAlive = false;
        }
    }
    return isKeepAlive;
}

yint THttpPacket::GetContentLength()
{
    const char *prefix = "Content-Length: ";
    const yint prefixLen = strlen(prefix);
    for (const TString &q : HeaderLines) {
        if (strncmp(q.data(), prefix, prefixLen) == 0) {
            return atoll(q.data() + prefixLen);
        }
    }
    return 0;
}


EHttpPacketParseResult ParseHttpPacket(const TVector<char> &buf, yint dataSize, THttpPacket *pPkt, yint *pHttpPktSize)
{
    enum {
        GENERIC,
        HAS_R,
    };
    yint state = GENERIC;
    const char *sz = buf.data();
    const char *qLine = sz;
    *pPkt = THttpPacket();
    for (const char *szEnd = sz + dataSize; sz < szEnd; ++sz) {
        if (state == GENERIC) {
            if (*sz == '\r') {
                pPkt->HeaderLines.push_back(TString(qLine, sz));
                state = HAS_R;
            }
        } else if (state == HAS_R) {
            if (*sz == '\n') {
                if (pPkt->HeaderLines.back().empty()) {
                    // complete query
                    //DebugPrintf("request %s\n", query.data());
                    ++sz;
                    yint contentLength = pPkt->GetContentLength();
                    if (sz + contentLength > szEnd) {
                        return HTTP_PKT_INCOMPLETE;
                    }
                    pPkt->Data.insert(pPkt->Data.end(), sz, sz + contentLength);
                    *pHttpPktSize = (sz + contentLength) - buf.data();
                    return HTTP_PKT_OK;
                }
                state = GENERIC;
                qLine = sz + 1;
            } else {
                return HTTP_PKT_BAD;
            }
        }
    }
    return HTTP_PKT_INCOMPLETE;
}
}
