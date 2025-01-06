#pragma once

namespace NNet
{
struct THttpPacket
{
    TVector<TString> HeaderLines;
    TVector<char> Data;

    TString GetRequest();
    yint GetHttpStatus();
    bool IsKeepalive();
    yint GetContentLength();
};

enum EHttpPacketParseResult
{
    HTTP_PKT_INCOMPLETE,
    HTTP_PKT_BAD,
    HTTP_PKT_OK,
};

EHttpPacketParseResult ParseHttpPacket(const TVector<char> &buf, yint dataSize, THttpPacket *pPkt, yint *pHttpPktSize);
}
