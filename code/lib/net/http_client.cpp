#include "stdafx.h"
#include "http_client.h"
#include "http_header.h"
#include "ip_address.h"


namespace NNet
{
bool THttpConnection::Connect()
{
    if (Sock != INVALID_SOCKET) {
        return true;
    }
    Sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    Y_VERIFY(Sock != INVALID_SOCKET);
    if (connect(Sock, (sockaddr *)&Addr, sizeof(Addr)) == SOCKET_ERROR) {
        Close();
        return false;
    }
    RecvOffset = 0;
    return true;
}


void THttpConnection::Close()
{
    if (Sock != INVALID_SOCKET) {
        closesocket(Sock);
        Sock = INVALID_SOCKET;
    }
}


THttpConnection::THttpConnection(const TString &addr, yint defaultPort)
{
    RecvBuf.resize(1 << 16);
    if (!ParseInetName(&Addr, &PureHost, addr.c_str(), defaultPort)) {
        DebugPrintf("invalid host %s\n", addr.c_str()); fflush(0);
        Y_VERIFY(0 && "invalid host");
    }
}


bool THttpConnection::FetchJson(const TString &request, const vector<char> &reqData, vector<char> *reply)
{
    if (!Connect()) {
        return false;
    }

    TString fullReq = reqData.empty() ? string("GET ") : string("POST ");
    fullReq += request + " HTTP/1.1\r\n";
    fullReq += TString("Host: ") + PureHost + "\r\n";
    fullReq += "Connection: keep-alive\r\n";
    fullReq += "Accept: application/json, */*; q=0.01\r\n";
    if (!reqData.empty()) {
        fullReq += "Content-Type: application/json, */*; q=0.01\r\n";
        fullReq += Sprintf("Content-Length: %llu\r\n", YSize(reqData));
    }
    fullReq += "\r\n";
    if (!reqData.empty()) {
        int headerSize = fullReq.size();
        fullReq.resize(headerSize + reqData.size());
        memcpy(&fullReq[headerSize], reqData.data(), reqData.size());
    }

    for (yint offset = 0, len = YSize(fullReq); offset < len;) {
        int sz = Min<yint>(1 << 24, len - offset);
        int rv = send(Sock, fullReq.data() + offset, sz, 0);
        if (rv < 0) {
            Close();
            return false;
        }
        offset += rv;
    }

    THttpPacket pkt;
    yint pktSize;
    for (;;) {
        if (YSize(RecvBuf) - RecvOffset < 10000) {
            RecvBuf.resize(YSize(RecvBuf) * 2);
        }
        int sz = Min<yint>(1 << 24, YSize(RecvBuf) - RecvOffset);
        int rv = recv(Sock, RecvBuf.data() + RecvOffset, sz, 0);
        if (rv == 0) {
            // peer closed connection
            Close();
            if (ParseHttpPacket(RecvBuf, RecvOffset, &pkt, &pktSize) == HTTP_PKT_OK) {
                break;
            }
            return false;
        } else if (rv == SOCKET_ERROR) {
            Close();
            return false;
        } else {
            RecvOffset += rv;
            EHttpPacketParseResult pr = ParseHttpPacket(RecvBuf, RecvOffset, &pkt, &pktSize);
            if (pr == HTTP_PKT_OK) {
                if (pkt.IsKeepalive()) {
                    RecvBuf.erase(RecvBuf.begin(), RecvBuf.begin() + pktSize);
                    RecvOffset = 0;
                } else {
                    Close();
                }
                break;
            } else if (pr == HTTP_PKT_BAD) {
                Close();
                return false;
            }
        }
    }

    // HTTP/1.0 200 OK?
    if (pkt.GetHttpStatus() == 200) {
        reply->swap(pkt.Data);
        return true;
    }
    return false;
}

}
