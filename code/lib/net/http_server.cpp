#include "stdafx.h"
#include "http_server.h"
#include "net_util.h"
#include "http_header.h"
#include "http_request.h"
#include "ip_address.h"
#include <errno.h>
//#include <ws2ipdef.h>


namespace NNet
{
#if (!defined(_win_) && !defined(_darwin_))
    int SEND_FLAGS = MSG_NOSIGNAL;
#else
    int SEND_FLAGS = 0;
#endif

////////////////////////////////////////////////////////////////////////////////

static bool StartListen(SOCKET *psListen, int *port)
{
    SOCKET &sListen = *psListen;
    if (sListen != INVALID_SOCKET) {
        closesocket(sListen);
    }
    //
    sListen = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sListen == INVALID_SOCKET) {
        return false;
    }
    {
        //int flag = 0;
        //setsockopt(sListen, IPPROTO_IPV6, IPV6_V6ONLY, (const char*)&flag, sizeof(flag));

        int flag = 1;
        setsockopt(sListen, SOL_SOCKET, SO_REUSEADDR, (const char*)&flag, sizeof(flag));
    }

    sockaddr_in name;
    Zero(name);
    name.sin_family = AF_INET;
    name.sin_addr.s_addr = INADDR_ANY;
    name.sin_port = htons((ui16)*port);

    if (bind(sListen, (sockaddr*)&name, sizeof(name)) != 0) {
        Y_ASSERT(0);
        closesocket(sListen);
        sListen = INVALID_SOCKET;
        return false;
    }
    if (listen(sListen, SOMAXCONN) != 0) {
        Y_ASSERT(0);
        closesocket(sListen);
        sListen = INVALID_SOCKET;
        return false;
    }
    if (*port == 0) {
        // figure out assigned port
        sockaddr_in resAddr;
        socklen_t len = sizeof(resAddr);
        if (getsockname(sListen, (sockaddr*)&resAddr, &len)) {
            Y_ASSERT(0);
            closesocket(sListen);
            sListen = INVALID_SOCKET;
            return false;
        }
        *port = ntohs(resAddr.sin_port);
    }
    MakeNonBlocking(sListen);
    return true;
}


////////////////////////////////////////////////////////////////////////////////
THttpServer::THttpServer(int port)
    : Listen(INVALID_SOCKET), ListenPort(port)
{
    if (!StartListen(&Listen, &ListenPort)) {
        fprintf(stderr, "StartListen() failed: %s\n", strerror(errno)); fflush(stderr);
        abort();
    }
}


THttpServer::~THttpServer()
{
    if (Listen != INVALID_SOCKET) {
        closesocket(Listen);
    }
}


void THttpServer::Poll(TTcpPoller *pl)
{
    yint dst = 0;
    for (yint i = 0; i < YSize(ConnArr); ++i) {
        TIntrusivePtr<TConnection> conn = ConnArr[i];
        if (conn->IsValid() && !conn->IsComplete()) {
            ConnArr[dst++] = conn;
            if (conn->SendBuffer.empty()) {
                pl->AddSocket(conn->s, POLLRDNORM);
            } else {
                pl->AddSocket(conn->s, POLLRDNORM | POLLWRNORM);
            }
        }
    }
    ConnArr.resize(dst);
    pl->AddSocket(Listen, POLLRDNORM);
}


void THttpServer::OnPoll(TTcpPoller *pl, TVector<TRequest> *pReqArr)
{
    yint dst = 0;
    for (yint i = 0; i < YSize(ConnArr); ++i) {
        TIntrusivePtr<TConnection> conn = ConnArr[i];
        yint events = pl->CheckSocket(conn->s);
        if (events & ~(POLLRDNORM | POLLWRNORM)) {
            continue;
        }
        if (events & POLLWRNORM) {
            conn->SendBufferedData();
        }
        if (events & POLLRDNORM) {
            conn->RecvQueries(pReqArr);
        }
        ConnArr[dst++] = conn;
    }
    ConnArr.resize(dst);

    yint events = pl->CheckSocket(Listen);
    if ((events) & ~(POLLRDNORM | POLLWRNORM)) {
        fprintf(stderr, "Nontrivial accept events 0x%llx\n", events); fflush(stderr);
        abort();
    } else if (events & POLLRDNORM) {
        SOCKET s = accept(Listen, (sockaddr *)nullptr, nullptr);
        if (s == INVALID_SOCKET) {
            int err = errno;
            // somehow errno 0 can happen on windows
            if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                fprintf(stderr, "accept() failed for signaled socket, errno %d\n", err); fflush(stderr);
                abort();
            }
        } else {
            SetNoTcpDelay(s);
            MakeNonBlocking(s);
            ConnArr.push_back(new TConnection(s));
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
void THttpServer::TConnection::SendBufferedData()
{
    int sz = Min<yint>(1ll << 24, YSize(SendBuffer) - SendOffset);
    if (sz == 0) {
        //DebugPrintf("called SendBufferedData() without data\n");
        return;
    }
    int rv = send(s, SendBuffer.data() + SendOffset, sz, SEND_FLAGS);
    if (rv == SOCKET_ERROR) {
        yint err = errno;
        if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
            //DebugPrintf("send(), unexpected errno %g\n", err * 1.);
            Close();
        }
    } else {
        SendOffset += rv;
        if (SendOffset == YSize(SendBuffer)) {
            SendBuffer.clear();
            SendOffset = 0;
        }
    }
}


void THttpServer::TConnection::RecvQueries(TVector<THttpServer::TRequest> *pReqArr)
{
    const yint MAX_QUERY_SIZE = 1000000;
    if (RecvOffset == YSize(RecvBuffer)) {
        if (RecvOffset > MAX_QUERY_SIZE) {
            // query is too long
            Close();
            return;
        }
        if (RecvOffset == 0) {
            RecvBuffer.resize(16384);
        } else {
            RecvBuffer.resize(YSize(RecvBuffer) * 2);
        }
    }
    int rv = recv(s, RecvBuffer.data() + RecvOffset, YSize(RecvBuffer) - RecvOffset, 0);
    if (rv == SOCKET_ERROR) {
        yint err = errno;
        if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
            //DebugPrintf("unexpected recv() error: %s\n", strerror(errno));
            Close();
            return;
        }
    } else if (rv == 0) {
        //DebugPrintf("send recv result 0, connection closed? socket %g, we expect no more data from this connection\n", s * 1.);
        Keepalive = false;
        LastReqId = RecvReqId;
    } else {
        Y_VERIFY(rv > 0);
        RecvOffset += rv;
        for (;;) {
            // try to read request
            THttpPacket httpQuery;
            yint httpQuerySize = 0;
            EHttpPacketParseResult ok = ParseHttpPacket(RecvBuffer, RecvOffset, &httpQuery, &httpQuerySize);
            if (ok == HTTP_PKT_OK) {
                Keepalive = httpQuery.IsKeepalive();
                TRequest req;
                req.Conn = this;
                req.ReqId = RecvReqId;
                TString szRequest = httpQuery.GetRequest();
                if (szRequest.empty() || !ParseRequest(&req.Req, szRequest.c_str())) {
                    //DebugPrintf("bad request %s from socket %g\n", szRequest.c_str(), s * 1.);
                    Close();
                    return;
                }
                httpQuery.Data.swap(req.Req.Data);
                pReqArr->push_back(req);
                memmove(RecvBuffer.begin(), RecvBuffer.begin() + httpQuerySize, RecvOffset - httpQuerySize);
                RecvOffset -= httpQuerySize;
                ++RecvReqId;
                if (!Keepalive) {
                    LastReqId = RecvReqId;
                }
            } else if (ok == HTTP_PKT_BAD) {
                //DebugPrintf("malformed http header received\n");
                Close();
                return;
            } else {
                break;
            }
        }
    }
}


void THttpServer::TConnection::SendReply(yint reqId, TVector<char> *pData)
{
    for (;;) {
        Y_VERIFY(!pData->empty());
        if (SendReqId == reqId) {
            if (SendBuffer.empty()) {
                int rv = send(s, pData->data(), YSize(*pData), SEND_FLAGS);
                if (rv == SOCKET_ERROR) {
                    yint err = errno;
                    if (err != 0 && err != EWOULDBLOCK && err != EAGAIN) {
                        //DebugPrintf("SendReply(), send(), unexpected errno %g\n", err * 1.);
                        Close();
                        return;
                    }
                }
                if (rv != YSize(*pData)) {
                    //DebugPrintf("failed to send response right away, socket %g, size %g, rv %g\n", s * 1., YSize(*pData) * 1., rv * 1.);
                    pData->swap(SendBuffer);
                    SendOffset = (rv > 0) ? rv : 0;
                }
            } else {
                SendBuffer.insert(SendBuffer.end(), pData->begin(), pData->end());
            }
            ++SendReqId;
            auto it = OutOfOrderReplies.find(SendReqId);
            if (it == OutOfOrderReplies.end()) {
                break;
            }
            //DebugPrintf("add out of order reply, socket %g, req id %g\n", s * 1., SendReqId * 1.);
            reqId = SendReqId;
            pData->swap(it->second);
            OutOfOrderReplies.erase(it);
        } else {
            //DebugPrintf("queue out of order response, socket %g\n", s * 1.);
            pData->swap(OutOfOrderReplies[reqId]);
            break;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
struct THttpReplayWriter
{
    TVector<char> Buf;

    void Write(yint sz, const void *data)
    {
        if (sz > 0) {
            yint ptr = YSize(Buf);
            Buf.resize(ptr + sz);
            memcpy(Buf.data() + ptr, data, sz);
        }
    }
    void Write(const TString &str)
    {
        Write(YSize(str), str.data());
    }
    void Write(const TVector<char> &vec)
    {
        Write(YSize(vec), vec.data());
    }
    void Write(const char *str)
    {
        Write(strlen(str), str);
    }
};


////////////////////////////////////////////////////////////////////////////////
static TString FormHeader(bool keepalive, const char *type, yint contentLength)
{
    TString reply;
    reply += "HTTP/1.1 200 OK\r\n";
    if (keepalive) {
        reply += "Connection: keep-alive\r\n";
    } else {
        reply += "Connection: close\r\n";
    }
    reply += "Content-Type: ";
    reply += type;
    reply += "\r\n";

    reply += "Content-Length: ";
    reply += Sprintf("%llu", contentLength);
    reply += "\r\n";

    reply += "Cache-control: private, no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Expires: Thu, 01 Jan 1970 00:00:01 GMT\r\n"
        "\r\n";

    return reply;
}


void THttpServer::TRequest::Reply(const TString &reply, const TVector<char> &data, const char *content)
{
    THttpReplayWriter rep;
    rep.Write(FormHeader(Conn->IsKeepalive(), content, YSize(reply) + YSize(data)));
    rep.Write(reply);
    rep.Write(data);
    Conn->SendReply(ReqId, &rep.Buf);
    Conn = 0;
}

void THttpServer::TRequest::ReplyNotFound()
{
    char reply[] = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n";
    THttpReplayWriter rep;
    rep.Write(reply);
    Conn->SendReply(ReqId, &rep.Buf);
    Conn = 0;
}

void THttpServer::TRequest::ReplyXML(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/xml");
}

void THttpServer::TRequest::ReplyHTML(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/html");
}

void THttpServer::TRequest::ReplyPlainText(const TString &reply)
{
    Reply(reply, TVector<char>(), "text/plain");
}

void THttpServer::TRequest::ReplyJson(const TVector<char> &data)
{
    Reply("", data, "application/json; charset=UTF-8");
}

void THttpServer::TRequest::ReplyBin(const TVector<char> &data)
{
    Reply("", data, "application/octet-stream");
}

void THttpServer::TRequest::ReplyBMP(const TVector<char> &data)
{
    Reply("", data, "image/x-MS-bmp");
}

}
