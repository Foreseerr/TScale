#pragma once

namespace NNet
{
class THttpConnection : public TThrRefBase
{
    TString PureHost;
    sockaddr_in Addr;
    SOCKET Sock = INVALID_SOCKET;
    TVector<char> RecvBuf;
    yint RecvOffset = 0;

private:
    bool Connect();
    void Close();

public:
    THttpConnection(const TString &addr, yint defaultPort);
    bool FetchJson(const TString &request, const vector<char> &reqData, vector<char> *reply);
};
}
