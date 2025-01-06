#pragma once
#include "data.h"
#include <lib/net/tcp_net.h>

TIntrusivePtr<IDataSource> ConnectDataServer(TIntrusivePtr<NNet::ITcpSendRecv> net, const TString &addr);
TIntrusivePtr<IDataSource> ConnectHttpDataServer(const TString &addr);
void RunDataServer(TIntrusivePtr<NNet::ITcpSendRecv> net, TIntrusivePtr<IDataSource> data);
