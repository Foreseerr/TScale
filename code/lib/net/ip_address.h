#pragma once

struct sockaddr_in;

namespace NNet
{
timeval MakeTimeval(float timeoutSec);
bool ParseInetName(sockaddr_in *pName, TString *pszPureHost, const TString &szAddress, int nDefaultPort);
void ReplacePort(TString *pszPureHost, int newPort);
TString GetHostName();
TString GetAddressString(const sockaddr_in &addr);
}
