#include "stdafx.h"
#include "fed_lib.h"

TGuid FedToken(0xbadf00d, 0x31337, 0x9ece30, 0x31415926);


bool IsValidUsername(const TString &x)
{
    if (x.empty()) {
        return false;
    }
    for (ui8 c : x) {
        if (!isalnum(c) && c != ' ' && c != '_') {
            return false;
        }
    }
    return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
