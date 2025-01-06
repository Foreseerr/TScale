#include "stdafx.h"
#include "cfg_file.h"


enum {
    PS_START,
    PS_DST,
    PS_ASSIGN,
    PS_CALL,
    PS_CALL_ARG,
};

static bool IsTokenChar(char c)
{
    return isalnum((ui8)c) || c == '_' || c == '.';
}

void ParseConfig(TConfigFile *pCfg, const TString &sz)
{
    yint state = PS_START;
    yint line = 1;
    TConfigFile::TOp op;
    for (const char *p = sz.c_str(); *p;) {
        if (*p == 10) {
            ++line;
            ++p;
            continue;
        }
        if (isspace((ui8)*p)) {
            ++p;
            continue;
        }
        if (*p == '#') {
            // skip commented line
            while (*p && *p != 10) {
                ++p;
            }
            continue;
        }
        TString token;
        if (IsTokenChar(*p)) {
            while (IsTokenChar(*p)) {
                token += *p++;
            }
        } else if (*p == '"') {
            ++p;
            while (*p && *p != '"') {
                token += *p++;
            }
            p += (*p) != 0;
        } else if (*p == '\'') {
            ++p;
            while (*p && *p != '\'') {
                token += *p++;
            }
            p += (*p) != 0;
        } else {
            token += *p++;
        }

        switch (state) {
        case PS_START:
            op.Dst = token;
            state = PS_DST;
            break;
        case PS_DST:
            if (token == "=") {
                op.Op = CFG_OP_ASSIGNMENT;
                state = PS_ASSIGN;
            } else if (token == "(") {
                op.Op = CFG_OP_CALL;
                state = PS_CALL;
            } else {
                DebugPrintf("Expected = or (, got %s at line %g\n", token.c_str(), line * 1.);
                abort();
            }
            break;
        case PS_ASSIGN:
            op.Args.push_back(token);
            pCfg->OpArr.push_back(op);
            op = TConfigFile::TOp();
            state = PS_START;
            break;
        case PS_CALL:
            if (token == ")") {
                pCfg->OpArr.push_back(op);
                op = TConfigFile::TOp();
                state = PS_START;
            } else {
                op.Args.push_back(token);
                state = PS_CALL_ARG;
            }
            break;
        case PS_CALL_ARG:
            if (token == ",") {
                state = PS_CALL;
            } else if (token == ")") {
                pCfg->OpArr.push_back(op);
                op = TConfigFile::TOp();
                state = PS_START;
            } else {
                DebugPrintf("Expected , or ) got %s at line %g\n", token.c_str(), line * 1.);
                abort();
            }
            break;
        }
    }
    if (state != PS_START) {
        DebugPrintf("unexpected eof at line %g\n", line * 1.);
    }
}


bool IsYes(const TString &str)
{
    TString strLow;
    for (ui8 x : str) {
        strLow.push_back(tolower(x));
    }
    return strLow == "1" || strLow == "true" || strLow == "yes";
}
