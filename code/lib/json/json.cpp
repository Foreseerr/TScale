#include "stdafx.h"
#include "json.h"
#include <util/string.h>


namespace NJson
{
///////////////////////////////////////////////////////////////////////////////////////////////////
static TString EncodeJsonString(const TString &x)
{
    // encode
    TString res = "\"";
    for (ui8 c : x) {
        if (c == '"' || c =='\\' || c == '/') {
            res.push_back('\\');
            res.push_back(c);
        } else if (c < 32) {
            if (c == '\b') {
                res.push_back('\\');
                res.push_back('b');
            } else if (c == '\f') {
                res.push_back('\\');
                res.push_back('f');
            } else if (c == '\n') {
                res.push_back('\\');
                res.push_back('n');
            } else if (c == '\r') {
                res.push_back('\\');
                res.push_back('r');
            } else if (c == '\t') {
                res.push_back('\\');
                res.push_back('t');
            } else {
                const char hex[] = "0123456789ABCDEF";
                res.push_back('\\');
                res.push_back('u');
                yint n = c;
                res.push_back(hex[(n >> 12) & 15]);
                res.push_back(hex[(n >> 8) & 15]);
                res.push_back(hex[(n >> 4) & 15]);
                res.push_back(hex[(n) & 15]);
            }
        } else {
            res.push_back(c);
        }
    }
    res.push_back('"');
    return res;
}

static TString DecodeJsonString(const TString &x)
{
    TString res;
    for (yint pos = 0, len = YSize(x); pos < len; ++pos) {
        char c = x[pos];
        if (c == '\\') {
            if (++pos == len) {
                break;
            }
            c = x[pos];
            if (c == '\\' || c == '"') {
                res += c;
            } else if (c == 'b') {
                res += '\b';
            } else if (c == 'f') {
                res += '\f';
            } else if (c == 'n') {
                res += '\n';
            } else if (c == 'r') {
                res += '\r';
            } else if (c == 't') {
                res += '\t';
            } else if (c == 'u') {
                yint code = 0;
                for (yint k = 0; k < 4; ++k) {
                    if (++pos == len) {
                        break;
                    }
                    c = x[pos];
                    if (c >= '0' && c <= '9') {
                        code = code * 16 + c - '0';
                    } else {
                        code = code * 16 + toupper((ui8)c) - 'A';
                    }
                }
                Unicode2Utf(code, &res);
            }
        } else {
            res.push_back(c);
        }
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// return value type
static yint ParseString(TIntrusivePtr<TJson> json, const TVector<char> &buf, yint *pBufPos, int *pStringPos)
{
    yint &pos = *pBufPos;
    enum {
        STRING_START,
        QUOTED_STRING,
        STRING,
    };
    yint state = STRING_START;
    TString res;
    for (yint len = YSize(buf); pos < len; ++pos) {
        ui8 c = buf[pos];
        if (state == STRING_START) {
            if (!isspace(c)) {
                if (c != '"') {
                    res.push_back(c);
                    state = STRING;
                } else {
                    state = QUOTED_STRING;
                }
            }
        } else if (state == QUOTED_STRING) {
            if (c == '"') {
                *pStringPos = json->AddString(DecodeJsonString(res));
                return TElement::STRING;
            } else {
                res.push_back(c);
            }
        } else if (state == STRING) {
            if (isalnum(c) || c == '.' || c == '+' || c == '-') {
                res.push_back(c);
            } else {
                --pos;
                *pStringPos = json->AddString(res);
                return TElement::VALUE;
            }
        }
    }
    return TElement::NONE;
}


struct TParseStateStack
{
    TVector<TElement> &Elements;
    TVector<yint> StateStack;
    TVector<yint> ElementStack;

    TParseStateStack(TVector<TElement> &elements) : Elements(elements) {}
    void PushState(yint pushState)
    {
        StateStack.push_back(pushState);
        ElementStack.push_back(YSize(Elements));
        Elements.push_back(TElement());
    }
    void PushState(yint newState, yint pushState)
    {
        StateStack.back() = newState;
        PushState(pushState);
    }
    bool PopState()
    {
        StateStack.pop_back();
        ElementStack.pop_back();
        return StateStack.empty();
    }
    void NewState(yint newState)
    {
        StateStack.back() = newState;
    }
    void NewElement(yint newState)
    {
        yint elemId = YSize(Elements);
        GetElement().Next = elemId;
        StateStack.back() = newState;
        ElementStack.back() = elemId;
        Elements.push_back(TElement());
    }
    yint GetState() const { return StateStack.back(); }
    TElement &GetElement() { return Elements[ElementStack.back()]; }
};


TIntrusivePtr<TJson> ParseJson(const TVector<char> &buf)
{
    TIntrusivePtr<TJson> json = new TJson;

    enum {
        OBJECT_FIELD,
        OBJECT_COLON,
        OBJECT_VALUE,
        OBJECT_NEXT_ELEMENT,
        ARRAY_VALUE,
        ARRAY_NEXT_ELEMENT,
    };
    TParseStateStack pss(json->Elements);
    pss.PushState(OBJECT_VALUE);
    for (yint pos = 0, len = YSize(buf); pos < len; ++pos) {
        char c = buf[pos];
        if (isspace(c)) {
            continue;
        }
        yint state = pss.GetState();
        TElement &element = pss.GetElement();
        if (state == OBJECT_FIELD) {
            yint tt = ParseString(json, buf, &pos, &element.NamePos);
            if (tt != TElement::STRING) {
                return 0;
            }
            pss.NewState(OBJECT_COLON);

        } else if (state == OBJECT_COLON) {
            if (c == ':') {
                pss.NewState(OBJECT_VALUE);
            } else {
                return 0;
            }

        } else if (state == OBJECT_VALUE) {
            if (c == '[') {
                element.Type = TElement::ARRAY;
                pss.PushState(OBJECT_NEXT_ELEMENT, ARRAY_VALUE);
            } else if (c == '{') {
                element.Type = TElement::OBJECT;
                pss.PushState(OBJECT_NEXT_ELEMENT, OBJECT_FIELD);
            } else {
                yint tt = ParseString(json, buf, &pos, &element.ValuePos);
                if (tt == TElement::VALUE || tt == TElement::STRING) {
                    element.Type = tt;
                } else {
                    return 0;
                }
                pss.NewState(OBJECT_NEXT_ELEMENT);
            }

        } else if (state == OBJECT_NEXT_ELEMENT) {
            if (c == ',') {
                pss.NewElement(OBJECT_FIELD);
            } else if (c == '}') {
                if (pss.PopState()) {
                    return json;
                }
            } else {
                return 0;
            }

        } else if (state == ARRAY_VALUE) {
            if (c == '[') {
                element.Type = TElement::ARRAY;
                pss.PushState(ARRAY_NEXT_ELEMENT, ARRAY_VALUE);
            } else if (c == '{') {
                element.Type = TElement::OBJECT;
                pss.PushState(ARRAY_NEXT_ELEMENT, OBJECT_FIELD);
            } else {
                yint tt = ParseString(json, buf, &pos, &element.ValuePos);
                if (tt == TElement::VALUE || tt == TElement::STRING) {
                    element.Type = tt;
                } else {
                    return 0;
                }
                pss.NewState(ARRAY_NEXT_ELEMENT);
            }

        } else if (state == ARRAY_NEXT_ELEMENT) {
            if (c == ',') {
                pss.NewElement(ARRAY_VALUE);
            } else if (c == ']') {
                if (pss.PopState()) {
                    return json;
                }
            } else {
                return 0;
            }
        }
    }
    return json;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TJsonRender
{
    TVector<char> Buf;
    bool Whitespace = false;

    TJsonRender(bool white) : Whitespace(white) {}
    void Write(char c)
    {
        Buf.push_back(c);
    }
    void WriteWhitespace(char c)
    {
        if (Whitespace) {
            Buf.push_back(c);
        }
    }
};

static void WriteSpace(yint printOffset, TJsonRender *dst)
{
    for (yint k = 0; k < printOffset; ++k) {
        dst->WriteWhitespace(' ');
        dst->WriteWhitespace(' ');
    }
}

static void WriteValue(TIntrusivePtr<TJson> json, yint stringPos, TJsonRender *dst)
{
    for (yint k = stringPos; k < YSize(json->Strings); ++k) {
        char c = json->Strings[k];
        if (c == 0) {
            break;
        }
        dst->Write(c);
    }
}

static void WriteString(TIntrusivePtr<TJson> json, yint stringPos, TJsonRender *dst)
{
    TString x = EncodeJsonString(json->GetString(stringPos));
    for (char c : x) {
        dst->Write(c);
    }
}

static void WriteJson(TIntrusivePtr<TJson> json, yint printOffset, bool printName, yint *pPos, yint fin, TJsonRender *dst)
{
    yint &pos = *pPos;
    while (pos < fin) {
        const TElement &elem = json->Elements[pos++];
        if (elem.Type == TElement::VALUE || elem.Type == TElement::STRING) {
            WriteSpace(printOffset, dst);
            if (printName) {
                WriteString(json, elem.NamePos, dst);
                dst->Write(':');
            }
            if (elem.Type == TElement::VALUE) {
                WriteValue(json, elem.ValuePos, dst);
            } else {
                WriteString(json, elem.ValuePos, dst);
            }

        } else if (elem.Type == TElement::OBJECT || elem.Type == TElement::ARRAY) {
            yint newFin = elem.Next >= 0 ? elem.Next : fin;
            WriteSpace(printOffset, dst);
            if (printName) {
                WriteString(json, elem.NamePos, dst);
                dst->WriteWhitespace(' ');
                dst->Write(':');
                dst->WriteWhitespace(' ');
            }
            if (elem.Type == TElement::OBJECT) {
                dst->Write('{');
                dst->WriteWhitespace('\n');
                WriteJson(json, printOffset + 2, true, pPos, newFin, dst);
                WriteSpace(printOffset, dst);
                dst->Write('}');
            } else if (elem.Type == TElement::ARRAY) {
                dst->Write('[');
                dst->WriteWhitespace('\n');
                WriteJson(json, printOffset + 2, false, pPos, newFin, dst);
                WriteSpace(printOffset, dst);
                dst->Write(']');
            }
        }
        if (elem.Next >= 0) {
            Y_ASSERT(pos == elem.Next);
            dst->Write(',');
        }
        dst->WriteWhitespace('\n');
    }
}

static void Render(TIntrusivePtr<TJson> json, TJsonRender *dst)
{
    yint pos = 0;
    WriteJson(json, 0, false, &pos, YSize(json->Elements), dst);

}

TVector<char> Render(TIntrusivePtr<TJson> json)
{
    TJsonRender buf(false);
    Render(json, &buf);
    return buf.Buf;
}

TString RenderString(TIntrusivePtr<TJson> json)
{
    TJsonRender buf(true);
    Render(json, &buf);
    buf.Buf.push_back(0);
    return buf.Buf.data();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
static void PrintArray(const TJsonIterator &arg)
{
    TJsonIterator data = arg.Expand();
    for (TJsonIterator data = arg.Expand(); data.IsValid(); data.Next()) {
        DebugPrintf("%s, ", data.GetValue());
    }
    DebugPrintf("\n");
}

static bool FindField(TJsonIterator &it, const char *sz)
{
    for (; it.IsValid(); it.Next()) {
        if (strcmp(it.GetName(), sz) == 0) {
            return true;
        }
    }
    return false;
}

void Test()
{
    const char *szTest = "[ { \"Text\" : [12,45,67], \"Target\" : [-1,28,-1]}, { \"Text\" : [34,78,21], \"Target\" : [-1,-1,8]}]";
    TVector<char> buf;
    buf.resize(strlen(szTest));
    strcpy(buf.data(), szTest);
    TIntrusivePtr<TJson> json = ParseJson(buf);
    Y_VERIFY(json.Get() != 0);
    {
        TJsonIterator root(json);
        TJsonIterator it = root.Expand();
        TJsonIterator frag = it.Expand();
        if (FindField(frag, "Text")) {
            DebugPrintf("fragment 0 text: ");
            PrintArray(frag);
        }
        it.Next();
        frag = it.Expand();
        if (FindField(frag, "Target")) {
            DebugPrintf("fragment 1 target: ");
            PrintArray(frag);
        }
    }
    TVector<char> res = Render(json);
    res.push_back(0);
    DebugPrintf("json:\n%s\n", res.data());
}
}
