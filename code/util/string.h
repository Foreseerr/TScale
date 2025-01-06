#pragma once


//////////////////////////////////////////////////////////////////////////////////////////////////
// utf utils
extern ui8 Utf8CodeLength[256];

enum EWordCase
{
    WORD_LOWER_CASE,
    WORD_CAPITAL_START,
    WORD_MIXED_CASE,
};

EWordCase GetWordCase(const TString &str);
TString ToLower(const TString &str);
TString UpcaseFirstLetter(const TString &str);


//////////////////////////////////////////////////////////////////////////////////////////////////
// cp1251 encoding
TString Utf2Win(const TString &utf8);
TString Win2Utf(const TString &cp1251);
char Unicode2Win(yint key);

template <class TDst>
void Unicode2Utf(ui32 code, TDst *pBuf)
{
    if (code < 128) {
        pBuf->push_back(code);
    } else {
        if (code < 0x800) {
            pBuf->push_back(0xc0 + (code >> 6));
            pBuf->push_back(0x80 + (code & 0x3f));
        } else if (code < 0x10000) {
            pBuf->push_back(0xe0 + (code >> 12));
            pBuf->push_back(0x80 + ((code >> 6) & 0x3f));
            pBuf->push_back(0x80 + (code & 0x3f));
        } else {
            pBuf->push_back(0xf0 + (code >> 18));
            pBuf->push_back(0x80 + ((code >> 12) & 0x3f));
            pBuf->push_back(0x80 + ((code >> 6) & 0x3f));
            pBuf->push_back(0x80 + (code & 0x3f));
        }
    }
}

