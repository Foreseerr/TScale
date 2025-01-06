#include "stdafx.h"
#include "bpe.h"
#include <lib/random/rand_utils.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
void TPackedBPETokenReader::Read(yint offset, yint len, TVector<TBPEToken> *p)
{
    // read
    TVector<ui8> buf;
    buf.resize(len * BytesPerToken);
    File.Seek(offset * BytesPerToken);
    File.Read(buf.data(), len * BytesPerToken);
    // unpack
    ui8 *bufPtr = buf.data();
    p->resize(len);
    for (yint i = 0; i < len; ++i) {
        ui64 res = 0;
        ui8 *resPtr = (ui8 *)&res;
        if (BytesPerToken == 3) {
            *resPtr++ = *bufPtr++;
            *resPtr++ = *bufPtr++;
            *resPtr++ = *bufPtr++;
            if (res == 0xffffff) {
                res = UNDEFINED_TOKEN;
            }
        } else if (BytesPerToken == 2) {
            *resPtr++ = *bufPtr++;
            *resPtr++ = *bufPtr++;
            if (res == 0xffff) {
                res = UNDEFINED_TOKEN;
            }
        }
        (*p)[i] = res;
    }
}


void TPackedBPETokenWriter::Write(const TVector<TBPEToken> &tokens)
{
    yint len = YSize(tokens);
    // pack
    TVector<ui8> buf;
    buf.resize(len * BytesPerToken);
    ui8 *bufPtr = buf.data();
    for (yint i = 0; i < len; ++i) {
        ui64 src = tokens[i];
        if (src == UNDEFINED_TOKEN) {
            src = 0xffffffffffffffull;
        }
        ui8 *srcPtr = (ui8*) &src;
        if (BytesPerToken == 3) {
            *bufPtr++ = *srcPtr++;
            *bufPtr++ = *srcPtr++;
            *bufPtr++ = *srcPtr++;
        } else if (BytesPerToken == 2) {
            *bufPtr++ = *srcPtr++;
            *bufPtr++ = *srcPtr++;
        }
    }
    // write
    File.Write(buf.data(), YSize(buf));
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TWordCount
{
    TString Word;
    yint Count = 0;
};
void CollectFrequentWords(const TVector<TVector<char>> &textArr, TVector<TString> *pRes, yint maxWordCount)
{
    TVector<TWordCount> wcArr;
    THashMap<TString, int> wordCounts;
    for (const TVector<char> &text : textArr) {
        TUtf8WordIterator it(text, 0, YSize(text));
        while (it.NextWord()) {
            const TString &word = it.GetWord();
            if (YSize(word) > 1) {
                wordCounts[it.Word] += 1;
            }
        }
    }
    for (auto it = wordCounts.begin(); it != wordCounts.end(); ++it) {
        TWordCount wc;
        wc.Word = it->first;
        wc.Count = it->second;
        wcArr.push_back(wc);
    }
    Sort(wcArr.begin(), wcArr.end(), [](const TWordCount &a, const TWordCount &b) { return a.Count > b.Count; });
    if (YSize(wcArr) > maxWordCount) {
        wcArr.resize(maxWordCount);
    }
    for (const TWordCount &wc : wcArr) {
        pRes->push_back(wc.Word);
    }
}


void CreateWordsetTokenizer(TTokenizer *pTokenizer, const TVector<TString> &words, TTokenizer::ETokenizer tk)
{
    pTokenizer->MakeByteEncoder(tk);
    for (const TString &w : words) {
        Y_ASSERT(YSize(w) > 1);
        pTokenizer->AddWord(w);
    }
}
