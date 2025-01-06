#include "stdafx.h"
#include <gpt/data/text_saveload.h>
#include <lib/random/rand_utils.h>


//const bool USE_CAPITAL_TOKEN = false;
const bool USE_CAPITAL_TOKEN = true;

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TWordStats
{
    THashMap<TString, yint> WordCount;
};

static void CollectWords(TWordStats *p, const TVector<char> &text)
{
    TString word;
    for (yint i = 0; i < YSize(text); ++i) {
        ui8 c = text[i];
        if (c >= 0x80 || isalpha(c)) {
            // do something smart here
            word += c;
        } else {
            if (!word.empty()) {
                if (USE_CAPITAL_TOKEN) {
                    if (GetWordCase(word) != WORD_MIXED_CASE) {
                        p->WordCount[ToLower(word)] += 1;
                    }
                } else {
                    p->WordCount[word] += 1;
                }
            }
            word = "";
        }
    }
    // skip last word
}


static void CollectWordsFromDocset(TWordStats *pStats, const TString &fileName)
{
    TVector<TVector<char>> docSet;
    LoadDocumentSetFromBin(&docSet, fileName);
    for (const TVector<char> &text : docSet) {
        CollectWords(pStats, text);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TStringCount
{
    TString Word;
    yint Count = 0;
};


struct TWordset
{
    TVector<TString> Words;
    THashMap<TString, bool> TakenWords;

public:
    void AddWord(const TString &str)
    {
        if (TakenWords.find(str) == TakenWords.end()) {
            Words.push_back(str);
            TakenWords[str];
        }
    }

    void AddLetters(const TString &str)
    {
        for (yint k = 0; k < YSize(str);) {
            yint clen = Utf8CodeLength[(ui8)str[k]];
            if (clen > 1) {
                AddWord(str.substr(k, clen));
            }
            k += clen;
        }
    }

    yint GetWordCount() const
    {
        return YSize(Words);
    }
};

const TString RussianLetters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";


static void CreateGreedyIterative(const TWordStats &ws, yint maxWordCount)
{
    TWordset bestFreqPiece;
    bestFreqPiece.AddLetters(RussianLetters);

    for (yint iter = 0; iter < 20; ++iter) {
        DebugPrintf(" iteration %g\n", iter * 1.);

        THashMap<TString, bool> hasPiece; // false for head chars of longer tokens
        for (const TString &str : bestFreqPiece.Words) {
            yint strLen = YSize(str);
            for (yint partLen = 0; partLen < strLen;) {
                partLen += Utf8CodeLength[(ui8)str[partLen]];
                hasPiece[str] |= partLen == strLen;
            }
        }

        THashMap<TString, yint> pieceCounts;
        for (auto it = ws.WordCount.begin(), itEnd = ws.WordCount.end(); it != itEnd; ++it) {
            const TString &str = it->first;
            yint weight = it->second;
            yint strLen = YSize(str);
            for (yint start = 0; start < strLen;) {
                ui8 c = (ui8)str[start];
                yint bestLen = 1;
                yint tokenLen = Utf8CodeLength[c];
                if (tokenLen > 1) {
                    TString sub = str.substr(start, tokenLen);
                    pieceCounts[sub] += weight; // alwyas add first utf character of the piece
                    auto itPiece = hasPiece.find(sub);
                    if (itPiece != hasPiece.end() && itPiece->second) {
                        bestLen = tokenLen;
                    } else {
                        // leading utf8 symbol not found
                        start += tokenLen;
                        continue;
                    }
                }
                TString bestExisting;
                TString bestNext;
                for (yint ptr = start + bestLen; ptr < strLen;) {
                    ptr += Utf8CodeLength[(ui8)str[ptr]];
                    Y_ASSERT(ptr <= strLen);
                    TString sub = str.substr(start, ptr - start);
                    auto itPiece = hasPiece.find(sub);
                    if (itPiece != hasPiece.end()) {
                        if (itPiece->second) {
                            bestExisting = sub;
                            bestLen = ptr - start;
                        }
                    } else {
                        bestNext = sub;
                        break;
                    }
                }
                if (!bestExisting.empty()) {
                    pieceCounts[bestExisting] += weight;
                }
                if (!bestNext.empty()) {
                    pieceCounts[bestNext] += weight;
                }
                start += bestLen;
            }

        }

        TVector<TStringCount> wcArr;
        for (auto it = pieceCounts.begin(); it != pieceCounts.end(); ++it) {
            TStringCount wc;
            wc.Word = it->first;
            wc.Count = it->second;
            if (YSize(wc.Word) > 1) {
                wcArr.push_back(wc);
            }
        }
        Sort(wcArr.begin(), wcArr.end(), [](const TStringCount &a, const TStringCount &b) { return a.Count > b.Count; });

        TWordset freqPiece;
        freqPiece.AddLetters(RussianLetters);
        for (const TStringCount &wc : wcArr) {
            if (freqPiece.GetWordCount() >= maxWordCount) {
                break;
            }
            freqPiece.AddWord(wc.Word);
        }
        bestFreqPiece = freqPiece;

        TOFStream f(Sprintf("d:/words_%g.txt", iter * 1.).c_str());
        for (const TString &word : freqPiece.Words) {
            f << word.c_str() << "\n";
        }
    }
    TTokenizer tokenizer;
    TTokenizer::ETokenizer tk = USE_CAPITAL_TOKEN ? TTokenizer::TK_GREEDY_CAPITAL : TTokenizer::TK_GREEDY;
    CreateWordsetTokenizer(&tokenizer, bestFreqPiece.Words, tk);
    Serialize(IO_WRITE, "d:/greedy_tokenizer.bin", tokenizer);
}


static void CreateFreqWordTokenizer(const TWordStats &ws, yint maxWordCount)
{
    TWordset freqWords;
    freqWords.AddLetters(RussianLetters);

    TVector<TStringCount> wcArr;
    for (auto it = ws.WordCount.begin(); it != ws.WordCount.end(); ++it) {
        TStringCount wc;
        wc.Word = it->first;
        wc.Count = it->second;
        if (YSize(wc.Word) > 1) {
            wcArr.push_back(wc);
        }
    }
    Sort(wcArr.begin(), wcArr.end(), [](const TStringCount &a, const TStringCount &b) { return a.Count > b.Count; });
    for (const TStringCount &wc : wcArr) {
        if (freqWords.GetWordCount() >= maxWordCount) {
            break;
        }
        freqWords.AddWord(wc.Word);
    }
    TTokenizer tokenizer;
    TTokenizer::ETokenizer tk = USE_CAPITAL_TOKEN ? TTokenizer::TK_WORD_CAPITAL : TTokenizer::TK_WORD;
    CreateWordsetTokenizer(&tokenizer, freqWords.Words, tk);
    Serialize(IO_WRITE, "d:/freqword_tokenizer.bin", tokenizer);
}


static void LoadRandomDocsets(TWordStats *pCounts, TMersenne<ui32> &rng, const TString &dir, yint fileCount, yint totalBinFileCount)
{
    for (yint k = 0; k < fileCount; ++k) {
        CollectWordsFromDocset(pCounts, Sprintf("%s/%d.bin", dir.c_str(), rng.Uniform(totalBinFileCount)));
    }
}


int main()
{
    //const yint TAKEN_COUNT = 5000;
    const yint TAKEN_COUNT = 50000;
    //const yint TAKEN_COUNT = 200000;

#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    DebugPrintf("collect words\n");
    TMersenne<ui32> rng(1313);
    TWordStats counts;
    LoadRandomDocsets(&counts, rng, "D:/text/cultura_y", 50, 7059);
    LoadRandomDocsets(&counts, rng, "D:/text/librusec", 20, 440);
    LoadRandomDocsets(&counts, rng, "D:/text/open_web_text", 40, 802);
    //LoadRandomDocsets(&counts, rng, "D:/text/cultura_y", 2, 7059);
    //LoadRandomDocsets(&counts, rng, "D:/text/librusec", 1, 440);
    //LoadRandomDocsets(&counts, rng, "D:/text/open_web_text", 2, 802);

    DebugPrintf("create frequent word tokenizer\n");
    CreateFreqWordTokenizer(counts, TAKEN_COUNT);

    DebugPrintf("create greedy tokenizer\n");
    CreateGreedyIterative(counts, TAKEN_COUNT);

    return 0;
}
