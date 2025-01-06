#pragma once
#include <util/string.h>

typedef ui32 TBPEToken;
const TBPEToken UNDEFINED_TOKEN = 0xffffffff;


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TPackedBPETokenReader : public TThrRefBase
{
    TFileStream File;
    yint BytesPerToken;
public:
    TPackedBPETokenReader(const TString &fname, yint bytesPerToken) : File(IO_READ, fname), BytesPerToken(bytesPerToken)
    {
        Y_VERIFY(File.IsValid() && "file not found");
    }
    void Read(yint offset, yint len, TVector<TBPEToken> *p);
};


struct TPackedBPETokenWriter : public TThrRefBase
{
    TFileStream File;
    yint BytesPerToken;
public:
    TPackedBPETokenWriter(const TString &fname, yint bytesPerToken) : File(IO_WRITE, fname), BytesPerToken(bytesPerToken) {}
    void Write(const TVector<TBPEToken> &tokens);
};


///////////////////////////////////////////////////////////////////////////////////////////////////


struct TUtf8WordIterator
{
    const TVector<char> &Text;
    yint Ptr = 0;
    yint Fin = 0;
    yint Utf8CharCount = 0;
    TString Word;
public:
    TUtf8WordIterator(const TVector<char> &text, yint start, yint fin) : Text(text), Ptr(start), Fin(fin), Utf8CharCount(0)
    {
        while (Ptr < Fin && Utf8CodeLength[(ui8)Text[Ptr]] == 255) {
            ++Ptr;
        }
    }

    bool NextWord()
    {
        Word.resize(0);
        while (Ptr < Fin) {
            ui8 c = Text[Ptr];
            ui8 clen = Utf8CodeLength[c];
            if (clen > 1) {
                // we assume all 128+ characters are letters
                if (Ptr + clen <= Fin) {
                    for (yint k = 0; k < clen; ++k) {
                        Word += Text[Ptr + k];
                    }
                    Utf8CharCount += 1;
                    Ptr += clen;
                } else {
                    // utf8 char does not fit into [start;fin)
                    break;
                }
            } else if (isalpha(c)) {
                Word += c;
                Utf8CharCount += 1;
                Ptr += 1;
            } else {
                if (!Word.empty()) {
                    break;
                }
                Word.resize(1);
                Word[0] = c;
                Utf8CharCount += 1;
                Ptr += 1;
                break;
            }
        }
        return !Word.empty();
    }

    const TString &GetWord() const
    {
        return Word;
    }

    yint GetUtf8CharCount() const
    {
        return Utf8CharCount;
    }
};




///////////////////////////////////////////////////////////////////////////////////////////////////
class TTokenizer
{
public:
    enum ETokenizer
    {
        TK_CHAR,
        TK_WORD,
        TK_WORD_CAPITAL,
        TK_GREEDY,
        TK_GREEDY_CAPITAL,
    };

private:
    ETokenizer TokenizerType = TK_WORD;
    yint TokenCount = 0;
    TVector<int> Letters;
    THashMap<TString, int> Word2Id;
    TVector<TString> Words;
    int DocStartToken = -1;
    int CapitalWordToken = -1;
    int FragmentStartToken = -1;
public:
    SAVELOAD(TokenizerType, TokenCount, Letters, Word2Id, Words, DocStartToken, CapitalWordToken, FragmentStartToken);

private:
    void GenLetterTokens(const TString &word, TVector<TBPEToken> *res) const
    {
        for (yint k = 0, len = YSize(word); k < len;) {
            ui8 c = word[k];
            yint clen = Utf8CodeLength[c];
            Y_VERIFY(k + clen <= len);
            if (clen > 1) {
                auto itWord = Word2Id.find(word.substr(k, clen));
                if (itWord != Word2Id.end() && itWord->second >= 0) {
                    res->push_back(itWord->second);
                    k += clen;
                    continue;
                }
            }
            for (yint t = 0; t < clen; ++t) {
                yint w = Letters[(ui8)word[k + t]];
                Y_VERIFY(w >= 0);
                res->push_back(w);
            }
            k += clen;
        }
    }

    void GenWordTokens(const TString &word, TVector<TBPEToken> *res) const
    {
        auto itWord = Word2Id.find(word);
        if (itWord != Word2Id.end() && itWord->second >= 0) {
            res->push_back(itWord->second);
        } else {
            GenLetterTokens(word, res);
        }
    }

    void GenGreedyTokens(const TString &str, TVector<TBPEToken> *res) const
    {
        yint strLen = YSize(str);
        for (yint start = 0; start < strLen;) {
            ui8 c = (ui8)str[start];
            TBPEToken bestToken = c;
            yint bestLen = 1;
            yint tokenLen = Utf8CodeLength[c];
            if (tokenLen > 1) {
                TString sub = str.substr(start, tokenLen);
                auto itWord = Word2Id.find(sub);
                if (itWord != Word2Id.end() && itWord->second >= 0) {
                    bestToken = itWord->second;
                    bestLen = tokenLen;
                } else {
                    // utf8 symbol not found, add as bytes
                    for (yint t = 0; t < tokenLen; ++t) {
                        res->push_back((ui8)str[start + t]);
                    }
                    start += tokenLen;
                    continue;
                }
            }
            for (yint ptr = start + bestLen; ptr < strLen;) {
                ptr += Utf8CodeLength[(ui8)str[ptr]];
                Y_ASSERT(ptr <= strLen);
                TString sub = str.substr(start, ptr - start);
                auto itWord = Word2Id.find(sub);
                if (itWord != Word2Id.end()) {
                    if (itWord->second >= 0) {
                        bestToken = itWord->second;
                        bestLen = ptr - start;
                    }
                } else {
                    break;
                }
            }
            res->push_back(bestToken);
            start += bestLen;
        }
    }

    yint AddSystemToken()
    {
        Words.push_back("");
        return TokenCount++;
    }

public:
    bool IsEmpty() const
    {
        return TokenCount == 0;
    }

    yint GetVocabSize() const
    {
        return TokenCount;
    }

    void MakeUsedLettersEncoder(const TVector<char> &text)
    {
        *this = TTokenizer();
        TokenizerType = TK_CHAR;
        Letters.resize(256, -1);
        for (yint i = 0; i < YSize(text); ++i) {
            ui8 c = text[i];
            if (Letters[c] < 0) {
                Letters[c] = TokenCount++;
                TString sz;
                sz.push_back(c);
                Words.push_back(sz);
            }
        }
        FragmentStartToken = AddSystemToken();
    }

    void MakeByteEncoder(ETokenizer tk)
    {
        *this = TTokenizer();
        TokenizerType = tk;
        Letters.resize(256, -1);
        for (yint c = 0; c < 256; ++c) {
            Letters[c] = TokenCount++;
            TString sz;
            sz.push_back(c);
            Words.push_back(sz);
        }
        FragmentStartToken = AddSystemToken();
        DocStartToken = AddSystemToken();
        if (tk == TK_WORD_CAPITAL || tk == TK_GREEDY_CAPITAL) {
            CapitalWordToken = AddSystemToken();
        }
    }

    void AddWord(const TString &str)
    {
        Y_VERIFY(!str.empty());
        yint strLen = YSize(str);
        yint partLen = 0;
        for(;;) {
            partLen += Utf8CodeLength[(ui8)str[partLen]];
            if (partLen >= strLen) {
                Word2Id[str] = TokenCount;
                Words.push_back(str);
                ++TokenCount;
                return;
            }
            if (partLen > 1) {
                TString sub = str.substr(0, partLen);
                auto it = Word2Id.find(sub);
                if (it == Word2Id.end()) {
                    Word2Id[sub] = -1;
                }
            }
        }
    }

    const TString &GetWord(yint id) const
    {
        return Words[id];
    }

    bool HasDocStartToken() const
    {
        return DocStartToken >= 0;
    }

    yint GetDocStartToken() const
    {
        Y_ASSERT(HasDocStartToken());
        return DocStartToken;
    }

    yint GetCapitalWordToken() const
    {
        return CapitalWordToken;
    }

    yint GetFragmentStartToken() const
    {
        return FragmentStartToken;
    }

    // text encoding, return text[] length in utf8 chars
    yint GenWords(const TVector<char> &text, yint start, yint fin, TVector<TBPEToken> *res) const
    {
        TUtf8WordIterator it(text, start, fin);
        while (it.NextWord()) {
            const TString &word = it.Word;
            if (TokenizerType == TK_CHAR) {
                GenLetterTokens(word, res);

            } else if (TokenizerType == TK_WORD) {
                auto itWord = Word2Id.find(word);
                if (itWord != Word2Id.end()) {
                    res->push_back(itWord->second);
                } else {
                    GenLetterTokens(word, res);
                }

            } else if (TokenizerType == TK_WORD_CAPITAL) {
                Y_ASSERT(CapitalWordToken >= 0);
                EWordCase wc = GetWordCase(word);
                if (wc == WORD_LOWER_CASE) {
                    GenWordTokens(word, res);
                } else if (wc == WORD_CAPITAL_START) {
                    res->push_back(CapitalWordToken);
                    GenWordTokens(ToLower(word), res);
                } else {
                    GenLetterTokens(word, res);
                }

            } else if (TokenizerType == TK_GREEDY) {
                GenGreedyTokens(word, res);

            } else if (TokenizerType == TK_GREEDY_CAPITAL) {
                Y_ASSERT(CapitalWordToken >= 0);
                EWordCase wc = GetWordCase(word);
                if (wc == WORD_LOWER_CASE) {
                    GenGreedyTokens(word, res);
                } else if (wc == WORD_CAPITAL_START) {
                    res->push_back(CapitalWordToken);
                    GenGreedyTokens(ToLower(word), res);
                } else {
                    GenGreedyTokens(word, res);
                }

            } else {
                Y_VERIFY(0 && "unknown tokenizer");
            }
        }
        return it.GetUtf8CharCount();
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
void CollectFrequentWords(const TVector<TVector<char>> &textArr, TVector<TString> *pRes, yint maxWordCount);
void CreateWordsetTokenizer(TTokenizer *pTokenizer, const TVector<TString> &words, TTokenizer::ETokenizer tk);
