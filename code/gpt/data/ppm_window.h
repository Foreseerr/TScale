#pragma once
#include "bpe.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
// small window version, using linked lists of all occurances of 1-2-4 char hashes
class TWindowPPMIndex
{
    enum {
        // max subsequence size
        MAX_LEN = 32,
        WINDOW = 4096,
        //WINDOW = 16384,

        HASH_TABLE_SIZE_LN = 20, // trade table size and false match candidates
        HASH_TABLE_SIZE = 1 << HASH_TABLE_SIZE_LN,
    };

    struct THash
    {
        TVector<ui16> Table;
        TVector<ui16> Next;
        SAVELOAD(Table, Next);

        THash()
        {
            Clear();
        }
        void Clear()
        {
            ClearPodArray(&Table, HASH_TABLE_SIZE);
            ClearPodArray(&Next, WINDOW);
        }
        yint GetEntry(yint h, yint pos)
        {
            yint base16 = pos & ~(0xffffll);
            yint res = base16 + Table[h];
            if (res >= pos) {
                res -= 0x10000;
            }
            return res;
        }
        void SetEntry(yint h, yint pos)
        {
            Next[pos & (WINDOW - 1)] = pos - Table[h] - 1;
            Table[h] = pos;
        }
        yint GetNext(yint pos)
        {
            return pos - Next[pos & (WINDOW - 1)] - 1;
        }
    };

    THash Hash1;
    THash Hash2;
    THash Hash4;
    yint PrevIndexPos = 0;
    yint PrevBestPos = 0;
    yint PrevBestLen = 0;

private:
    static yint CalcHash(ui64 data)
    {
        ui64 h = data * 0xc3a5c85c97cb3127ULL;
        return h >> (64 - HASH_TABLE_SIZE_LN);
    }

public:
    SAVELOAD(Hash1, Hash2, Hash4);

    void Clear()
    {
        Hash1.Clear();
        Hash2.Clear();
        Hash4.Clear();
        PrevIndexPos = 0;
        PrevBestPos = 0;
        PrevBestLen = 0;
    }

    void IndexPos(const TVector<TBPEToken> &text, yint indexPos, yint *pBestLen, yint *pBestPos)
    {
        yint h1 = *(ui16 *)&text[indexPos];
        yint h2 = -1;
        if (indexPos > 0) {
            h2 = CalcHash(*(ui32 *)&text[indexPos - 1]);
        }
        yint h4 = -1;
        if (indexPos > 2) {
            h4 = CalcHash(*(ui64 *)&text[indexPos - 3]);
        }
        yint bestLen = 0;
        yint bestPos = 0;
        yint minPtr = Max<yint>(0, indexPos - WINDOW + MAX_LEN);

        if (PrevBestLen > 0 && indexPos == PrevIndexPos + 1) {
            if (text[indexPos] == text[PrevBestPos + 1]) {
                bestLen = PrevBestLen + 1;
                bestPos = PrevBestPos + 1;
            }
        }

        if (bestLen == 0) {
            yint ptr = Hash1.GetEntry(h1, indexPos);
            while (ptr >= minPtr) {
                if (text[indexPos] == text[ptr]) {
                    bestLen = 1;
                    bestPos = ptr;
                    break;
                }
                ptr = Hash1.GetNext(ptr);
            }
            if (bestLen == 1) {
                ptr = Hash2.GetEntry(h2, indexPos);
                yint minPtr2 = Max<yint>(1, minPtr);
                ui32 refText2 = *(ui32 *)(text.data() + indexPos - 1);
                while (ptr >= minPtr2) {
                    ui32 chkText2 = *(ui32 *)(text.data() + ptr - 1);
                    if (refText2 == chkText2) {
                        bestLen = 2;
                        bestPos = ptr;
                        break;
                    }
                    ptr = Hash2.GetNext(ptr);
                }

                if (bestLen == 2) {
                    yint minPtr3 = Max<yint>(2, minPtr);
                    ui16 refText3 = text[indexPos - 2];
                    while (ptr >= minPtr3) {
                        ui32 chkText2 = *(ui32 *)(text.data() + ptr - 1);
                        ui16 chkText3 = text[ptr - 2];
                        if (refText2 == chkText2 && refText3 == chkText3) {
                            bestLen = 3;
                            bestPos = ptr;
                            break;
                        }
                        ptr = Hash2.GetNext(ptr);
                    }

                    if (bestLen == 3) {
                        ptr = Hash4.GetEntry(h4, indexPos);
                        yint minPtr4 = Max<yint>(3, minPtr);
                        ui64 refText4 = *(ui64 *)(text.data() + indexPos - 3);
                        while (ptr >= minPtr4) {
                            ui64 chkText4 = *(ui64 *)(text.data() + ptr - 3);
                            if (refText4 == chkText4) {
                                bestLen = 4;
                                bestPos = ptr;
                                break;
                            }
                            ptr = Hash4.GetNext(ptr);
                        }

                        if (bestLen == 4) {
                            yint maxLen = MAX_LEN;
                            ui64 refText4 = *(ui64 *)(text.data() + indexPos - 3);
                            for (; ptr >= minPtr4;) {
                                ui64 chkText4 = *(ui64 *)(text.data() + ptr - 3);
                                if (chkText4 == refText4) {
                                    if (ptr < maxLen) {
                                        maxLen = ptr + 1;
                                    }
                                    yint len = maxLen;
                                    for (yint offset = 4; offset < maxLen; ++offset) {
                                        if (text[indexPos - offset] != text[ptr - offset]) {
                                            len = offset;
                                            break;
                                        }
                                    }
                                    if (len > bestLen) {
                                        bestLen = len;
                                        bestPos = ptr;
                                    }
                                }
                                ptr = Hash4.GetNext(ptr);
                            }
                        }
                    }
                }
            }
        }

        //// brute force reference
        //if (bestLen < MAX_LEN) { // longest matches sometime diverge in prev best branch (reference code can not find longer then max_len, prev best match can exceed max_len)
        //    yint trueBestLen = 0;
        //    yint trueBestPos = 0;
        //    for (yint ptr = indexPos - 1; ptr >= minPtr; --ptr) {
        //        yint maxLen = Min<yint>(ptr + 1, MAX_LEN);
        //        yint len = maxLen;
        //        for (yint offset = 0; offset < maxLen; ++offset) {
        //            if (text[indexPos - offset] != text[ptr - offset]) {
        //                len = offset;
        //                break;
        //            }
        //        }
        //        if (len > trueBestLen) {
        //            trueBestLen = len;
        //            trueBestPos = ptr;
        //        }
        //    }
        //    Y_VERIFY(trueBestLen == bestLen);
        //    Y_VERIFY(trueBestPos == bestPos);
        //}
        *pBestLen = bestLen;
        if (bestLen > 0) {
            *pBestPos = bestPos; // matching interval are (indexPos - bestLen; indexPos] and (bestPos - bestLen; bestPos]
        }
        PrevBestLen = bestLen;
        PrevBestPos = bestPos;
        PrevIndexPos = indexPos;
        Hash1.SetEntry(h1, indexPos);
        if (h2 >= 0) {
            Hash2.SetEntry(h2, indexPos);
        }
        if (h4 >= 0) {
            Hash4.SetEntry(h4, indexPos);
        }
    }
};


void ComputeWindowPPM(const TVector<TBPEToken> &text, TVector<TBPEToken> *pResPPM, yint docStartToken);
