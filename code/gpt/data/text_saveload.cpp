#include "stdafx.h"
#include "text_saveload.h"
#include <lib/random/rand_utils.h>
#include <lib/file/dir.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
void GenerateArithmetic()
{
    TMersenne<ui32> rng(1313);
    TOFStream f("arith.txt");
    for (yint k = 0; k < 100000000; ++k) {
        yint maxVal = 10;
        int rr = rng.Uniform(4);
        if (rr == 0) {
            maxVal = 100000;
        } else if (rr == 1) {
            maxVal = 10000;
        } else if (rr == 2) {
            maxVal = 1000;
        } else if (rr == 3) {
            maxVal = 100;
        }
        int op = rng.Uniform(4);
        if (op > 1) {
            maxVal = Min<int>(maxVal, 10000);
        }
        yint n1 = rng.Uniform(maxVal);
        yint n2 = rng.Uniform(maxVal);
        if (op == 0) {
            f << n1 << " + " << n2 << " = " << n1 + n2 << "\n";
        } else if (op == 1) {
            f << n1 << " - " << n2 << " = " << n1 - n2 << "\n";
        } else {
            f << n1 << " * " << n2 << " = " << n1 * n2 << "\n";
        }
    }
}


void GenerateArithmetic97()
{
    // Grokking, binary ops train https://arxiv.org/pdf/2201.02177v1.pdf
    TVector<TString> samples;
    const yint MOD = 97;
    for (yint x = 0; x < MOD; ++x) {
        for (yint y = 0; y < MOD; ++y) {
            //yint val = (x + y) % MOD;
            yint val = (x * x + x * y + y * y + x) % MOD;
            samples.push_back(Sprintf("%c * %c = %c", x + 128, y + 128, val + 128));
        }
    }
    TMersenne<ui32> rng(1313);
    Shuffle(samples.begin(), samples.end(), rng);
    TOFStream f("arith97.txt");
    f << "\n";
    for (const TString &str : samples) {
        f << str.c_str() << "\n";
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
static void CollectFilesRecursive(const TString &prefix, TVector<TString> *pRes)
{
    TVector<TFindFileResult> dir;
    FindAllFiles(prefix, &dir);
    for (const TFindFileResult &ff : dir) {
        if (ff.IsDir) {
            CollectFilesRecursive(prefix + "/" + ff.Name, pRes);
        }
    }
    for (const TFindFileResult &ff : dir) {
        if (!ff.IsDir) {
            pRes->push_back(prefix + "/" + ff.Name);
        }
    }
}


void LoadDocument(TVector<char> *pRes, const TString &fileName)
{
    Y_VERIFY(ReadWholeFile(fileName, pRes));
}


void LoadDocumentSetFromFiles(TVector<TVector<char>> *pRes, const TString &dir)
{
    TVector<TString> files;
    CollectFilesRecursive(dir, &files);
    //printf("Load %g files\n", YSize(files) * 1.);
    for (const TString &ff : files) {
        //printf("Load %s\n", ff.c_str());
        TVector<char> &text = *pRes->insert(pRes->end());
        Y_VERIFY(ReadWholeFile(ff, &text));
    }
}


void LoadDocumentSetFromBin(TVector<TVector<char>> *pRes, const TString &fileName)
{
    TFileStream f(IO_READ, fileName);
    pRes->reserve(100000);
    while (f.IsValid()) {
        ui32 sz = 0;
        if (f.Read(&sz, sizeof(sz)) != sizeof(sz)) {
            break;
        }
        TVector<char> &dst = *pRes->insert(pRes->end());
        dst.resize(sz);
        yint chk = f.Read(dst.data(), sz);
        if (chk != sz) {
            DebugPrintf("file %s, expected to read %g bytes, get %g bytes \n", fileName.c_str(), sz * 1., chk * 1.);
            break;
        }
    }
}


template <int WIDTH>
void CopyInts(TVector<TBPEToken> *p, const ui8 *src, yint len)
{
    p->yresize(len);
    const ui8 *srcPtr = src;
    TBPEToken *dstPtr = p->data();
    for (yint t = 0; t < len; ++t) {
        ui64 x = 0;
        ui8 *xPtr = (ui8 *)&x;
        for (yint x = 0; x < WIDTH; ++x) {
            *xPtr++ = *srcPtr++;
        }
        *dstPtr++ = x;
    }
}

void LoadTokenized(const TString &fileName, yint tokenWidth, yint headerSize, TVector<TBPEToken> *p)
{
    TFileStream f1(IO_READ, fileName);
    Y_VERIFY(f1.IsValid());
    if (headerSize > 0) {
        TVector<ui8> header;
        header.resize(headerSize);
        f1.Read(header.data(), headerSize);
    }
    yint len = (f1.GetLength() - headerSize) / tokenWidth;
    TVector<ui8> buf;
    buf.yresize(len * tokenWidth);
    f1.Read(buf.data(), YSize(buf));
    switch (tokenWidth) {
    case 1: CopyInts<1>(p, buf.data(), len); break;
    case 2: CopyInts<2>(p, buf.data(), len); break;
    case 3: CopyInts<3>(p, buf.data(), len); break;
    case 4: CopyInts<4>(p, buf.data(), len); break;
    default:
        Y_VERIFY(0 && "unexpected token width");
    }
}


void SaveDocumentSetToBin(const TVector<TVector<char>> &textArr, const TString &fileName)
{
    TFileStream f(IO_WRITE, fileName);
    for (const TVector<char> &text : textArr) {
        ui32 sz = YSize(text);
        f.Write(&sz, sizeof(sz));
        f.Write(text.data(), sz);
    }
}


//void Repack()
//{
//    TString prefix = "text/cultura_y/";
//    TVector<TFindFileResult> dir;
//    FindAllFiles(prefix, &dir);
//    yint resId = 0;
//    TVector<TVector<char>> resDocs;
//    yint totalSize = 0;
//    for (const TFindFileResult &ff : dir) {
//        TVector<TVector<char>> docs;
//        LoadDocumentSetFromBin(&docs, prefix + ff.Name);
//        for (yint k = 0; k < YSize(docs); ++k) {
//            resDocs.push_back(docs[k]);
//            totalSize += YSize(docs[k]);
//            if (totalSize > 150 * 1000000) {
//                SaveDocumentSetToBin(resDocs, Sprintf("%d.bin", (int)resId));
//                ++resId;
//                resDocs.resize(0);
//                totalSize = 0;
//            }
//        }
//    }
//    SaveDocumentSetToBin(resDocs, Sprintf("%d.bin", (int)resId));
//}
