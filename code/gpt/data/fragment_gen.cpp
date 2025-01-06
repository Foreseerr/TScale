#include "stdafx.h"
#include "fragment_gen.h"
#include <lib/file/dir.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
TLMatchSearch::TLMatchSearch(const TString &lmIndexDir, TBPEToken docStartToken) : DocStartToken(docStartToken)
{
    ChunkArr.reserve(10000);
    DebugPrintf("Loading lm match index\n"); fflush(0);
    for (yint k = 0;; ++k) {
        TString fname = Sprintf("%s/chunk.%d", lmIndexDir.c_str(), (int)k);
        if (DoesFileExist(fname)) {
            TLMatchChunkIndex &chunk = *ChunkArr.insert(ChunkArr.end());
            Serialize(IO_READ, fname, chunk);
            printf("."); fflush(0);
        } else {
            break;
        }
    }
    printf("\n"); fflush(0);
}


void TLMatchSearch::Search(const TBPEToken *prefix, yint *bestLen, TBPEToken *pNextToken) const
{
    for (const TLMatchChunkIndex &chunk : ChunkArr) {
        Lookup(DocStartToken, chunk, prefix, bestLen, pNextToken);
    }
}


void LookupLMatch(const TString &lmIndexDir, TBPEToken docStartToken, const TLMatchChunkIndex &chunk, TLMatchChunkResult *pRes)
{
    DebugPrintf("Lookup lmatch\n"); fflush(0);
    for (yint k = 0;; ++k) {
        TString fname = Sprintf("%s/chunk.%d", lmIndexDir.c_str(), (int)k);
        if (DoesFileExist(fname)) {
            TLMatchChunkIndex prev;
            Serialize(IO_READ, fname, prev);
            LookupPrevChunk(docStartToken, prev, chunk, pRes);
            printf("."); fflush(0);
        } else {
            break;
        }
    }
    printf("\n"); fflush(0);
}
