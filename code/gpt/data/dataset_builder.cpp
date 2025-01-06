#include "stdafx.h"
#include "dataset_builder.h"
#include "ppm_lmatch.h"
#include "text_saveload.h"
#include <lib/file/dir.h>
#include <util/thread.h>
#include <lib/hp_timer/hp_timer.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TThreadHolder : public TThrRefBase
{
    TThread Thr;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
static void ReadChunk(const TString &dir, TLMatchChunkIndex *p, yint id)
{
    Serialize(IO_READ, Sprintf("%s/chunk.%d", dir.c_str(), (int)id), *p);
}

static void WriteChunk(const TString &dir, TLMatchChunkIndex &chunk, yint id)
{
    Serialize(IO_WRITE, Sprintf("%s/chunk.%d", dir.c_str(), (int)id), chunk);
}

static void ReadChunk(const TString &dir, TLMatchChunkResult *p, yint id)
{
    Serialize(IO_READ, Sprintf("%s/res.%d", dir.c_str(), (int)id), *p);
}

static void WriteChunk(const TString &dir, TLMatchChunkResult &chunk, yint id)
{
    Serialize(IO_WRITE, Sprintf("%s/res.%d", dir.c_str(), (int)id), chunk);
}


struct TIndexLMatchWorkers
{
    enum {
        INDEX_CHUNK,
        LOOKUP_CHUNK,
    };
    TBPEToken DocStartToken;
    TVector<TIntrusivePtr<TThreadHolder>> Workers;
    TString LMatchIndexDir;
    yint ChunkCount;
    yint Op = 0;
    std::atomic<yint> CurChunk;

public:
    TIndexLMatchWorkers(TBPEToken docStartToken, const TString &dir, yint chunkCount) : DocStartToken(docStartToken), LMatchIndexDir(dir), ChunkCount(chunkCount)
    {
    }

    void Run(yint op, yint workerCount)
    {
        Op = op;
        CurChunk = ChunkCount - 1;
        for (yint k = 0; k < workerCount; ++k) {
            TThreadHolder *p = new TThreadHolder;
            p->Thr.Create(this);
            Workers.push_back(p);
        }
    }

    void WaitCompletion()
    {
        Workers.clear();
    }

    void WorkerThread()
    {
        for (;;) {
            yint id = CurChunk.fetch_add(-1);
            if (id < 0) {
                break;
            }
            if (Op == INDEX_CHUNK) {
                //NHPTimer::STime tStart;
                //NHPTimer::GetTime(&tStart);
                TLMatchChunkIndex chunk;
                ReadChunk(LMatchIndexDir, &chunk, id);
                TLMatchChunkResult res;
                IndexChunk(DocStartToken, &chunk, &res);
                WriteChunk(LMatchIndexDir, chunk, id);
                WriteChunk(LMatchIndexDir, res, id);
                //DebugPrintf("%g secs\n", NHPTimer::GetTimePassed(&tStart));

            } else if (Op == LOOKUP_CHUNK) {
                TLMatchChunkResult res;
                ReadChunk(LMatchIndexDir, &res, id);
                TLMatchChunkIndex cur;
                ReadChunk(LMatchIndexDir, &cur, id);
                for (yint k = 0; k < id; ++k) {
                    TLMatchChunkIndex prev;
                    ReadChunk(LMatchIndexDir, &prev, k);
                    LookupPrevChunk(DocStartToken, prev, cur, &res);
                }
                WriteChunk(LMatchIndexDir, res, id);

            } else {
                Y_VERIFY(0 && "unknown op");
            }
            printf("."); fflush(0);
        }
    }
};


void TDatasetBuilder::ComputeOnDiskLMatch(const TString &lmIndexDir)
{
    Y_VERIFY(!Dataset->DocsetArr.empty());
    Y_VERIFY(!lmIndexDir.empty() && "should specify LMatch index location")
    TString lmIndexFilename = lmIndexDir + LMATCH_INDEX_FNAME;
    // set lmatch index offset for all docsets
    yint lmIndexOffset = 0;
    for (auto &ds : Dataset->DocsetArr) {
        ds.LMatchFilename = lmIndexFilename;
        ds.LMatchOffset = lmIndexOffset;
        lmIndexOffset += ds.TotalTokens;
    }

    // create index if it does not exist
    if (!DoesFileExist(lmIndexFilename)) {
        //constexpr yint MAX_CHUNK_SIZE = 2000000000;
        constexpr yint MAX_CHUNK_SIZE = 500 * 1000000;
        //constexpr yint MAX_CHUNK_SIZE = 100 * 1000000; // test 100m chunks
        constexpr yint WORKER_THREADS = 8;
        //constexpr yint WORKER_THREADS = 1;
        yint chunkCount = 0;
        yint bytesPerToken = Dataset->DocsetArr[0].BytesPerToken;
        // segment all text into chunks
        {
            printf("Generate chunks\n"); fflush(0);
            TLMatchChunkIndex chunk;
            chunk.FillHead();
            for (auto &ds : Dataset->DocsetArr) {
                Y_VERIFY(bytesPerToken == ds.BytesPerToken && "all datasets should have same bytes per token");
                TIntrusivePtr<TPackedBPETokenReader> reader = new TPackedBPETokenReader(ds.IndexFilename, ds.BytesPerToken);
                for (yint offset = 0; offset < ds.TotalTokens;) {
                    yint chunkPtr = YSize(chunk.Text);
                    yint sz = Min<yint>(ds.TotalTokens - offset, MAX_CHUNK_SIZE - YSize(chunk.Text));
                    TVector<TBPEToken> frag;
                    reader->Read(offset, sz, &frag);
                    chunk.Text.insert(chunk.Text.end(), frag.begin(), frag.end());
                    offset += sz;
                    if (YSize(chunk.Text) == MAX_CHUNK_SIZE) {
                        WriteChunk(lmIndexDir, chunk, chunkCount++);
                        yint copySize = MAX_COMPACT_INDEX_MATCH_LEN + 1;
                        for (yint k = 0; k < copySize; ++k) {
                            chunk.Text[k] = chunk.Text[MAX_CHUNK_SIZE - copySize + k];
                        }
                        chunk.Text.resize(copySize);
                        printf("."); fflush(0);
                    }
                }
            }
            chunk.Text.push_back(UNDEFINED_TOKEN);
            WriteChunk(lmIndexDir, chunk, chunkCount++);
            printf("\n"); fflush(0);
        }

        TIndexLMatchWorkers workers(GetDocStartToken(), lmIndexDir, chunkCount);

        printf("Index each chunk\n"); fflush(0);
        workers.Run(TIndexLMatchWorkers::INDEX_CHUNK, WORKER_THREADS);
        workers.WaitCompletion();
        printf("\n"); fflush(0);

        printf("Lookup history for each chunk\n"); fflush(0);
        workers.Run(TIndexLMatchWorkers::LOOKUP_CHUNK, WORKER_THREADS);
        workers.WaitCompletion();
        printf("\n"); fflush(0);

        // write result
        {
            printf("Write result\n"); fflush(0);
            TIntrusivePtr<TPackedBPETokenWriter> writeLMatch = new TPackedBPETokenWriter(lmIndexFilename, bytesPerToken);
            yint resSize = 0;
            for (yint id = 0; id < chunkCount; ++id) {
                TLMatchChunkResult res;
                ReadChunk(lmIndexDir, &res, id);
                writeLMatch->Write(res.NextToken);
                resSize += YSize(res.NextToken);
                printf("."); fflush(0);
            }
            Y_VERIFY(resSize == lmIndexOffset);
            printf("\n"); fflush(0);
        }

        // erase res.* files
        for (yint id = 0; id < chunkCount; ++id) {
            EraseFile(Sprintf("%s/res.%d", lmIndexDir.c_str(), (int)id));
        }
    }
}


void TDatasetBuilder::ComputeInMemoryLMatch()
{
    DebugPrintf("in-memory LMatch compute\n");
    TLMatchChunkIndex chunk;
    chunk.FillHead();
    for (auto &ds : Dataset->DocsetArr) {
        chunk.Text.insert(chunk.Text.end(), ds.Text.begin(), ds.Text.end());
    }
    chunk.Text.push_back(UNDEFINED_TOKEN);

    TLMatchChunkResult res;
    IndexChunk(GetDocStartToken(), &chunk, &res);

    yint ptr = 0;
    for (auto &ds : Dataset->DocsetArr) {
        yint len = YSize(ds.Text);
        ds.LMatch.resize(len);
        for (yint t = 0; t < len; ++t) {
            ds.LMatch[t] = res.NextToken[ptr + t];
        }
        ptr += len;
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
TIntrusivePtr<TDataset> MakeCharDataset(TTokenizer *pTokenizer, const TVector<char> &text, float testFraction, bool usePPM, bool useLMatch)
{
    pTokenizer->MakeUsedLettersEncoder(text);

    TVector<TBPEToken> data;
    yint charLen = pTokenizer->GenWords(text, 0, YSize(text), &data);

    TDatasetParams params(pTokenizer->GetVocabSize());
    params.CountDocset(data, 0, charLen, testFraction);

    Y_VERIFY(!pTokenizer->HasDocStartToken());
    TIntrusivePtr<TDatasetBuilder> db = new TDatasetBuilder(usePPM, useLMatch, pTokenizer->GetVocabSize(), -1, pTokenizer->GetFragmentStartToken());
    float weight = 1;
    db->AddTokenizedDocset(data, params, weight);
    return db->MakeDataset("");
}


void AddDocset(TIntrusivePtr<TDatasetBuilder> pBuilder, const TTokenizer &tokenizer, const TVector<TVector<char>> &docSet, float weight, float testFraction)
{
    TVector<TBPEToken> data;
    TBPEToken docStart = tokenizer.GetDocStartToken();
    data.push_back(docStart);
    yint totalLen = 0;
    for (const TVector<char> &text : docSet) {
        totalLen += tokenizer.GenWords(text, 0, YSize(text), &data);
        data.push_back(docStart);
    }
    TDatasetParams params(tokenizer.GetVocabSize());
    params.CountDocset(data, 0, totalLen, testFraction);

    pBuilder->AddTokenizedDocset(data, params, weight);
}



///////////////////////////////////////////////////////////////////////////////////////////////////
struct TIndexedDataset
{
    TDatasetParams Params;
    yint VocabSize = 0;
    SAVELOAD(Params, VocabSize);
};


void AddIndexedDocset(TDatasetBuilder *pBuilder, const TString &dir, float weight)
{
    TIndexedDataset hdr;
    Serialize(IO_READ, dir + DATASET_HDR_FNAME, hdr);
    pBuilder->AddIndexedDocset(dir, hdr.Params, hdr.VocabSize, weight);
}


struct TDocsetIndexContext
{
    const TTokenizer &Tokenizer;
    yint DocStartToken = 0;
    TString Dir;
    bool UsePPM = false;
    float TestFraction = 0;
    TVector<TFindFileResult> AllFiles;
    std::atomic<yint> CurFileId;
    TVector<TIntrusivePtr<TThreadHolder>> Workers;
    TAtomic WriteLock;
    yint Offset = 0;
    TIntrusivePtr<TPackedBPETokenWriter> IndexFile;
    TIntrusivePtr<TPackedBPETokenWriter> IndexFilePPM;
    TDatasetParams Params;
    yint TokenWidth = 0;
    yint SrcHeaderSize = 0;

public:
    TDocsetIndexContext(const TTokenizer &tokenizer, yint vocabSize, yint docStartToken, const TString &dir, bool usePPM, float testFraction)
        : Tokenizer(tokenizer)
        , DocStartToken(docStartToken)
        , Dir(dir)
        , UsePPM(usePPM)
        , TestFraction(testFraction)
        , CurFileId(0)
        , WriteLock(0)
        , Params(vocabSize)
    {
        // erase old result
        EraseFile(dir + DATASET_INDEX_FNAME);
        EraseFile(dir + DATASET_HDR_FNAME);
        EraseFile(dir + DATASET_PPM_FNAME);
        // 
        FindAllFiles(dir, &AllFiles);
        //
        IndexFile = new TPackedBPETokenWriter(dir + DATASET_INDEX_FNAME, Params.BytesPerToken);
        if (UsePPM) {
            IndexFilePPM = new TPackedBPETokenWriter(dir + DATASET_PPM_FNAME, Params.BytesPerToken);
        }
    }

    void SetLoadTokenizedParams(yint tokenWidth, yint srcHeaderSize)
    {
        TokenWidth = tokenWidth;
        SrcHeaderSize = srcHeaderSize;
    }

    void RunWorkers(yint workerCount)
    {
        for (yint k = 0; k < workerCount; ++k) {
            TThreadHolder *p = new TThreadHolder;
            p->Thr.Create(this);
            Workers.push_back(p);
        }
    }

    void WaitCompletion()
    {
        Workers.clear();
    }

    void WorkerThread()
    {
        for (;;) {
            yint fileId = CurFileId.fetch_add(1);
            if (fileId >= YSize(AllFiles)) {
                return;
            }
            const TFindFileResult &ff = AllFiles[fileId];
            if (ff.IsDir) {
                continue;
            }

            TVector<TBPEToken> data;
            TString srcFileName = Dir + "/" + ff.Name;
            yint utf8charCount = 0;
            if (Tokenizer.IsEmpty()) {
                // load tokenized
                LoadTokenized(srcFileName, TokenWidth, SrcHeaderSize, &data);
                utf8charCount += YSize(data);
            } else {
                TVector<TVector<char>> docSet;
                LoadDocumentSetFromBin(&docSet, srcFileName);

                //NHPTimer::STime tStart;
                //NHPTimer::GetTime(&tStart);
                TBPEToken docStart = DocStartToken;
                data.push_back(docStart);
                for (const TVector<char> &text : docSet) {
                    utf8charCount += Tokenizer.GenWords(text, 0, YSize(text), &data);
                    data.push_back(docStart);
                }
            }

            TVector<TBPEToken> ppm;
            if (UsePPM) {
                ComputeWindowPPM(data, &ppm, DocStartToken);
            }

            {
                TGuard<TAtomic> gg(WriteLock);
                Params.CountDocset(data, Offset, utf8charCount, TestFraction);
                IndexFile->Write(data);
                if (UsePPM) {
                    IndexFilePPM->Write(ppm);
                }
                Offset += YSize(data);
                DebugPrintf(".");
                //DebugPrintf("time passed %g\n", NHPTimer::GetTimePassed(&tStart));
            }
        }
    }
};


static void IndexDir(const TString &dir, TDocsetIndexContext &ctx, yint vocabSize)
{
    const yint WORKER_COUNT = 8;
    ctx.RunWorkers(WORKER_COUNT);
    ctx.WaitCompletion();

    TIndexedDataset hdr;
    hdr.Params = ctx.Params;
    hdr.VocabSize = vocabSize;
    Serialize(IO_WRITE, dir + "/index_hdr.bin", hdr);
    DebugPrintf("\n");
}


void IndexDocsetDir(const TString &dir, const TTokenizer &tokenizer, bool usePPM, float testFraction)
{
    DebugPrintf("Indexing %s folder\n", dir.c_str());
    yint vocabSize = tokenizer.GetVocabSize();
    yint docStartToken = tokenizer.GetDocStartToken();
    TDocsetIndexContext ctx(tokenizer, vocabSize, docStartToken, dir, usePPM, testFraction);
    IndexDir(dir, ctx, tokenizer.GetVocabSize());
}


void IndexTokenizedDir(const TString &dir, yint vocabSize, yint docStartToken, bool usePPM, float testFraction, yint tokenWidth, yint headerSize)
{
    DebugPrintf("Indexing tokenized data in %s folder\n", dir.c_str());
    TTokenizer emptyTokenizer;
    TDocsetIndexContext ctx(emptyTokenizer, vocabSize, docStartToken, dir, usePPM, testFraction);
    ctx.SetLoadTokenizedParams(tokenWidth, headerSize);
    IndexDir(dir, ctx, vocabSize);
}

