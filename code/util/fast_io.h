#pragma once

enum EIODirection
{
    IO_READ,
    IO_WRITE,
};

struct IBinaryStream
{
    enum {
        MAX_BLOCK_SIZE = 1 << 24
    };

	virtual ~IBinaryStream() {};
    yint Write(const void *userBuffer, yint size)
    {
        if (size < MAX_BLOCK_SIZE) {
            return WriteImpl(userBuffer, size);
        } else {
            const char *pData = (const char *)userBuffer;
            yint totalWritten = 0;
            for (yint offset = 0; offset < size;) {
                yint blkSize = Min<yint>(MAX_BLOCK_SIZE, size - offset);
                totalWritten += WriteImpl(pData + offset, blkSize);
                offset += blkSize;
            }
            return totalWritten;
        }
    }
    yint Read(void *userBuffer, yint size)
    {
        if (size < MAX_BLOCK_SIZE) {
            return ReadImpl(userBuffer, size);
        } else {
            char *pData = (char *)userBuffer;
            yint totalRead = 0;
            for (yint offset = 0; offset < size;) {
                yint blkSize = Min<yint>(MAX_BLOCK_SIZE, size - offset);
                totalRead += ReadImpl(pData + offset, blkSize);
                offset += blkSize;
            }
            return totalRead;
        }
    }
    virtual yint WriteImpl(const void *userBuffer, yint size) = 0;
    virtual yint ReadImpl(void *userBuffer, yint size) = 0;
    virtual bool IsValid() const = 0;
	virtual bool IsFailed() const = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef _MSC_VER

class TFileStream : public IBinaryStream
{
    HANDLE hFile;
    bool bFailed;
public:
    TFileStream(EIODirection ioDir, const TString &szFile) : bFailed(false)
    {
        DWORD dwAccess = 0, dwCreate = 0;
        if (ioDir == IO_READ) {
            dwAccess = GENERIC_READ;
            dwCreate = OPEN_EXISTING;
        } else {
            dwAccess = GENERIC_WRITE;
            dwCreate = OPEN_ALWAYS;
        }
        hFile = CreateFileA(szFile.c_str(), dwAccess, FILE_SHARE_READ, 0, dwCreate, FILE_ATTRIBUTE_NORMAL, 0);
    }
    ~TFileStream()
    {
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
        }
    }
    yint WriteImpl(const void *pData, yint size)
    {
        DWORD nWritten = 0;
        BOOL b = WriteFile(hFile, pData, size, &nWritten, 0);
        if (!b) {
            bFailed = true;
        }
        return nWritten;
    }
    yint ReadImpl(void *pData, yint size)
    {
        DWORD nRead = 0;
        BOOL b = ReadFile(hFile, pData, size, &nRead, 0);
        if (!b) {
            bFailed = true;
        }
        return nRead;
    }
    yint GetLength()
    {
        LARGE_INTEGER nLeng;
        GetFileSizeEx(hFile, (PLARGE_INTEGER)&nLeng);
        return nLeng.QuadPart;
    }
    yint Seek(yint pos)
    {
        LARGE_INTEGER i;
        i.QuadPart = pos;
        i.LowPart = SetFilePointer(hFile, i.LowPart, &i.HighPart, FILE_BEGIN);
        return i.QuadPart;
    }
    bool IsValid() const { return hFile != INVALID_HANDLE_VALUE; }
    bool IsFailed() const { return bFailed; }
};

#else

class TFileStream : public IBinaryStream
{
    FILE *File = 0;
public:
    TFileStream(EIODirection ioDir, const TString &szFile)
    {
        File = fopen(szFile.c_str(), (ioDir == IO_READ) ? "rb" : "wb");
    }
    ~TFileStream()
    {
        if (File) {
            fclose(File);
        }
    }
    yint WriteImpl(const void *pData, yint size)
    {
        if (File) {
            yint written = fwrite(pData, 1, size, File);
            if (written != size) {
                fclose(File);
                File = 0;
            }
            return written;
        }
        return 0;
    }
    yint ReadImpl(void *pData, yint size)
    {
        if (File) {
            yint readCount = fread(pData, 1, size, File);
            if (readCount != size) {
                fclose(File);
                File = 0;
            }
            return readCount;
        }
        return 0;
    }
    yint GetLength()
    {
        yint pos = ftello(File);
        fseeko(File, 0, SEEK_END);
        yint fsize = ftello(File);
        fseeko(File, pos, SEEK_SET);
        return fsize;
    }
    yint Seek(yint pos)
    {
        fseeko(File, pos, SEEK_SET);
        return ftello(File);
    }
    bool IsValid() const { return File != 0; }
    bool IsFailed() const { return File == 0; } // no recovery after fail so we just close the file in such case
};

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
class TMemStream : public IBinaryStream
{
    TVector<ui8> Data;
    yint Pos = 0;

private:
    yint WriteImpl(const void *userBuffer, yint size) override
    {
        if (size == 0) {
            return 0;
        }
        if (Pos + size > YSize(Data)) {
            Data.yresize(Pos + size);
        }
        memcpy(Data.data() + Pos, userBuffer, size);
        Pos += size;
        return size;
    }
    yint ReadImpl(void *userBuffer, yint size) override
    {
        yint res = Min<yint>(YSize(Data) - Pos, size);
        if (res > 0) {
            memcpy(userBuffer, &Data[Pos], res);
            Pos += res;
        }
        return res;
    }
    bool IsValid() const override
    {
        return true;
    }
    bool IsFailed() const override
    {
        return false;
    }

public:
    TMemStream() : Pos(0)
    {
    }
    TMemStream(TVector<ui8> *data) : Pos(0)
    {
        Data.swap(*data);
    }
    yint GetPos() const
    {
        return Pos;
    }
    yint GetLength() const
    {
        return YSize(Data);
    }
    void Seek(yint pos)
    {
        Y_VERIFY(pos >= 0);
        if (pos > YSize(Data)) {
            Data.resize(pos, 0);
        }
        Pos = pos;
    }
    void Truncate()
    {
        Data.resize(Pos);
    }
    void Swap(TVector<ui8> *data)
    {
        data->swap(Data);
        Pos = 0;
    }
    void Swap(TMemStream &p)
    {
        Data.swap(p.Data);
        Pos = 0;
        p.Pos = 0;
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// due to prefetching reads more then needed from IBinaryStream
class TBufferedStream : public TNonCopyable
{
    enum { PREFETCH_SIZE = 1 << 20 };
    TVector<ui8> Buf;
    IBinaryStream *Stream = 0;
    TMemStream *MemStream = 0;
    yint Pos = 0;
    yint BufSize = 0;
    yint StartStreamSize = 0;
    bool IsReadingFlag = false;
    bool IsEof = false;

private:
    void ReadLarge(void *userBuffer, yint size);
    void WriteLarge(const void *userBuffer, yint size);

public:
    TBufferedStream(EIODirection ioDir, IBinaryStream &stream) : Stream(&stream), IsReadingFlag(ioDir == IO_READ)
    {
        Buf.yresize(PREFETCH_SIZE);
        if (!IsReadingFlag) {
            BufSize = PREFETCH_SIZE;
        }
    }
    TBufferedStream(EIODirection ioDir, TMemStream &stream) : MemStream(&stream), IsReadingFlag(ioDir == IO_READ)
    {
        Pos = MemStream->GetPos();
        MemStream->Swap(&Buf);
        BufSize = YSize(Buf);
        StartStreamSize = BufSize;
    }
    ~TBufferedStream();

public:
    bool IsReading() const
    {
        return IsReadingFlag;
    }
    inline void Read(void *userBuffer, yint size)
    {
        Y_ASSERT(IsReadingFlag);
        if (!IsEof && size + Pos <= BufSize) {
            memcpy(userBuffer, Buf.data() + Pos, size);
            Pos += size;
        } else {
            ReadLarge(userBuffer, size);
        }
    }
    inline void Write(const void *userBuffer, yint size)
    {
        Y_ASSERT(!IsReadingFlag);
        if (Pos + size <= BufSize) {
            memcpy(Buf.data() + Pos, userBuffer, size);
            Pos += size;
        } else {
            WriteLarge(userBuffer, size);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
class TSeqReader
{
    enum { BUF_SIZE = 1 << 16 };
    TVector<char> Buf;
    yint Pos, BufSize;
    TFileStream F;
    bool IsEofFlag;
public:
    TSeqReader(const TString &szFile) : Pos(0), BufSize(0), F(IO_READ, szFile), IsEofFlag(false)
    {
        Buf.resize(BUF_SIZE);
    }
    TString ReadLine()
    {
        TString szRes;
        for(;;++Pos) {
            if (Pos == BufSize) {
                Pos = 0;
                BufSize = F.Read(Buf.data(), BUF_SIZE);
                if (BufSize == 0) {
                    IsEofFlag = true;
                    break;
                }
            }
            if (Buf[Pos] == '\x0d')
                continue;
            if (Buf[Pos] == '\x0a') {
                ++Pos;
                break;
            }
            szRes += Buf[Pos];
        }
        return szRes;
    }
    bool IsEof() const { return IsEofFlag; }
    bool IsValid() const { return F.IsValid(); }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
bool ReadWholeFile(const TString &szFileName, TVector<char> *res);
