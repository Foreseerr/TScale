#pragma once


class TFormattedReader
{
    TSeqReader File;
    TVector<const char*> Words;
    string Buf;
    int Line;
public:
    TFormattedReader(const TString &filename)
        : File(filename)
        , Line(-1)
    {
        if (!File.IsValid())
            abort();
    }
    bool ReadLine()
    {
        if (File.IsEof())
            return false;
        ++Line;
        Buf = File.ReadLine();
        if (Buf.empty())
            return false;
        Split(&Buf[0], &Words);
        return true;
    }
    const TVector<const char*>& GetWords() const { return Words; }
    int GetLine() const { return Line; } // zero based line number
};
