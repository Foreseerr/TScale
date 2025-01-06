#pragma once
#include "fast_io.h"

template <class T> class TArray2D;
template<int n> struct TInt2Type {};


struct IBinSaver
{
public:
	typedef TString stdTString;
private:
	static char __cdecl TestDataPath(...) { return 0; }
    static int __cdecl TestDataPath(stdTString*) { return 0; }
	template<class T1>
    static int __cdecl TestDataPath(TArray2D<T1>*) { return 0; }
	template<class T1>
    static int __cdecl TestDataPath(TVector<T1>*) { return 0; }
	template<class T1>
    static int __cdecl TestDataPath(list<T1>*) { return 0; }
	template<class T1, class T2, class T3>
    static int __cdecl TestDataPath(THashMap<T1,T2,T3>*) { return 0; }
	//
	template<class T>
	void __cdecl CallObjectSerialize(T *p, ...)
	{
		//p->T::operator&(*this); // does not allow virtual operator&
        (*p) &(*this);
	}
	template<class T>
	void __cdecl CallObjectSerialize(T *p, TInt2Type<1> *pp)
	{
		DataChunk(p, sizeof(T));
	}
	//
	// TVector
	template <class T> void DoTVector(TVector<T> &data)
	{
		yint nSize;
		if (IsReading) {
			data.clear();
			Add(&nSize);
			data.yresize(nSize);
		} else {
			nSize = YSize(data);
			Add(&nSize);
		}
		for (yint i = 0; i < nSize; i++) {
            Add(&data[i]);
        }
	}
	template <class T> void DoDataTVector(TVector<T> &data)
	{
		yint nSize = YSize(data);
        if (IsReading) {
            data.clear();
            Add(&nSize);
            data.yresize(nSize);
        } else {
            nSize = YSize(data);
            Add(&nSize);
        }
        DataChunk(data.data(), sizeof(T) * nSize);
	}
	// THashMap
	template <class T1,class T2,class T3> 
		void DoHashMap(THashMap<T1,T2,T3> &data)
	{
		if (IsReading) {
			data.clear();
			yint nSize;
			Add(&nSize);
			TVector<T1> indices;
			indices.resize(nSize);
			for (yint i = 0; i < nSize; ++i) {
                Add(&indices[i]);
            }
            for (yint i = 0; i < nSize; ++i) {
                Add(&data[indices[i]]);
            }
		} else {
			yint nSize = YSize(data);
			Add(&nSize);
			TVector<T1> indices;
			indices.resize(nSize);
			yint i = 1;
            for (typename THashMap<T1, T2, T3>::iterator pos = data.begin(); pos != data.end(); ++pos, ++i) {
                indices[nSize - i] = pos->first;
            }
            for (i = 0; i < nSize; ++i) {
                Add(&indices[i]);
            }
            for (i = 0; i < nSize; ++i) {
                Add(&data[indices[i]]);
            }
		}
	}
	template<class T> void Do2DArray(TArray2D<T> &a)
	{
		yint nXSize = GetXSize(a), nYSize = GetYSize(a);
		Add(&nXSize);
		Add(&nYSize);
		if (IsReading) {
            a.SetSizes(nXSize, nYSize);
        }
        for (yint y = 0; y < nYSize; ++y) {
            for (yint x = 0; x < nXSize; ++x) {
                Add(&a[y][x]);
            }
        }
	}
    template<class T> void Do2DArrayData(TArray2D<T> &a)
    {
        yint nXSize = GetXSize(a), nYSize = GetYSize(a);
        Add(&nXSize);
	    Add(&nYSize);
        if (IsReading) {
            a.SetSizes(nXSize, nYSize);
        }
        if (nXSize * nYSize > 0) {
            DataChunk(&a[0][0], sizeof(T) * nXSize * nYSize);
        }
    }

    // to serialize to mem these functions should be virtual?
    TBufferedStream &BufIO;
	bool IsReading;

	void DataChunk(void *pData, yint size)
	{
		if (IsReading) {
            BufIO.Read(pData, size);
		} else {
            BufIO.Write(pData, size);
		}
	}
	void DataChunkTString(TString &data)
	{
        if (IsReading) {
            int count = 0;
            BufIO.Read(&count, 4);
            if (count > 0) {
                data.resize(count);
                BufIO.Read(&data[0], count);
            }
        } else {
            int count = data.size();
            BufIO.Write(&count, 4);
            if (count > 0) {
                BufIO.Write(&data[0], count);
            }
        }
	}

public:
    void AddRawData(void *pData, int nSize) { DataChunk(pData, nSize); }

    template <class T>
    static constexpr bool HasTrivialSerializer(T *p)
    {
        return sizeof(TestDataPath(p)) == 1 && sizeof((*p) & (*(IBinSaver*)nullptr)) == 1;
    }

    template<class T>
    void Add(T *p) 
	{
		const int N_HAS_SERIALIZE_TEST = sizeof((*p)&(*this));
        TInt2Type<N_HAS_SERIALIZE_TEST> separator;
		CallObjectSerialize(p, &separator);
	}
	void Add(stdTString *pStr)
	{
		DataChunkTString(*pStr);
	}
	template<class T1>
	void Add(TVector<T1> *pVec) 
	{
        if (HasTrivialSerializer<T1>(nullptr)) {
            DoDataTVector(*pVec);
        } else {
            DoTVector(*pVec);
        }
	}
	template<class T1, class T2, class T3>
	void Add(THashMap<T1,T2,T3> *pHash) 
	{
		DoHashMap(*pHash);
	}
	template<class T1>
	void Add(TArray2D<T1> *pArr)
	{
        if (HasTrivialSerializer<T1>(nullptr)) {
            Do2DArrayData(*pArr);
        } else {
            Do2DArray(*pArr);
        }
	}
	template<class T1>
	void Add(list<T1> *pList) 
	{
		list<T1> &data = *pList;
		if (IsReading) {
			yint nSize;
			Add(&nSize);
			data.clear();
			data.insert(data.begin(), nSize, T1());
		} else {
			yint nSize = YSize(data);
			Add(&nSize);
		}
        for (typename list<T1>::iterator k = data.begin(); k != data.end(); ++k) {
            Add(&(*k));
        }
	}
	template <class T1, class T2> 
	void Add(pair<T1, T2> *pData) 
	{
		Add(&(pData->first));
		Add(&(pData->second));
	}
    // variadic add multiple fields
    template <typename T>
    void AddVariadic(T &a)
    {
        Add(&a);
    }
    template <typename T, typename... TRest>
    void AddVariadic(T &a, TRest&...x)
    {
        Add(&a);
        AddVariadic(x...);
    }

	IBinSaver(TBufferedStream &bufIO) : BufIO(bufIO), IsReading(bufIO.IsReading()) {}
};
template <class T>
inline char operator&(T &c, IBinSaver &ss) { return 0; }


////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void ReadStruct(TBufferedStream &bufIO, T &c)
{
    IBinSaver f(bufIO);
    f.Add(&c);
}

template<class T>
inline void WriteStruct(TBufferedStream &bufIO, T &c)
{
    IBinSaver f(bufIO);
    f.Add(&c);
}

template<class T>
inline void Serialize(EIODirection ioDir, const TString &szName, T &c)
{
    TFileStream file(ioDir, szName);
    Y_VERIFY(file.IsValid() && "file not found or can not be created");
    TBufferedStream bufIO(ioDir, file);
	IBinSaver f(bufIO);
	f.Add(&c);
}

#define SAVELOAD(...) int operator&(IBinSaver &f) { f.AddVariadic(__VA_ARGS__); return 0; }
#define SAVELOAD_OVERRIDE(...) int operator&(IBinSaver &f) override { f.AddVariadic(__VA_ARGS__); return 0; }
