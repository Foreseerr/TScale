#pragma once
#include <gpt/model_params/model_matrix.h>
#include <gpt/model_params/sse_utils.h>
#include <gpt/train_config/train_step.h>
#include <immintrin.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
constexpr yint MODEL_INT8_DELTA_TILE = 128;

struct TModelMatrixInt8Delta
{
    i8 *Data = 0;
    float *TileScale = 0;
    int XSize = 0;

    TModelMatrixInt8Delta(i8 *data, float *tileScale, int xSize) : Data(data), TileScale(tileScale), XSize(xSize) {}
    i8 *GetRow(yint y) const { return Data + y * XSize; }
    float *GetTileScaleRow(yint y) const { return TileScale + y * XSize / MODEL_INT8_DELTA_TILE; }
};

struct TModelMatrixHalfDelta
{
    struct TRow
    {
        float Scale;
        float Sum2; // sum2 of scaled values
    };
    yint SizeX = 0;
    yint SizeY = 0;
    TVector<fp16> Delta;
    TVector<TRow> Rows;

public:
    void Init(yint xSize, yint ySize)
    {
        SizeX = xSize;
        SizeY = ySize;
        ClearPodArray(&Delta, xSize * ySize);
        ClearPodArray(&Rows, ySize);
    }
    float CalcSum2() const
    {
        float sum2 = 0;
        for (const TRow &row : Rows) {
            sum2 += row.Sum2;
        }
        return sum2;
    }
    const fp16 *GetRow(yint y) const { return &Delta[y * SizeX]; }
    fp16 *GetRow(yint y) { return &Delta[y * SizeX]; }
    void GetAllData(TArray2D<float> *p) const;
};

void Copy(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta);
void Add(TModelMatrixHalfDelta *p, const TModelMatrixInt8Delta &delta);
void Compress(TModelMatrixInt8Delta *p, const TArray2D<float> &data);


///////////////////////////////////////////////////////////////////////////////////////////////////
struct TModelMatrixBitDelta
{
    TVector<float> DeltaRowDisp;
    TVector<ui64> BitDelta;
    SAVELOAD(DeltaRowDisp, BitDelta);

    bool IsEmpty() const
    {
        return BitDelta.empty();
    }
    void Clear()
    {
        DeltaRowDisp.resize(0);
        BitDelta.resize(0);
    }
    void Swap(TModelMatrixBitDelta *p)
    {
        DeltaRowDisp.swap(p->DeltaRowDisp);
        BitDelta.swap(p->BitDelta);
    }
};


struct TModelMatrixBitDeltaTail
{
    TVector<ui64> BitDelta;

    void Init(yint xSize, yint ySize)
    {
        Y_VERIFY((xSize % 64) == 0);
        ClearPodArray(&BitDelta, ySize * xSize / 64);
    }
};


void SumBitDelta(const TModelMatrixBitDelta &a, const TModelMatrixBitDelta &b, TModelMatrixBitDeltaTail *pTail, TModelMatrixBitDelta *pRes);


///////////////////////////////////////////////////////////////////////////////////////////////////
class TModelMatrixData
{
    TArray2D<float> Matr;
    TArray2D<float> AvrgDelta1; // single EMA is good enough
    float SumWeight = 0;
    TVector<float> RowDisp;
    float Sum2 = 0;

private:
    void OnDataUpdate()
    {
        Sum2 = CalcMatrixSum2(Matr);
    }

public:
    yint GetXSize() const { return Matr.GetXSize(); }
    yint GetYSize() const { return Matr.GetYSize(); }
    EModelMatrixDisp GetRowDispType() const { return YSize(RowDisp) > 1 ? MM_ROW_DISP : MM_MATRIX_DISP; }

    // direct data access
    float *GetRow(yint y) { return &Matr[y][0]; }
    float GetSum2() const { return Sum2; }

    // Set/Get ops
    void GetData(TModelMatrix *p) const
    {
        p->Create(Matr, AvrgDelta1, SumWeight, RowDisp);
    }
    void SetData(const TModelMatrix &data)
    {
        Matr = data.GetMatrix();
        AvrgDelta1 = data.GetGrad1();
        RowDisp = data.GetRowDisp();
        SumWeight = data.GetSumWeight();
        OnDataUpdate();
    }

    // delta ops
    void AddDelta(const TModelMatrixHalfDelta &delta, const TTrainingStep &step);
    bool AddBitDelta(const TModelMatrixBitDelta &bitDelta, const TTrainingStep &step);
    void CompressDelta(const TModelMatrixHalfDelta &delta, TModelMatrixBitDelta *pBitDelta, TArray2D<float> *pDeltaTail);

    // pack ops
    void PackMatrix(TBufferedStream &f) const
    {
        ::PackMatrix(f, Matr);
    }
    void AddPackedMatrix(TBufferedStream &f, float scale)
    {
        ::AddPackedMatrix(&Matr, f, scale);
        OnDataUpdate();
    }
    void GetRowDisp(TModelRowDisp *p) const
    {
        p->AddMatrixRowDisp(RowDisp, SumWeight);
    }
    void SetRowDisp(const TModelRowDisp &rd, yint *pPtr)
    {
        yint &ptr = *pPtr;
        for (float &val : RowDisp) {
            val = rd.RowDisp[ptr++] * rd.SumWeight;
        }
        SumWeight = rd.SumWeight;
    }
    void ScaleGrad(float x)
    {
        ScaleMatrixAligned(&AvrgDelta1, x);
    }
};
