#pragma once
#include <gpt/att/nodes_batch.h>
#include <gpt/model_params/model_params.h>
#include "par_matrix.h"


constexpr int MAIN_DEVICE = 0;

using NCuda::TParModelMatrix;
using NCuda::TCPUMatrixAdd;
using NCuda::TModelMatrixScale;
using NCuda::EModelMatrixQuant;
using NCuda::TParModelMatrixBase;

///////////////////////////////////////////////////////////////////////////////////////////////////
struct TAttentionParams : public TThrRefBase
{
    TVector<TIntrusivePtr<TParModelMatrix<TFastModelFloat>>> MatrArr;
    float AlibiSlope = 0;
    yint AttentionWidthId = 0;

    TAttentionParams(TIntrusivePtr<TCPUMatrixAdd> cpuMatrixAdd, TIntrusivePtr<TModelMatrixScale> matrixScale,
        const TModelParams::TAttentionMatrices &att, const TModelDescr::TAttentionPosParams &layerParams,
        EModelMatrixQuant quant);
    void GetParams(TModelParams::TAttentionMatrices *p, TModelDescr::TAttentionPosParams *pLayerParams);
    void SetParams(const TModelParams::TAttentionMatrices &att, const TModelDescr::TAttentionPosParams &layerParams);
    void GetGradient(TModelParams::TAttentionMatrices *p, TModelDescr::TAttentionPosParams *pLayerParams);
};


struct IModel : public TThrRefBase
{
    // assume no operations in fly
    virtual void GetParamsImpl(TModelParams *p) = 0;
    virtual void SetParamsImpl(const TModelParams &p) = 0;
    virtual void GetGradientImpl(TModelParams *p) = 0;
    // retrieve param storage
    virtual TModelDescr GetModelDescr() = 0;
    virtual TIntrusivePtr<TParModelMatrix<TEmbedFloat>> GetEmbedding() = 0;
    virtual TIntrusivePtr<TParModelMatrix<TFastModelFloat>> GetFinalLayer() = 0;
    virtual const TAttentionParams &GetAttention(yint d) = 0;
    virtual const TVector<float> &GetBias() = 0;
    virtual TModelMatrixScale *GetMatrixScale() = 0;
    virtual NCuda::TCudaPOD<int> GetCurrentIteration() = 0;
    // 
    virtual yint GetDeviceCount() = 0;
    virtual void StartIteration(const TTrainingStep &step, EAddToModel addToModel) = 0;
    virtual void WaitCompute() = 0;
    virtual void WaitDelayedCompute() = 0;
    virtual void ConvertMatrices() = 0; // for internal use, performs delayed conversion
};


///////////////////////////////////////////////////////////////////////////////////////////////////
struct IComputeContext : public TThrRefBase
{
protected:
    virtual IModel *GetModel() = 0;
    virtual void OnParamsUpdate() = 0;

    template <class T>
    void ForEachModelMatrix(T func)
    {
        IModel *pModel = GetModel();
        TModelDescr modelDescr = GetModelDescr();
        func(pModel->GetEmbedding().Get());
        func(pModel->GetFinalLayer().Get());
        for (yint d = 0; d < YSize(modelDescr.Layers); ++d) {
            const TAttentionParams &att = pModel->GetAttention(d);
            for (auto &mm : att.MatrArr) {
                func(mm.Get());
            }
        }
    }

public:
    void GetParams(TModelParams *p)
    {
        WaitUpdates();
        GetModel()->GetParamsImpl(p);
    }

    void SetParams(const TModelParams &p)
    {
        WaitUpdates();
        GetModel()->SetParamsImpl(p);
        OnParamsUpdate();
    }

    void GetGradient(TModelParams *p)
    {
        WaitUpdates();
        GetModel()->GetGradientImpl(p);
    }

    void ScaleGrad(float x)
    {
        WaitUpdates();
        ForEachModelMatrix([&](TParModelMatrixBase *p) { p->ScaleGrad(x); });
    }

    void PackMatrices(TBufferedStream &f)
    {
        WaitUpdates();
        ForEachModelMatrix([&](TParModelMatrixBase *p) { p->PackMatrix(f); });
    }

    // should call WaitUpdates() before and ApplyModifiedMatrices() after
    void AddPackedMatricesImpl(TBufferedStream &f, float scale)
    {
        //WaitUpdates();
        ForEachModelMatrix([&](TParModelMatrixBase *p) { p->AddPackedMatrixImpl(f, scale); });
        //OnParamsUpdate();
    }
    void ApplyModifiedMatrices()
    {
        GetModel()->ConvertMatrices();
        OnParamsUpdate();
    }

    void GetRowDisp(TModelRowDisp *pRes)
    {
        WaitUpdates();
        ForEachModelMatrix([&](TParModelMatrixBase *p) { p->GetRowDisp(pRes); });
    }

    void SetRowDisp(const TModelRowDisp &rd)
    {
        WaitUpdates();
        yint ptr = 0;
        ForEachModelMatrix([&](TParModelMatrixBase *p) { p->SetRowDisp(rd, &ptr); });
        OnParamsUpdate();
    }

public:
    enum EInitType {
        INIT_TRAIN,
        INIT_TEST
    };
    virtual yint GetDeviceCount() = 0;
    virtual TModelDescr GetModelDescr() = 0;
    virtual float GetAvrgTrainErr() = 0;

    virtual void WaitUpdates() = 0;
    virtual TNodesBatch &GetNodes(yint deviceId) = 0;
    virtual TVector<ui32> &GetDropTable(yint deviceId) = 0;

    virtual void Init(yint deviceId, EInitType initType) = 0;
    virtual void ComputeFinalStateVectors(TVector<TVector<float>> *pStateVectors) = 0;
    virtual void ComputeFragmentPredictions(TVector<TVector<float>> *pPrediction) = 0;
    virtual void ComputeFragmentPredictions(TVector<float> *pPrediction) = 0;
    virtual float ComputeScore() = 0;
    virtual void Backprop(const TTrainingStep &step, EAddToModel addToModel) = 0;

};


struct IMMDeltaHookGen;
TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params, IMMDeltaHookGen *deltaHookGen);
TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelDescr &modelDescr, const TVector<float> &biasArr, IMMDeltaHookGen *deltaHookGen);

inline TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params)
{
    return CreateModel(deviceCount, params, nullptr);
}
