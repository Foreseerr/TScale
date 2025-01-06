#include "stdafx.h"
#include "model.h"
#include <lib/cuda/cuda_arrays.h>
#include "par_matrix.h"

using namespace NCuda;


///////////////////////////////////////////////////////////////////////////////////////////////////
TAttentionParams::TAttentionParams(TIntrusivePtr<TCPUMatrixAdd> cpuMatrixAdd, TIntrusivePtr<TModelMatrixScale> matrixScale,
    const TModelParams::TAttentionMatrices &att, const TModelDescr::TAttentionPosParams &layerParams,
    EModelMatrixQuant quant)
{
    yint count = YSize(att.MatrArr);
    MatrArr.resize(count);
    for (yint k = 0; k < count; ++k) {
        MatrArr[k] = CreateModelMatrix<TFastModelFloat>(cpuMatrixAdd, matrixScale, MODEL_DISCR_SCALE, att.MatrArr[k], quant, MM_SYNC_GRADIENT);
    }
    AlibiSlope = layerParams.AlibiSlope;
    AttentionWidthId = layerParams.AttentionWidthId;
}

void TAttentionParams::GetParams(TModelParams::TAttentionMatrices *p, TModelDescr::TAttentionPosParams *pLayerParams)
{
    yint count = YSize(MatrArr);
    p->MatrArr.resize(count);
    for (yint k = 0; k < count; ++k) {
        MatrArr[k]->GetData(&p->MatrArr[k]);
    }
    pLayerParams->AlibiSlope = AlibiSlope;
    pLayerParams->AttentionWidthId = AttentionWidthId;
}

void TAttentionParams::SetParams(const TModelParams::TAttentionMatrices &att, const TModelDescr::TAttentionPosParams &layerParams)
{
    yint count = YSize(MatrArr);
    for (yint k = 0; k < count; ++k) {
        MatrArr[k]->SetData(att.MatrArr[k]);
    }
    AlibiSlope = layerParams.AlibiSlope;
    AttentionWidthId = layerParams.AttentionWidthId;
}

void TAttentionParams::GetGradient(TModelParams::TAttentionMatrices *p, TModelDescr::TAttentionPosParams *pLayerParams)
{
    yint count = YSize(MatrArr);
    p->MatrArr.resize(count);
    for (yint k = 0; k < count; ++k) {
        MatrArr[k]->GetDeltaData(&p->MatrArr[k]);
    }
    pLayerParams->AlibiSlope = AlibiSlope;
    pLayerParams->AttentionWidthId = AttentionWidthId;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
class TModel : public IModel
{
    TModelDescr ModelDescr;
    TIntrusivePtr<TCPUMatrixAdd> MatrixAdd;

    TIntrusivePtr<TModelMatrixScale> MatrixScale;
    TIntrusivePtr<TParModelMatrix<TFastModelFloat>> FinalLayer;
    TIntrusivePtr<TParModelMatrix<TEmbedFloat>> Embedding;
    TVector<TIntrusivePtr<TAttentionParams>> LayerArr;
    TVector<float> Bias;
    TIntrusivePtr<IMMDeltaHookGen> DeltaHookGen;


    void Create(yint deviceCount, const TModelParams &modelParams)
    {
        ModelDescr = modelParams.ModelDescr;
        yint attentionCount = ModelDescr.GetAttentionCount();
        yint maxMatrixCount = DivCeil(attentionCount * MP_ATT_COUNT + MP_MODEL_COUNT, 32) * 32;

        EModelMatrixQuant quant = MM_QUANT_NONE;
        if (ModelDescr.HasFlag(MPF_SIM_QUANT_2BIT)) {
            quant = MM_QUANT_2BIT;
        } else if (ModelDescr.HasFlag(MPF_SIM_QUANT_4BIT)) {
            quant = MM_QUANT_4BIT;
        }

        MatrixScale = new TModelMatrixScale(maxMatrixCount);
        MatrixAdd = new TCPUMatrixAdd(deviceCount, maxMatrixCount, DeltaHookGen.Get());
        FinalLayer = CreateModelMatrix<TFastModelFloat>(MatrixAdd, MatrixScale, MODEL_DISCR_SCALE, modelParams.MatrArr[MP_MODEL_FINAL], MM_QUANT_NONE, MM_SYNC_GRADIENT);
        Embedding = CreateModelMatrix<TEmbedFloat>(MatrixAdd, MatrixScale, MODEL_DISCR_SCALE, modelParams.MatrArr[MP_MODEL_EMBED], MM_QUANT_NONE, MM_DELAY_GRADIENT);
        LayerArr.resize(YSize(ModelDescr.Layers));
        for (yint d = 0; d < YSize(ModelDescr.Layers); ++d) {
            LayerArr[d] = new TAttentionParams(MatrixAdd, MatrixScale, modelParams.LayerArr[d], ModelDescr.Layers[d], quant);
        }
        Bias = modelParams.Bias;
        Y_VERIFY(YSize(Bias) == ModelDescr.VocabSize);
        MatrixAdd->LaunchWorkers();
    }

public:
    TModel(yint deviceCount, const TModelParams &modelParams, IMMDeltaHookGen *deltaHookGen)
        : DeltaHookGen(deltaHookGen)
    {
        Create(deviceCount, modelParams);
    }

    void GetParamsImpl(TModelParams *p) override
    {
        p->ModelDescr = ModelDescr;
        yint depth = YSize(LayerArr);
        p->LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            LayerArr[d]->GetParams(&p->LayerArr[d], &p->ModelDescr.Layers[d]);
        }
        p->MatrArr.resize(MP_MODEL_COUNT);
        FinalLayer->GetData(&p->MatrArr[MP_MODEL_FINAL]);
        Embedding->GetData(&p->MatrArr[MP_MODEL_EMBED]);
        p->Bias = Bias;
    }

    void SetParamsImpl(const TModelParams &p) override
    {
        Y_VERIFY(ModelDescr == p.GetModelDescr());
        yint depth = YSize(p.LayerArr);
        LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            LayerArr[d]->SetParams(p.LayerArr[d], p.ModelDescr.Layers[d]);
        }
        FinalLayer->SetData(p.MatrArr[MP_MODEL_FINAL]);
        Embedding->SetData(p.MatrArr[MP_MODEL_EMBED]);
        Bias = p.Bias;
        //NeedCopyToDevice = true;
    }

    void GetGradientImpl(TModelParams *p) override
    {
        p->ModelDescr = ModelDescr;
        yint depth = YSize(LayerArr);
        p->LayerArr.resize(depth);
        for (yint d = 0; d < depth; ++d) {
            LayerArr[d]->GetGradient(&p->LayerArr[d], &p->ModelDescr.Layers[d]);
        }
        p->MatrArr.resize(MP_MODEL_COUNT);
        FinalLayer->GetDeltaData(&p->MatrArr[MP_MODEL_FINAL]);
        Embedding->GetDeltaData(&p->MatrArr[MP_MODEL_EMBED]);
        ClearPodArray(&p->Bias, YSize(Bias));
    }

    TModelDescr GetModelDescr() override
    {
        return ModelDescr;
    }

    TIntrusivePtr<TParModelMatrix<TEmbedFloat>> GetEmbedding() override
    {
        return Embedding;
    }

    TIntrusivePtr<TParModelMatrix<TFastModelFloat>> GetFinalLayer() override
    {
        return FinalLayer;
    }

    const TAttentionParams &GetAttention(yint d) override
    {
        return *LayerArr[d];
    }

    const TVector<float> &GetBias() override
    {
        return Bias;
    }

    TModelMatrixScale *GetMatrixScale() override
    {
        return MatrixScale.Get();
    }

    NCuda::TCudaPOD<int> GetCurrentIteration() override
    {
        return MatrixAdd->GetCurrentIteration();
    }

    yint GetDeviceCount() override
    {
        return MatrixAdd->GetDeviceCount();
    }

    void StartIteration(const TTrainingStep &step, EAddToModel addToModel) override
    {
        MatrixAdd->StartIteration(step, addToModel);
    }

    void WaitCompute() override
    {
        MatrixAdd->Wait();
    }

    void WaitDelayedCompute() override
    {
        MatrixAdd->WaitDelayedCompute();
    }

    void ConvertMatrices() override
    {
        MatrixAdd->ConvertMatrices();
    }
};


TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelParams &params, IMMDeltaHookGen *deltaHookGen)
{
    return new TModel(deviceCount, params, deltaHookGen);
}

TIntrusivePtr<IModel> CreateModel(yint deviceCount, const TModelDescr &modelDescr, const TVector<float> &biasArr, IMMDeltaHookGen *deltaHookGen)
{
    TModelParams params; // TModel can be created directly from model descr
    InitModelZero(&params, modelDescr, biasArr);
    return new TModel(deviceCount, params, deltaHookGen);
}
