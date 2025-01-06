#include "stdafx.h"
#include "net_train.h"
#include "cpu_infer.h"
#include "mmlu_score.h"
#include "fed_sim.h"
#include <gpt/data/data.h>
#include <gpt/data/bpe.h>
#include <gpt/data/fragment_gen.h>
#include <gpt/att/sliding_window.h>
#include <gpt/compute/gpt_cpu.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/compute/gpt_cpu.h>
#include <gpt/compute/gpt_cuda.cuh>
#include <gpt/data_config/data_config.h>
#include <gpt/train_config/train_config.h>
#include <gpt/train_ctx/train_ctx.h>
#include <gpt/train_ctx/backprop.h>
#include <lib/cuda/cuda_init.h>
#include <lib/random/rand_utils.h>
#include <lib/hp_timer/hp_timer.h>
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data script
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static TString DATA_SCRIPT =
    " make_byte_tokenizer()"
    //" load_folder('code')"
    " load_folder('code')"
    ;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Train script
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static TString TRAIN_SCRIPT =
    " SAVE_MODEL = false"
    " EVAL_INTERVAL = 100"
    " TRAIN_CONFIG = 'b256f64'"
    " DROP_CONFIG = 'drop0.6ch1'"
    " MODEL_DIMS = 'e256h2d42'"
    " create_model(MPF_TAIL_LOSS)"
    " train()"
    ;

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef _win_
static TString WorkingFolder = "d:/";
#else
static TString WorkingFolder = "";
#endif


///////////////////////////////////////////////////////////////////////////////////////////////////
static TIntrusivePtr<IComputeContext> CreateTrainComputeContext(TIntrusivePtr<IModel> pModel, yint nodeCount)
{
    //TIntrusivePtr<IComputeContext> pCtx = NCPU_GPT::CreateContext(pModel, nodeCount);
    TIntrusivePtr<IComputeContext> pCtx = NCUDA_GPT::CreateContext(pModel, nodeCount);
    return pCtx;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute model params distribution
static void ComputeMatrixParamDistr(TModelParams *pParams)
{
    for (TModelParams::TAttentionMatrices &att : pParams->LayerArr) {
        TModelMatrix &matr = att.MatrArr[MP_ATT_V];
        const TArray2D<float> &data = matr.GetMatrix();
        yint xSize = matr.GetXSize();
        yint ySize = matr.GetYSize();
        double sum2 = 0;
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                sum2 += Sqr(data[y][x]);
            }
        }
        double sko = sqrt(sum2 / xSize / ySize);
        double count3 = 0;
        double count5 = 0;
        double count7 = 0;
        for (yint y = 0; y < ySize; ++y) {
            for (yint x = 0; x < xSize; ++x) {
                double val = fabs(data[y][x] / sko);
                if (val > 3) {
                    count3 += 1;
                }
                if (val > 5) {
                    count5 += 1;
                }
                if (val > 7) {
                    count7 += 1;
                }
            }
        }
        double scale = 100. / xSize / ySize;
        DebugPrintf("sko %g, 3sko %g%%, 5sko %g%%, 7sko %g%%\n", sko, count3 * scale, count5 * scale, count7 * scale);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
static void ComputeAverageModel(TModelParams *p, yint finishIter, yint iterInterval)
{
    TString pathTemplate = WorkingFolder + "eden_gpt_%.8gk.bin";
    //TString pathTemplate = WorkingFolder + "models/fed_small/model_%.8g.bin ";

    // model averaging boosts perf on test significantly
    int startIter = finishIter - iterInterval;
    double modelCount = 1;
    TModelParams &res = *p;
    Serialize(IO_READ, Sprintf(pathTemplate.c_str(), startIter / 1000.), res);
    TAllModelMatrices sum;
    GetMatrices(&sum, res);
    const int STEP = 1000;
    //const int STEP = 100;
    for (int iter = startIter + STEP; iter <= finishIter; iter += STEP) {
        TModelParams params;
        Serialize(IO_READ, Sprintf(pathTemplate.c_str(), iter / 1000.), params);
        TAllModelMatrices pp;
        GetMatrices(&pp, params);
        sum.AddScaled(pp, 1);
        modelCount += 1;
        printf(".");
    }
    printf("\n");
    sum.Scale(1 / modelCount);
    SetMatrices(&res, sum);
    //ComputeMatrixParamDistr(&startParams);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// compute score on test set
static void ComputeExactTest(TIntrusivePtr<IDataSource> data, const TModelParams &params)
{
    yint fragLen = params.ModelDescr.FragLen;
    //yint testBatchSize = BUFFER_LEN / GetNodeCount(fragLen);
    //yint testBatchSize = 4;
    yint testBatchSize = 1;

    TIntrusivePtr<IModel> pModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> pCtx = CreateTrainComputeContext(pModel, testBatchSize * fragLen);
    double sumTestErr = 0;
    double sumCount = 0;
    int rngSeed = 31331;
    for (yint iter = 1;; ++rngSeed, ++iter) {
        TVector<TFragment> batchArr;
        data->SampleFragments(IDataSource::TEST, rngSeed, testBatchSize, fragLen, &batchArr);
        float testErr = CalcModelErr(batchArr, pCtx.Get()) * data->GetStats().Compression;
        if (isnan(testErr)) {
            DebugPrintf("rseed %g, score is nan\n", rngSeed * 1.);
        }
        sumTestErr += testErr;
        sumCount += 1;
        if ((iter % 100) == 0) {
            DebugPrintf("iter %gk, avrg test score %g\n", iter / 1000., sumTestErr / sumCount); fflush(0);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// check if results are reproducible
template <class T>
double CalcDiff(const TVector<TVector<T>> &cpuPredArr, const TVector<TVector<T>> &gpuPredArr)
{
    double totalDiff = 0;
    for (yint t = 0; t < YSize(cpuPredArr); ++t) {
        for (yint k = 0; k < YSize(cpuPredArr[t]); ++k) {
            totalDiff += Sqr(cpuPredArr[t][k] - gpuPredArr[t][k]);
        }
    }
    return sqrt(totalDiff / YSize(cpuPredArr) / YSize(cpuPredArr[0]));
}


//static bool TestMatch(const TArray2D<float> &a, const TArray2D<float> &b)
//{
//    for (yint y = 0; y < a.GetYSize(); ++y) {
//        for (yint x = 0; x < a.GetXSize(); ++x) {
//            if (a[y][x] != b[y][x]) {
//                printf("%g != %g  (%x %x)\n", a[y][x], b[y][x], *(int*)&a[y][x], *(int*)&b[y][x]);
//                return false;
//            }
//        }
//    }
//    return true;
//}
//
//void TestReproducibility(const TTrainContext &trainCtx, IComputeContext *pCtx, TXRng &rng, const TVector<TFragment> &fragArr)
//{
//    const TDescentConfig &dc = trainCtx.GetDescentConfig();
//
//    TModelParams point1;
//    pCtx->GetParams(&point1);
//    pCtx->SetParams(point1);
//
//    TXRng chkRng = rng;
//    TVector<TNodeTarget> batchTarget;
//    MakeTrain(rng, fragArr, dc.TokenDrop, dc.ChannelDrop, pCtx, MAIN_DEVICE, &batchTarget);
//    pCtx->Backprop(dc.Step, GRADIENT_APPLY);
//
//    TModelParams point2;
//    pCtx->GetParams(&point2);
//
//    for (yint testId = 0; testId < 5; ++testId) {
//        pCtx->SetParams(point1);
//
//        pCtx->Backprop(dc.Step, GRADIENT_APPLY);
//
//        TModelParams chk;
//        pCtx->GetParams(&chk);
//
//        bool hasMismatch = false;
//        if (!TestMatch(chk.LabelEmbed.GetMatrix(), point2.LabelEmbed.GetMatrix())) {
//            printf("Label embed mismatch\n");
//            hasMismatch = true;
//        }
//        for (yint d = 0; d < YSize(point2.LayerArr); ++d) {
//            for (yint k = 0; k < YSize(point2.LayerArr[d]); ++k) {
//                const TModelParams::TAttentionMatrices &att1 = point2.LayerArr[d][k];
//                const TModelParams::TAttentionMatrices &att2 = chk.LayerArr[d][k];
//                if (!TestMatch(att1.QK, att2.QK)) {
//                    printf("Layer %g, att %g, QK mismatch\n", d * 1., k * 1.);
//                    hasMismatch = true;
//                }
//                if (!TestMatch(att1.QV, att2.QV)) {
//                    printf("Layer %g, att %g, QV mismatch\n", d * 1., k * 1.);
//                    hasMismatch = true;
//                }
//                if (!TestMatch(att1.V, att2.V)) {
//                    printf("Layer %g, att %g, V mismatch\n", d * 1., k * 1.);
//                    hasMismatch = true;
//                }
//                if (!TestMatch(att1.K, att2.K)) {
//                    printf("Layer %g, att %g, K mismatch\n", d * 1., k * 1.);
//                    hasMismatch = true;
//                }
//                if (!TestMatch(att1.Combiner, att2.Combiner)) {
//                    printf("Layer %g, att %g, Combiner mismatch\n", d * 1., k * 1.);
//                    hasMismatch = true;
//                }
//            }
//        }
//        if (hasMismatch) {
//            printf("attempt %g\n", testId + 1.);
//            while (hasMismatch) {
//                SchedYield();
//            }
//        }
//    }
//}


///////////////////////////////////////////////////////////////////////////////////////////////////
void CheckCpuGpuMatch(const TDescentConfig &dc, TIntrusivePtr<IDataSource> data)
{
    const yint CHECK_BATCH_SIZE = 1;
    //const yint CHECK_BATCH_SIZE = 32;
    const yint CHECK_FRAG_LEN = 64 - 1;
    const float CHECK_CHANNEL_DROP = 1;

    TXRng chkRng(1313);
    TModelParams params;
    yint vocabSize = data->GetStats().VocabSize;
    yint modelFlags = 0;
    //TString modelDescrStr = "e128h1d1w64";
    TString modelDescrStr = "e384h2d1w64";
    //TString modelDescrStr = "e512h2d6w64";
    //TString modelDescrStr = "e128d6w64";
    TModelDescr modelDescr;
    InitModelDescr(&modelDescr, modelDescrStr, ALIBI_V3, vocabSize, modelFlags);
    InitModel(&params, chkRng, modelDescr, COMBINER_INIT_RANDOM, data->GetStats().Bias);
    //Serialize(IO_READ, WorkingFolder + "eden_gpt_134k.bin", params);
    //params.ResetGrad();

    TIntrusivePtr<IModel> cpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> cpuCtx = NCPU_GPT::CreateContext(cpuModel, CHECK_BATCH_SIZE * dc.TrainFragLen);

    TIntrusivePtr<IModel> gpuModel = CreateModel(1, params);
    TIntrusivePtr<IComputeContext> gpuCtx = NCUDA_GPT::CreateContext(gpuModel, CHECK_BATCH_SIZE * dc.TrainFragLen);

    TVector<TFragment> fragArr;
    data->SampleFragments(IDataSource::TRAIN, 1313, CHECK_BATCH_SIZE, CHECK_FRAG_LEN, &fragArr);

    MakeTest(fragArr, cpuCtx.Get(), MAIN_DEVICE);
    MakeTest(fragArr, gpuCtx.Get(), MAIN_DEVICE);

    TVector<TVector<float>> cpuPredArr;
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);
    TVector<TVector<float>> gpuPredArr;
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    int t = 15;
    //int t = 0;
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", cpuPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n", CalcDiff(cpuPredArr, gpuPredArr) * 10000);

    TXRng cpuRng = chkRng;
    TXRng gpuRng = chkRng;
    TTrainingStep largeStep = dc.Step;
    largeStep.ScaleRate(10);
    MakeTrain(cpuRng, fragArr, dc.TokenDrop, CHECK_CHANNEL_DROP, cpuCtx.Get(), MAIN_DEVICE);
    cpuCtx->Backprop(largeStep, GRADIENT_APPLY);
    cpuCtx->ComputeFragmentPredictions(&cpuPredArr);

    MakeTrain(gpuRng, fragArr, dc.TokenDrop, CHECK_CHANNEL_DROP, gpuCtx.Get(), MAIN_DEVICE);
    gpuCtx->Backprop(largeStep, GRADIENT_APPLY);
    gpuCtx->ComputeFragmentPredictions(&gpuPredArr);

    DebugPrintf("\nAfter backprop\n");
    for (yint k = 0; k < 5; ++k) {
        DebugPrintf("%g - %g\n", cpuPredArr[t][k], gpuPredArr[t][k]);
    }
    DebugPrintf("\nDiff %g bp\n\n", CalcDiff(cpuPredArr, gpuPredArr) * 10000);
    __debugbreak();
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void TrainModel(yint startIteration, yint deviceCount, bool printIterTrainErr, const TTrainContext &trainCtx, TIntrusivePtr<TModelParamsHolder> pParams)
{
    const TDescentConfig &dc = trainCtx.GetDescentConfig();
    const TDeviceBatchConfig &dbc = trainCtx.GetDeviceBatchConfig();

#ifdef _MSC_VER
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(), 0xffff); // use P-cores, mask is cpu dependent
#endif

    // create model
    TIntrusivePtr<IModel> pModel = CreateModel(deviceCount, pParams->Params);
    pParams = 0;
    TIntrusivePtr<IComputeContext> pCtx = CreateTrainComputeContext(pModel, dbc.GetDeviceMaxNodeCount());

    //TOFStream fTrainLog(WorkingFolder + "train_log.txt");
    NHPTimer::STime tStart;
    NHPTimer::GetTime(&tStart);
    for (yint iter = startIteration; iter <= trainCtx.GetMaxIters(); ++iter) {
        if ((iter % trainCtx.GetEvalInterval()) == 0) {
            if (trainCtx.IsSaveModel()) {
                TModelParams params;
                pCtx->GetParams(&params);
                Serialize(IO_WRITE, Sprintf((WorkingFolder + "eden_gpt_%.8gk.bin").c_str(), iter / 1000.), params);
            }
            float trainErr = CalcModelErr(trainCtx.GetScoreTrainBatches(), pCtx.Get()) * trainCtx.GetCompression();
            float testErr = CalcModelErr(trainCtx.GetScoreTestBatches(), pCtx.Get()) * trainCtx.GetCompression();
            if (testErr != 0) {
                DebugPrintf("iter %.8gk, %g sec, train err %g, test err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr, testErr); fflush(0);
            } else {
                DebugPrintf("iter %.8gk, %g sec, train err %g\n", iter / 1000., NHPTimer::GetTimePassed(&tStart), trainErr); fflush(0);
            }
            //fTrainLog << trainErr << "\t" << testErr << Endl;
        }

        ui64 rngSeed = (iter + 0xbadf00d) * 0x39ef28172812ull;
        TVector<TFragment> fragArr;
        trainCtx.SampleTrainBatches(rngSeed, &fragArr);

        TXRng iterRng(iter);
        BackpropBatch(iterRng, dc, dbc, trainCtx.GetStep(iter), fragArr, pCtx);

        if (printIterTrainErr) {
            DebugPrintf("iter %g, train err %g\n", iter + 0., pCtx->GetAvrgTrainErr()); fflush(0);
        }

        //printf("Iter %.8gk\n", iter / 1000.);
        //TestReproducibility(trainCtx, pCtx.Get(), iterRng, fragArr);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
static void ReadNonEmptyLines(TVector<TString> *pRes, const TString &fName)
{
    TSeqReader fr(fName);
    Y_VERIFY(fr.IsValid() && "file not found");
    while (!fr.IsEof()) {
        TString sz = fr.ReadLine();
        if (!sz.empty()) {
            pRes->push_back(sz);
        }
    }
}


class TTrainScriptParser
{
    TTrainModelConfigParser TrainCfg;
    yint DeviceCount = 1;
    yint StartIteration = 0;
    bool SaveModel = true;
    yint MaxIters = 2000000;
    yint EvalInterval = 1000;
    yint EvalBatchCount = 20;
    bool PrintIterTrainErr = false;
    yint LimitNodeCount = 0;
    bool BiasReset = false;

public:
    TTrainScriptParser(yint limitNodeCount) : LimitNodeCount(limitNodeCount)
    {
        DeviceCount = GetCudaDeviceCount();
    }
    void ParseScript(const TVector<TConfigFile::TOp> &opArr, TIntrusivePtr<IDataSource> data, const TString &lmIndexDir)
    {
        Y_VERIFY(data.Get() != 0);
        for (yint ptr = 0; ptr < YSize(opArr); ++ptr) {
            const TConfigFile::TOp &op = opArr[ptr];
            if (op.Op == CFG_OP_ASSIGNMENT) {
                if (op.Dst == "MAX_ITERS") {
                    MaxIters = atof(op.Args[0].c_str());
                } else if (op.Dst == "DEVICE_COUNT") {
                    DeviceCount = atof(op.Args[0].c_str());
                    Y_VERIFY(DeviceCount >= 1 && DeviceCount < 100);
                } else if (op.Dst == "EVAL_INTERVAL") {
                    EvalInterval = atof(op.Args[0].c_str());
                } else if (op.Dst == "EVAL_BATCH_COUNT") {
                    EvalBatchCount = atof(op.Args[0].c_str());
                } else if (op.Dst == "SAVE_MODEL") {
                    SaveModel = IsYes(op.Args[0]);
                } else if (op.Dst == "PRINT_ITER_TRAIN_ERR") {
                    PrintIterTrainErr = IsYes(op.Args[0]);
                } else if (op.Dst == "BIAS_RESET") {
                    BiasReset = IsYes(op.Args[0]);
                } else if (TrainCfg.ParseScriptOp(op, data)) {
                    ;
                } else {
                    DebugPrintf("unknown config variable %s\n", op.Dst.c_str());
                }

            } else if (op.Op == CFG_OP_CALL) {
                if (op.Dst == "load_checkpoint") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    StartIteration = atoi(op.Args[0].c_str());
                    DebugPrintf("Load checkpoint %gk\n", StartIteration / 1000.);
                    TrainCfg.StartParams = new TModelParamsHolder();
                    Serialize(IO_READ, Sprintf((WorkingFolder + "eden_gpt_%.8gk.bin").c_str(), StartIteration / 1000.), TrainCfg.StartParams->Params);
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());

                    // process ops
                } else if (op.Dst == "train" || op.Dst == "net_train") {
                    Y_VERIFY(!TrainCfg.StartParams->Params.IsEmpty());
                    TDescentConfig dc = TrainCfg.MakeDescentConfig();
                    TTrainContext trainCtx(data, dc, DeviceCount, LimitNodeCount, SaveModel, MaxIters, EvalInterval);

                    DebugPrintf("%s %s %s 0x%x, size %gM\n",
                        GetModelDescrString(TrainCfg.StartParams->Params.GetModelDescr()).c_str(),
                        dc.GetTrainConfig().c_str(),
                        dc.GetDropConfig().c_str(),
                        (int)TrainCfg.StartParams->Params.ModelDescr.Flags,
                        CountModelSize(TrainCfg.StartParams->Params) / 1000000.);

                    // create batches for train & test score compute, can use different sizes
                    TDeviceBatchConfig dbc = trainCtx.GetDeviceBatchConfig();
                    trainCtx.MakeScoreBatches(EvalBatchCount, dbc.DeviceBatchSize, dbc.FragLen);

                    // keep train params
                    {
                        TModelParams &params = TrainCfg.StartParams->Params;
                        params.ModelDescr.FragLen = dc.TrainFragLen;
                        if (BiasReset) {
                            ClearPodArray(&params.Bias, YSize(params.Bias));
                        }
                    }

                    if (op.Dst == "train") {
                        TrainModel(StartIteration, DeviceCount, PrintIterTrainErr, trainCtx, TrainCfg.StartParams.Release());
                        //TestGradient(StartIteration, DeviceCount, PrintIterTrainErr, trainCtx, TrainCfg.StartParams.Release());
                    } else if (op.Dst == "net_train") {
                        Y_VERIFY(YSize(op.Args) == 1);
                        TVector<TString> workerArr;
                        ReadNonEmptyLines(&workerArr, op.Args[0]);
                        NNetTrain::RunMaster(StartIteration, DeviceCount, workerArr, trainCtx, TrainCfg.StartParams.Release());
                    } else {
                        Y_ASSERT(0);
                    }

                } else if (op.Dst == "compute_exact_test") {
                    TModelParams params;
                    if (op.Args.empty()) {
                        params = TrainCfg.StartParams->Params;
                    } else {
                        yint finishIter = atoi(op.Args[0].c_str());
                        yint iterInterval = YSize(op.Args) > 1 ? atoi(op.Args[1].c_str()) : 0;
                        ComputeAverageModel(&params, finishIter, iterInterval);
                    }
                    ComputeExactTest(data, params);

                } else if (op.Dst == "compute_choice_score") {
                    Y_VERIFY(YSize(op.Args) == 1);
                    yint fragmentStartToken = data->GetStats().FragmentStartToken;
                    yint docStartToken = data->GetStats().DocStartToken;
                    ComputeChoiceScore(TrainCfg.StartParams->Params, op.Args[0], docStartToken, fragmentStartToken, lmIndexDir);

                } else if (op.Dst == "check_cpu_gpu_match") {
                    TDescentConfig dc = TrainCfg.MakeDescentConfig();
                    CheckCpuGpuMatch(dc, data);

                } else if (TrainCfg.ParseScriptOp(op, data)) {

                } else {
                    DebugPrintf("unknown function %s\n", op.Dst.c_str());
                    abort();
                }
            }
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
extern yint MatrixAddWorkerThreadCount;

void TestMatMulFp16(bool isHalfAccum);
void TestMatMulInt8();
void TestMatMulFp8();
void TestAttFp8();
void TestAttGradQVfp8();
void TestCudaSort();
void TestLMatch();
//void Repack();


int main(int argc, char **argv)
{
    //TestMatMulFp16(true); // fp16half
    //TestMatMulFp16(false);
    //TestMatMulInt8();
    //TestMatMulFp8();
    //TestAttFp8();
    //TestAttGradQVfp8();
    //TestCudaSort();
    //Repack();
    //GenerateArithmetic();
    //GenerateArithmetic97();
    //NCPUInfer::Check();
    //NFedSim::Run();
    //TestLMatch();
    //return 0;

    yint limitNodeCount = 24 * 1024;
    TOpt cmdline("d:s:n:w:t:", argc, argv);
    TString workerPort;
    for (const TOpt::TParam &param : cmdline.Params) {
        if (param.Name == "s") {
            //DebugPrintf("Train script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            TRAIN_SCRIPT = cfg.data();
        } else if (param.Name == "d") {
            //DebugPrintf("Datasource script %s\n", param.Args[0].c_str());
            TVector<char> cfg;
            Y_VERIFY(ReadWholeFile(param.Args[0], &cfg));
            DATA_SCRIPT = cfg.data();
        } else if (param.Name == "n") {
            limitNodeCount = atoi(param.Args[0].c_str());
        } else if (param.Name == "w") {
            workerPort = param.Args[0];
        } else if (param.Name == "t") {
            MatrixAddWorkerThreadCount = atoi(param.Args[0].c_str());
        }
    }
    if (!workerPort.empty()) {
        NNetTrain::RunWorker(atoi(workerPort.c_str()));
        return 0;
    }

    // load data
    TString lmIndexDir;
    TIntrusivePtr<IDataSource> data = CreateDataSource(DATA_SCRIPT, &lmIndexDir);
    if (data.Get() == 0) {
        DebugPrintf("no dataset no train\n");
        return 0;
    }

    // train script
    TConfigFile trainCfg;
    ParseConfig(&trainCfg, TRAIN_SCRIPT);
    TTrainScriptParser trainScript(limitNodeCount);
    trainScript.ParseScript(trainCfg.OpArr, data, lmIndexDir);

    return 0;
}
