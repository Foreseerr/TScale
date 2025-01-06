#pragma once
#include <gpt/data/dataset_builder.h>
#include <lib/config/cfg_file.h>
#include <lib/config/config.h>


///////////////////////////////////////////////////////////////////////////////////////////////////
class TDataSourceConfigParser
{
    TTokenizer Tokenizer;
    TIntrusivePtr<TDatasetBuilder> DataBuild;
    bool UsePPM = false;
    bool UseLMatch = false;
    float TestFraction = 0.05f;
    yint VocabSize = 0;
    yint DocStartToken = -1;
    yint FragmentStartToken = -1;
    TString LMatchIndexDir;
    TIntrusivePtr<IDataSource> Dataset;

private:
    void LoadTokenizerParams()
    {
        VocabSize = Tokenizer.GetVocabSize();
        DocStartToken = Tokenizer.HasDocStartToken() ? Tokenizer.GetDocStartToken() : -1;
        FragmentStartToken = Tokenizer.GetFragmentStartToken();
    }

    void CreateDatasetBuilder()
    {
        if (!Tokenizer.IsEmpty()) {
            Y_VERIFY(VocabSize == Tokenizer.GetVocabSize());
            Y_VERIFY(FragmentStartToken == Tokenizer.GetFragmentStartToken());
            if (Tokenizer.HasDocStartToken()) {
                Y_VERIFY(DocStartToken == Tokenizer.GetDocStartToken());
            } else {
                Y_VERIFY(DocStartToken == -1);
            }
        }
        Y_VERIFY(VocabSize > 0);
        if (DataBuild.Get() == 0) {
            DataBuild = new TDatasetBuilder(UsePPM, UseLMatch, VocabSize, DocStartToken, FragmentStartToken);
        }
    }

    void MakeDataset()
    {
        if (DataBuild.Get()) {
            Dataset = DataBuild->MakeDataset(LMatchIndexDir);
            DataBuild = 0;
            const IDataSource::TDataStats &stats = Dataset->GetStats();
            Y_VERIFY(stats.VocabSize == VocabSize);
            Y_VERIFY(stats.DocStartToken == DocStartToken);
        }
    }

public:
    void ParseScript(const TVector<TConfigFile::TOp> &opArr);
    TIntrusivePtr<IDataSource> GetDataset() { return Dataset; }
    TString GetLMatchIndexDir() const { return LMatchIndexDir; }
};


inline TIntrusivePtr<IDataSource> CreateDataSource(const TString &dataScript, TString *pLMatchIndexDir)
{
    TConfigFile cfg;
    ParseConfig(&cfg, dataScript);
    TDataSourceConfigParser dataCfg;
    dataCfg.ParseScript(cfg.OpArr);
    *pLMatchIndexDir = dataCfg.GetLMatchIndexDir(); // need better way to provide access to lmatch index
    return dataCfg.GetDataset();
}
