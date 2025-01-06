#pragma once
#include <gpt/model_params/model_params.h>


class TLMatchSearch;
void ComputeChoiceScore(const TModelParams &params, const TString &queryFile, yint docStartToken, yint fragmentStartToken, const TString &lmIndexDir);
