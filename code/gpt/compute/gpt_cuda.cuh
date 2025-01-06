#pragma once
#include "model.h"

namespace NCUDA_GPT
{
TIntrusivePtr<IComputeContext> CreateContext(TIntrusivePtr<IModel> pModel, yint nodeCount);
}
