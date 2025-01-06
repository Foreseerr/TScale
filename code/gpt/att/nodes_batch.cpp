#include "stdafx.h"
#include "nodes_batch.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
void TNodesBatch::Init(yint attentionWidthCount)
{
    SampleIndex.resize(0);
    LabelArr.resize(0);
    LabelPtr.resize(0);
    LabelPtr.push_back(0);
    Target.resize(0);
    AttArr.resize(attentionWidthCount);
    for (TAttentionInfo &att : AttArr) {
        att.Init();
    }
}


void TNodesBatch::AddSample(int idx, const TVector<TLabelIndex> &labels, const TVector<TVector<TAttentionSpan>> &attSpansArr)
{
    SampleIndex.push_back(idx);
    LabelArr.insert(LabelArr.end(), labels.begin(), labels.end());
    LabelPtr.push_back(YSize(LabelArr));
    Y_VERIFY(YSize(AttArr) == YSize(attSpansArr));
    for (yint k = 0; k < YSize(AttArr); ++k) {
        AttArr[k].AddSpans(attSpansArr[k]);
        AttArr[k].AddSample();
    }
}


