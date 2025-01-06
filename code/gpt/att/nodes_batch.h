#pragma once
#include "att.h"

typedef ui32 TLabelIndex;
const TLabelIndex LABEL_INVALID_INDEX = 0xffffffff;
const TLabelIndex LABEL_NEGATIVE = 0x80000000;
const TLabelIndex LABEL_MASK = 0x7fffffff;


struct TNodeTarget
{
    yint Node = 0;
    yint TargetId = 0;

    TNodeTarget() {}
    TNodeTarget(yint nodeId, yint targetId) : Node(nodeId), TargetId(targetId) {}
};

inline bool operator==(const TNodeTarget &a, const TNodeTarget &b)
{
    return a.Node == b.Node && a.TargetId == b.TargetId;
}


struct TNodesBatch
{
    TVector<TLabelIndex> LabelArr;
    TVector<ui32> LabelPtr;
    TVector<int> SampleIndex;
    TVector<TNodeTarget> Target;
    TVector<TAttentionInfo> AttArr;

    void Init(yint attentionWidthCount);
    void AddSample(int idx, const TVector<TLabelIndex> &labels, const TVector<TVector<TAttentionSpan>> &attSpansArr);
    yint GetNodeCount() const { return YSize(SampleIndex); }
};

