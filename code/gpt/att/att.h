#pragma once


struct TAttentionSpan
{
    // inclusive borders
    int Start = 0;
    int Finish = 0;

    TAttentionSpan() {}
    TAttentionSpan(int start, int finish) : Start(start), Finish(finish) {}
    void Shift(int delta)
    {
        Y_ASSERT(Finish >= Start);
        Start += delta;
        Finish += delta;
    }
};

struct TAttentionInfo
{
    TVector<TAttentionSpan> Spans;
    TVector<int> SpanPtr;

    void Init()
    {
        Spans.resize(0);
        SpanPtr.resize(1);
        SpanPtr[0] = 0;
    }
    yint GetSampleCount() const { return YSize(SpanPtr) - 1; }
    bool IsEmpty() const { return YSize(SpanPtr) <= 1; }
    void AddSpan(const TAttentionSpan &span)
    {
        Spans.push_back(span);
    }
    void AddSpans(const TVector<TAttentionSpan> &vec)
    {
        for (const TAttentionSpan &span : vec) {
            AddSpan(span);
        }
    }
    void AddSample()
    {
        SpanPtr.push_back(YSize(Spans));
    }
};


void SortAttentionSpans(TAttentionInfo *p);
TAttentionInfo TransposeAttention(const TAttentionInfo &att);


///////////////////////////////////////////////////////////////////////////////////////////////////
// grouped attention
template <int N>
struct TAttentionSpanGroup
{
    int Start = 0;
    int Finish = 0;
    // inclusive borders
    int SpanStart[N];
    int SpanFinish[N];
};


template <int N>
struct TAttentionInfoGrouped
{
    TVector<TAttentionSpanGroup<N>> SpanGroups;
    TVector<int> SpanGroupPtr;

    void Init()
    {
        SpanGroups.resize(0);
        SpanGroupPtr.resize(1);
        SpanGroupPtr[0] = 0;
    }
    void AddSpanGroups(const TVector<TAttentionSpanGroup<N>> &arr)
    {
        for (const TAttentionSpanGroup<N> &gg : arr) {
            SpanGroups.push_back(gg);
        }
        SpanGroupPtr.push_back(YSize(SpanGroups));
    }
    void AddEmptySpanGroup()
    {
        SpanGroupPtr.push_back(YSize(SpanGroups));
    }
    yint GetGroupCount() const { return YSize(SpanGroupPtr) - 1; }
};


// spans are aligned to TO_ALIGN
template <int N, int TO_ALIGN>
void GroupAttention(const TAttentionInfo &attSrc, TAttentionInfoGrouped<N> *p)
{
    const int FAR_RIGHT = 1000000000;

    if (attSrc.IsEmpty()) {
        p->Init();
        return;
    }
    TAttentionInfo att = attSrc;
    SortAttentionSpans(&att);

    yint sampleCount = att.GetSampleCount();
    yint groupCount = (sampleCount + N - 1) / N;
    p->Init();
    for (yint g = 0; g < groupCount; ++g) {
        TVector<TAttentionSpanGroup<N>> ggArr;
        TVector<yint> ptrArr;
        TVector<yint> ptrFinArr;
        for (yint from = g * N; from < (g + 1) * N; ++from) {
            if (from < sampleCount) {
                ptrArr.push_back(att.SpanPtr[from]);
                ptrFinArr.push_back(att.SpanPtr[from + 1]);
            } else {
                ptrArr.push_back(0);
                ptrFinArr.push_back(0);
            }
        }
        for (;;) {
            TAttentionSpanGroup<N> gg;
            for (yint k = 0; k < N; ++k) {
                gg.SpanStart[k] = FAR_RIGHT;
                gg.SpanFinish[k] = FAR_RIGHT - 1;
            }
            // select leftmost span
            int start = FAR_RIGHT;
            for (yint k = 0; k < N; ++k) {
                if (ptrArr[k] != ptrFinArr[k]) {
                    const TAttentionSpan &span = att.Spans[ptrArr[k]];
                    Y_ASSERT(span.Finish >= span.Start);
                    start = Min(start, span.Start);
                }
            }
            // no more spans in the group
            if (start == FAR_RIGHT) {
                break;
            }
            // decide which spans we take into group
            TVector<bool> taken;
            taken.resize(N, false);
            int finish = start;
            for (;;) {
                bool hasAdded = false;
                for (yint k = 0; k < N; ++k) {
                    if (!taken[k] && ptrArr[k] != ptrFinArr[k]) {
                        const TAttentionSpan &span = att.Spans[ptrArr[k]];
                        if (span.Start / TO_ALIGN >= start / TO_ALIGN || span.Start / TO_ALIGN <= finish / TO_ALIGN) {
                            finish = Max(finish, span.Finish);
                            taken[k] = true;
                            hasAdded = true;
                        }
                    }
                }
                if (!hasAdded) {
                    break;
                }
            }
            // from and add group
            gg.Start = FAR_RIGHT;
            gg.Finish = 0;
            for (yint k = 0; k < N; ++k) {
                if (taken[k]) {
                    const TAttentionSpan &span = att.Spans[ptrArr[k]];
                    gg.SpanStart[k] = span.Start;
                    gg.SpanFinish[k] = span.Finish;
                    gg.Start = Min(gg.Start, span.Start);
                    gg.Finish = Max(gg.Finish, span.Finish);
                    ++ptrArr[k];
                }
            }
            gg.Start = (gg.Start / TO_ALIGN) * TO_ALIGN;
            gg.Finish = (gg.Finish / TO_ALIGN) * TO_ALIGN;
            ggArr.push_back(gg);
        }
        p->AddSpanGroups(ggArr);
    }
}

