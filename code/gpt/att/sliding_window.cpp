#include "stdafx.h"
#include "sliding_window.h"
#include <gpt/data/data.h>


const yint HASH_VOCAB_SIZE_LN = 11;
const yint HASH_VOCAB_SIZE = 1ull << HASH_VOCAB_SIZE_LN;
const yint HASH_VOCAB_COUNT = 3;


static void AddToken(bool hashedVocab, bool needNoiseLabels, TVector<TLabelIndex> *p, yint token)
{
    yint base = needNoiseLabels ? NOISE_LABELS_COUNT : 0;
    if (hashedVocab) {
        ui64 hh = 0x9ae16a3b2f90404fULL;
        for (yint k = 0; k < HASH_VOCAB_COUNT; ++k) {
            hh = (hh + token) * 0xc949d7c7509e6557ULL;
            yint token_hash = hh >> (64 - HASH_VOCAB_SIZE_LN);
            p->push_back(base + token_hash);
        }
    } else {
        p->push_back(base + token);
    }
}


static void AddNoiseTokens(bool needNoiseLabels, yint lossType, TXRng &rng, TVector<TLabelIndex> *p)
{
    if (needNoiseLabels && lossType == ATT_GRAPH_TRAIN_LOSS) {
        p->push_back(rng.Uniform(NOISE_LABELS_COUNT));
        p->push_back(LABEL_NEGATIVE + rng.Uniform(NOISE_LABELS_COUNT));
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// attention graph
static bool IsHashedVocab(const TModelDescr &modelDescr)
{
    if (modelDescr.HasFlag(MPF_HASHED_EMBED)) {
        return true;
    } else {
        return false;
    }
}

static yint GetLabelCount(const TModelDescr &modelDescr)
{
    yint res = 0;
    if (!modelDescr.HasFlag(MPF_DISABLE_NOISE_LABELS)) {
        res += NOISE_LABELS_COUNT;
    }
    if (IsHashedVocab(modelDescr)) {
        res += HASH_VOCAB_SIZE;
    } else if (modelDescr.HasFlag(MPF_MLM_BERT)) {
        yint wideLimitWindow = modelDescr.GetWideLimitWindow();
        res += wideLimitWindow + (modelDescr.VocabSize + 1);
    } else {
        if (modelDescr.HasFlag(MPF_ABS_POSITIONS)) {
            yint wideLimitWindow = modelDescr.GetWideLimitWindow();
            res += wideLimitWindow;
        }
        res += (modelDescr.VocabSize + 1); // add input tokens unconditionally
        if (modelDescr.HasFlag(MPF_PPM)) {
            res += (modelDescr.VocabSize + 1);
        }
        if (modelDescr.HasFlag(MPF_LMATCH)) {
            res += (modelDescr.VocabSize + 1);
        }
    }
    return res;
}

void InitModelDescr(TModelDescr *pRes, const TString &modelDescrStr, EAlibi alibi, yint vocabSize, ui64 flags)
{
    InitModelDescr(pRes, modelDescrStr, alibi, vocabSize, 0, flags);
    pRes->LabelCount = GetLabelCount(*pRes);
}

static void AddAttSpans(yint docStart, yint nodeId, yint limitWindow, yint finishOffset, TVector<TVector<TAttentionSpan>> *pAtt)
{
    yint attStart = Max<yint>(docStart, nodeId - limitWindow);
    yint attFinish = nodeId - finishOffset;
    if (attFinish >= attStart) {
        (*pAtt)[nodeId].push_back(TAttentionSpan(attStart, attFinish));
    }
}


// process single fragment
static void GenerateAttentionGraph(
    const TModelDescr &modelDescr, TXRng &rng, float tokenDrop,
    const TFragment &frag, yint lossType,
    TVector<TVector<TLabelIndex>> *pLabels,
    TVector<TVector<TVector<TAttentionSpan>>> *pAttArr,
    TVector<TNodeTarget> *pTargetArr, TVector<yint> *pNodeToSampleIndex)
{
    bool isHashedVocab = IsHashedVocab(modelDescr);
    bool needNoiseLabels = !modelDescr.HasFlag(MPF_DISABLE_NOISE_LABELS);
    yint len = YSize(frag.Text);
    pLabels->resize(len);
    yint attentionWidthCount = modelDescr.GetAttentionWidthCount();
    pAttArr->resize(attentionWidthCount);
    for (yint wa = 0; wa < attentionWidthCount; ++wa) {
        (*pAttArr)[wa].resize(len);
    }
    pNodeToSampleIndex->resize(len);

    if (modelDescr.HasFlag(MPF_MLM_BERT)) {
        // samples
        yint wideLimitWindow = modelDescr.GetWideLimitWindow();
        Y_VERIFY(len <= wideLimitWindow && "absolute position encoding is impossible, sequence too long");
        yint docStart = 0;
        for (yint t = 0; t < len; ++t) {
            yint nodeId = t;

            AddNoiseTokens(needNoiseLabels, lossType, rng, &(*pLabels)[nodeId]);
            yint lblBase = 0;

            // position
            AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + t);

            // add labels
            lblBase += wideLimitWindow;
            if (frag.Text[t] != UNDEFINED_TOKEN) {
                AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 1 + frag.Text[t]);
            } else {
                AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 0);
            }

            // target
            if (frag.Target[t] != UNDEFINED_TOKEN) {
                pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
            }

            // add attention spans, same for all widths
            for (yint wa = 0; wa < attentionWidthCount; ++wa) {
                (*pAttArr)[wa][nodeId].push_back(TAttentionSpan(0, len - 1));
            }

            (*pNodeToSampleIndex)[nodeId] = frag.Offset + t;
        }

    } else {
        // samples
        yint docStart = 0;
        for (yint t = 0; t < len; ++t) {
            yint nodeId = t;

            // detect document start and limit attention to the document
            if (modelDescr.HasFlag(MPF_USE_DOC_START_TOKEN)) {
                if (t > 0 && frag.Text[t] == modelDescr.DocStartToken) {
                    docStart = nodeId;
                }
            }
            if (modelDescr.HasFlag(MPF_GROK_BINARY_OP)) {
                if (t > 0 && frag.Text[t] == 0) {
                    docStart = nodeId;
                }
            }

            // noise tokens
            AddNoiseTokens(needNoiseLabels, lossType, rng, &(*pLabels)[nodeId]);
            yint lblBase = 0;

            // position
            if (modelDescr.HasFlag(MPF_ABS_POSITIONS)) {
                yint wideLimitWindow = modelDescr.GetWideLimitWindow();
                Y_VERIFY(len <= wideLimitWindow && "absolute position encoding is impossible, sequence too long");
                AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + t);
                lblBase += wideLimitWindow;
            }

            // input token
            if (rng.GenRandReal3() <= tokenDrop) {
                // make gaps to fill by training
                AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 1 + frag.Text[t]);
            } else {
                AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 0);
            }
            lblBase += 1 + modelDescr.VocabSize;

            // ppm features
            if (modelDescr.HasFlag(MPF_PPM)) {
                if (frag.PPM[t] != UNDEFINED_TOKEN) {
                    if (rng.GenRandReal3() <= tokenDrop) {
                        AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 1 + frag.PPM[t]);
                    } else {
                        AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 0); // skip token
                    }
                }
                lblBase += 1 + modelDescr.VocabSize;
            }
            if (modelDescr.HasFlag(MPF_LMATCH)) {
                if (frag.LMatch[t] != UNDEFINED_TOKEN) {
                    if (rng.GenRandReal3() <= tokenDrop) {
                        AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 1 + frag.LMatch[t]);
                    } else {
                        AddToken(isHashedVocab, needNoiseLabels, &(*pLabels)[nodeId], lblBase + 0); // skip token
                    }
                }
                lblBase += 1 + modelDescr.VocabSize;
            }

            // add attention span
            if (modelDescr.HasFlag(MPF_ABS_POSITIONS)) {
                Y_VERIFY(attentionWidthCount == 1);
                TVector<TVector<TAttentionSpan>> &attArr = (*pAttArr)[0];
                attArr[nodeId].push_back(TAttentionSpan(docStart, nodeId)); // include self attention
            } else {
                for (yint wa = 0; wa < attentionWidthCount; ++wa) {
                    yint limitWindow = modelDescr.AttentionWidthArr[wa];
                    if (limitWindow == 0) {
                        TVector<TVector<TAttentionSpan>> &selfAtt = (*pAttArr)[wa];
                        //selfAtt[nodeId].push_back(TAttentionSpan(0, 0)); // add attention to start token
                        selfAtt[nodeId].push_back(TAttentionSpan(nodeId, nodeId)); // self attention
                    } else {
                        yint finishOffset = 1;
                        //yint finishOffset = (limitWindow > 1) ? 1 : 0;
                        AddAttSpans(docStart, nodeId, limitWindow, finishOffset, &(*pAttArr)[wa]);
                    }
                }
            }

            // add loss
            if (modelDescr.HasFlag(MPF_GROK_BINARY_OP)) {
                // special loss, target only binary op result, 0 is special token for this dataset meaning start of sample
                if (docStart > 0 && t + 1 < YSize(frag.Target) && frag.Target[t + 1] == 0) {
                    pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
                }
            } else if (!frag.Target.empty()) {
                if (frag.Target[t] != UNDEFINED_TOKEN) {
                    bool isTestLoss = true;
                    if (modelDescr.HasFlag(MPF_TAIL_LOSS)) {
                        isTestLoss = (t >= 0.5 * len); // account second half in reported loss
                    }
                    if (lossType == ATT_GRAPH_TRAIN_LOSS || (lossType == ATT_GRAPH_TEST_LOSS && isTestLoss)) {
                        pTargetArr->push_back(TNodeTarget(nodeId, frag.Target[t]));
                    }
                }
            }

            (*pNodeToSampleIndex)[nodeId] = frag.Offset + t;
        }
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////////
// make train/test contexts

void InitLabelData(const TModelDescr &modelDescr, TXRng &rng, float tokenDrop,
    const TVector<TFragment> &fragArr, yint lossType,
    TNodesBatch *pNodes)
{
    pNodes->Init(modelDescr.GetAttentionWidthCount());

    for (const TFragment &frag : fragArr) {
        yint ptr = pNodes->GetNodeCount();

        TVector<TVector<TLabelIndex>> fragLabels;
        TVector<TVector<TVector<TAttentionSpan>>> fragAttSpansArr;
        TVector<TNodeTarget> fragTargets;
        TVector<yint> fragNodeToSampleIndex;
        GenerateAttentionGraph(modelDescr, rng, tokenDrop,
            frag, lossType,
            &fragLabels, &fragAttSpansArr, &fragTargets, &fragNodeToSampleIndex);

        yint nodeCount = YSize(fragLabels);
        for (yint t = 0; t < nodeCount; ++t) {
            TVector<TVector<TAttentionSpan>> rrArr;
            rrArr.resize(YSize(fragAttSpansArr));
            for (yint wa = 0; wa < YSize(fragAttSpansArr); ++wa) {
                Y_ASSERT(nodeCount == YSize(fragAttSpansArr[wa]));
                TVector<TAttentionSpan> rr = fragAttSpansArr[wa][t];
                for (TAttentionSpan &span : rr) {
                    span.Shift(ptr);
                }
                rrArr[wa] = rr;
            }
            pNodes->AddSample(fragNodeToSampleIndex[t], fragLabels[t], rrArr);
        }

        for (TNodeTarget nt : fragTargets) {
            nt.Node += ptr;
            pNodes->Target.push_back(nt);
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// results are discarded, so we don't care about race conditions
TXRng NopRng;
