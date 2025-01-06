#include "stdafx.h"
#include "rope.h"

void FillRopeBuf(TArray2D<float> *pRopeBuf, yint width, yint maxLen)
{
    //float theta = 10000;
    float theta = 500000;
    TVector<float> freqArr;
    for (yint k = 0; k < width / 2; ++k) {
        freqArr.push_back(exp(-log(theta) * (k * 2. / width)));
    }
    TArray2D<float> &rope = *pRopeBuf;
    rope.SetSizes(width, maxLen);
    for (yint t = 0; t < maxLen; ++t) {
        for (yint base = 0; base < width; base += 2 * 32) {
            for (yint k = 0; k < 32; ++k) {
                float angle = t * freqArr[base / 2 + k];
                rope[t][base + k] = cos(angle);
                rope[t][base + k + 32] = sin(angle);
            }
        }
    }
}


// reference ropeBuf application
template <class TFunc>
void ReferenceApplyRope(yint t, yint dim, yint headSize, const TArray2D<float> &ropeBuf, float ropeRotateDir, TFunc getElem)
{
    for (yint offset = 0; offset < dim; offset += headSize) {
        for (yint blk = 0; blk < headSize; blk += 64) {
            for (yint k = 0; k < 32; ++k) {
                float &v0 = *getElem(offset + blk + k);
                float &v1 = *getElem(offset + blk + k + 32);
                float cosValue = ropeBuf[t][blk + k];
                float sinValue = ropeBuf[t][blk + k + 32] * ropeRotateDir;
                float r0 = v0 * cosValue - v1 * sinValue;
                float r1 = v0 * sinValue + v1 * cosValue;
                v0 = r0;
                v1 = r1;
            }
        }
    }
}

void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, yint t, TVector<float> *p)
{
    yint dim = YSize(*p);
    yint headSize = ropeBuf.GetXSize();
    ReferenceApplyRope(t, dim, headSize, ropeBuf, ropeRotateDir, [&](yint k) { return &(*p)[k]; });
}

void ApplyRope(const TArray2D<float> &ropeBuf, float ropeRotateDir, TArray2D<float> *p)
{
    yint dim = p->GetXSize();
    yint headSize = ropeBuf.GetXSize();
    yint len = p->GetYSize();
    Y_ASSERT(len <= ropeBuf.GetYSize());
    for (yint t = 0; t < len; ++t) {
        ReferenceApplyRope(t, dim, headSize, ropeBuf, ropeRotateDir, [&](yint k) { return &(*p)[t][k]; });
    }
}
