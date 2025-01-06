#include "stdafx.h"
#include "ppm_window.h"
#include <lib/hp_timer/hp_timer.h>


void ComputeWindowPPM(const TVector<TBPEToken> &text, TVector<TBPEToken> *pResPPM, yint docStartToken)
{
    TWindowPPMIndex ppm;
    yint len = YSize(text);
    pResPPM->resize(len);
    for (yint t = 0; t < YSize(text); ++t) {
        TBPEToken token = text[t];
        if (token == docStartToken) {
            (*pResPPM)[t] = UNDEFINED_TOKEN;
            ppm.Clear();
        } else {
            yint bestLen = 0;
            yint bestPos = 0;
            ppm.IndexPos(text, t, &bestLen, &bestPos);
            if (bestLen > 0) {
                (*pResPPM)[t] = text[bestPos + 1];
            } else {
                (*pResPPM)[t] = UNDEFINED_TOKEN;
            }
        }
    }
}
