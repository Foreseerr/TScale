#pragma once


struct TTrainingStep
{
    float Rate = 0.01f;
    float L2Reg = 0;
    float Beta1 = 0;
    float Weight0 = 1;
    float Weight1 = 0;
    float DispDecay = 0.9f;

    TTrainingStep() {}
    TTrainingStep(float rate) : Rate(rate) {}
    float GetShrinkMult() const
    {
        return 1 - Rate * L2Reg;
    }
    void ScaleRate(float x)
    {
        Rate *= x;
    }
};
