#pragma once

struct TPoissonCalc
{
    TVector<double> FacLogs;

    TPoissonCalc()
    {
        FacLogs.push_back(0);
        for (yint i = 1; i < 1000000; ++i) {
            FacLogs.push_back(FacLogs.back() + log(i + 0.));
        }
    }
    double CalcProbLog(double lambda, yint k) const
    {
        //return log(lambda) * k - lambda - FacLogs[k];
        return lambda * k - exp(lambda) - FacLogs[k];
    }
    double CalcGrad(double lambda, yint k) const
    {
        //return k / lambda - 1;
        return k - exp(lambda);
    }
};
