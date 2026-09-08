#ifndef HPD_REGION_APPROXIMATOR
#define HPD_REGION_APPROXIMATOR

#include <assert.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

class Dirichlet3D
{
    protected:

        std::vector<double> v1{1.0, 0.0};
        std::vector<double> v2{0.5, sqrt(3)/2};
        std::vector<double> v3{0.0, 0.0};

        double mx_1{-sqrt(3)};
        double mx_2{sqrt(3)};
        double mx_3{0.0};
        double b_1{sqrt(3)};
        double b_2{0.0};
        double b_3{0.0};

    public:

        Dirichlet3D() {

        }

        std::vector<double> calcDirichlet3dMean (double a1, double a2, double a3) 
        {
            double a0{a1 + a2 + a3};
            std::vector<double> mean{a1, a2, a3};

            for (std::size_t ind{0}; ind < mean.size();  ++ind)
                mean[ind] /= a0;

            return mean;
        }

        std::vector<double> calcDirichlet3dVar (double a1, double a2, double a3)
        {
            double a0{a1 + a2 + a3};
            double denom{a0 * a0 * (a0 + 1)};
            std::vector<double> var{a1, a2, a3};

            for (std::size_t ind{0}; ind < var.size(); ++ind)
                var[ind] = (var[ind] * (a0 - var[ind])) / denom;

            return var;
        }

        std::vector<std::vector<double>> calcDirichlet3dCov (double a1, double a2, double a3)
        {
            double a0{a1 + a2 + a3};
            double denom{a0 * a0 * (a0 + 1)};
            std::vector<std::vector<double>> cov(3, std::vector<double>(3));

            cov[0][0] = (a1 * (a0 - a1)) / denom;
            cov[1][1] = (a2 * (a0 - a2)) / denom;
            cov[2][2] = (a3 * (a0 - a3)) / denom;

            cov[0][1] = (-a1 * a2) / denom;
            cov[0][2] = (-a1 * a3) / denom;
            cov[1][2] = (-a2 * a3) / denom;

            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            return cov;
        }

        double lineX1Cart (double x)
        {
            return mx_1 * x + b_1;
        }

        double lineX2Cart (double x)
        {
            return mx_2 * x + b_2;
        }

        double lineX3Cart (double x)
        {
            return mx_3 * x + b_3;
        }

        double lineX1CartInv (double y)
        {
            return (y - b_1) / mx_1;
        }

        double lineX2CartInv (double y)
        {
            return (y - b_2) / mx_2;
        }

        double lineX3CartInv (double y)
        {
            return (y - b_3) / mx_3;
        }

        std::vector<double> lineX1X1 (double x_1)
        {
            double x = v1[0] - x_1 * 0.5;
            double y = v1[1] + x_1 * (sqrt(3) / 2);
            std::vector<double> r{x, y};
            return r;
        }

        std::vector<double> lineX2X2 (double x_2)
        {
            double x = v2[0] - x_2 * 0.5;
            double y = v2[1] - x_2 * (sqrt(3) / 2);
            std::vector<double> r{x, y};
            return r;
        }

        std::vector<double> lineX3X3 (double x_3)
        {
            double x = v3[0] + x_3;
            double y = v3[1];
            std::vector<double> r{x, y};
            return r;
        }

        std::vector<double> calcMeanBaryCoords (double a1, double a2, double a3)
        {
            double a0{a1 + a2 + a3};
            std::vector<double> baryPair(2);

            baryPair[0] = (a1 + 2 * a3) / (2 * a0);
            baryPair[1] = (sqrt(3) / 2) * a1 / a0;

            return baryPair;
        }
};

#endif
