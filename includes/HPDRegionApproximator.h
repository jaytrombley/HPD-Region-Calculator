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

        std::vector<double> v1 { 1.0, 0.0 };
        std::vector<double> v2 { 0.5, sqrt(3)/2 };
        std::vector<double> v3 { 0.0, 0.0 };

        double mx_1 { -sqrt(3) };
        double mx_2 { sqrt(3) };
        double mx_3 { 0.0 };
        double b_1 { sqrt(3) };
        double b_2 { 0.0 };
        double b_3 { 0.0 };

    public:

        Dirichlet3D() {

        }
};

#endif
