#include <cmath>
#include <iostream>
#include "PF.h"
double fx(VectorXd v) {
//    return std::cos(x) + std::sin(y);
    return v(0) + 1;
}

double fy(VectorXd v) {
//    return std::cos(y) + pow(x, 2) - 1;
//    return std::cos(v(0)) + 1 / (v(0) + 1) + log(abs(v(0)));
    return 0.5*v(1) + 2.5*v(1)/(1+pow(v(1), 2)) + 8*cos(1.2*v(1));
}


int main() {

    double sss = 0;
    PF particle_filter;
    RPF regularized_particle_filter;
    double (* px)(VectorXd), (* py)(VectorXd);
    px = fx;
    py = fy;
    for(int j = 0; j < 5; ++j) {
        std::cout << j << std::endl;
        particle_filter.initialize(1, 1, 1000, 2, 10, 10, 0.1, 0.1, 1, 1);
        for (int i = 0; i < 10000; ++i) {
            particle_filter.prediction(px, py);
            particle_filter.update();
            particle_filter.resample();
        }
        sss += particle_filter.smse;
    }
    std::cout << sss / 50000 << std:: endl;
    sss = 0;
    for(int j = 0; j < 5; ++j) {
        std::cout << j << std::endl;
        regularized_particle_filter.initialize(1, 1, 1000, 2, 10, 10, 0.1, 0.1, 1, 1);
        for (int i = 0; i < 10000; ++i) {
            regularized_particle_filter.prediction(px, py);
            regularized_particle_filter.update();
            regularized_particle_filter.resample();
        }
        sss += regularized_particle_filter.smse;
    }
    std::cout << sss / 50000 << std:: endl;
    return 0;

}
