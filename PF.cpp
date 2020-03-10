//
// Created by Shadow on 2020/3/4.
//

#include "PF.h"
#include <random>
#include <cmath>
#include <iostream>
#include <Eigen/Cholesky>
#include <ctime>

double gaussian(double x, double y, double x_std, double y_std, double xu, double yu) {
    return exp(-(pow((x-xu), 2)/pow(x_std, 2) + pow((y-yu), 2)/pow(y_std, 2))/2);
}

void PF::initialize(double x, double y, int particle_n, int state_n, double x_cov, double y_cov, double x_s, double y_s, double xr, double yr) {
    this->particle_num = particle_n;
    this->particles.resize(particle_n);
    this->state_size = state_n;
    this->x_std = x_s;
    this->y_std = y_s;
    this->weights.resize(particle_n);
    this->real = VectorXd(state_n);
    std::normal_distribution<double> normal_x(x, x_cov);
    std::normal_distribution<double> normal_y(y, y_cov);
    srand(time(NULL));
    std::default_random_engine eng;
    eng.seed(rand());

    for(int i = 0; i < this->particle_num; ++i) {
        Particle p(1, this->state_size);
        this->weights[i] = 1;
        p.state(0) = normal_x(eng);
        p.state(1) = normal_y(eng);
//        std::cout << p.state(0) << " " << p.state(1) << std::endl;
        this->particles[i] = p;
    }
    real << xr, yr;
    this->is_initialized = true;

}

void PF::prediction(double (*px)(VectorXd), double (*py)(VectorXd)) {
    std::default_random_engine eng;
    std::normal_distribution<double> normal_x(0, x_std), normal_y(0, y_std);
    double x, y;

    for (int i = 0; i < this->particle_num; ++i) {
        particles[i].state(0) = px(particles[i].state) + normal_x(eng);
        particles[i].state(1) = py(particles[i].state) + normal_y(eng);
//        std::cout << particles[i].state[0] << particles[i].state[1] << std::endl;
    }
    VectorXd v(state_size);
    v << px(real), py(real);
    real = v;
}

void PF::update() {
    double sum{0};
    VectorXd U = VectorXd::Zero(state_size);
    for (int i = 0; i < this->particle_num; ++i) {
        this->particles[i].weight = gaussian(particles[i].state(0), particles[i].state(1), 1, 1, real(0), real(1));
        sum += this->particles[i].weight;
    }
    for (int i = 0; i < this->particle_num; ++i) {
        this->particles[i].weight /= sum;
        this->weights[i] = this->particles[i].weight;
        U += this->particles[i].weight * this->particles[i].state;
//        y_mean += this->particles[i].weight * this->particles[i].state(1);
    }
    this->state_u = U;
    smse += sqrt(pow(real(0)- state_u(0), 2) + pow(real(1)- state_u(1), 2));
//    std::cout.precision(8);
//    std::cout << this->real(0)<< "    " << this->real(1)<< "   " << state_u(0) << "    " << state_u(1) << "   " << sqrt(pow(real(0)- state_u(0), 2) + pow(real(1)- state_u(1), 2)) << "   " << this->p_eff << std::endl;
}

void PF::resample() {
    p_eff = 0;
    for(int i = 0; i < this->particle_num; ++i) p_eff += pow(weights[i], 2);
    p_eff = 1/p_eff;
    if (p_eff < 750) {
        std::vector<Particle> r_particle;
        r_particle.resize(this->particle_num);
        srand(time(NULL));
        std::default_random_engine eng;
        eng.seed(rand());
        std::discrete_distribution<int> discreteDistribution(this->weights.begin(), this->weights.end());
        for (int i = 0; i < this->particle_num; ++i)
            r_particle[i] = this->particles[discreteDistribution(eng)];
        this->particles = r_particle;
    }
}

void RPF::resample() {
    p_eff = 0;
    for(int i = 0; i < this->particle_num; ++i) p_eff += pow(weights[i], 2);
    p_eff = 1/p_eff;
    if (p_eff < 750) {
        MatrixXd cov = MatrixXd::Zero(state_size, state_size);
        for (int i = 0; i < particle_num; ++i)
            cov += weights[i] * (particles[i].state - state_u) * (particles[i].state - state_u).transpose();
        LLT<MatrixXd> llt(cov);
        MatrixXd L = llt.matrixL();
//        std::cout << L << std::endl;
        std::vector<Particle> r_particle;
        r_particle.resize(this->particle_num);
        srand(time(NULL));
        std::default_random_engine eng;
        eng.seed(rand());
        std::discrete_distribution<int> discreteDistribution(this->weights.begin(), this->weights.end());
        for (int i = 0; i < this->particle_num; ++i)
            r_particle[i] = this->particles[discreteDistribution(eng)];
        this->particles = r_particle;
        double s = state_size, p = particle_num;
        double A = pow(4/(s + 2), 1/(s + 4));
        double h = A * pow(p, -1/(s + 4));
//        std::cout<<h<<std::endl;
        std::normal_distribution<double> d{0, 1};
        int j;
        for (int i = 0; i < particle_num; ++i) {
            VectorXd v(state_size);
            v << d(eng), d(eng);
            v = L * v;
//            std::cout <<r_particle[i].state << std::endl;
            r_particle[i].state += h * v;
//            std::cout << h * v << std::endl << r_particle[i].state << std::endl << std::endl;
//            std::cout <<  v << h * v << std::endl << std::endl;
        }

        this->particles = r_particle;
    }
}
