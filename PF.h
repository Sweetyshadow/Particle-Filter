//
// Created by Shadow on 2020/3/4.
//

#ifndef PF_PF_H
#define PF_PF_H


#include<vector>
#include<Eigen/Dense>
using namespace Eigen;
struct Particle {
    int id;
    VectorXd state;
    double weight;
    Particle(){};
    Particle(double w, int state_size):weight(w), state(VectorXd(state_size)) {}
};

class PF {
protected:
    int particle_num, state_size;
    double x_std, y_std;
    VectorXd real, state_u;
    std::vector<double> weights;
    double p_eff = 0;


public:
    std::vector<Particle> particles;

    PF() : particle_num(0){}
    ~PF() = default;

    void initialize(double x, double y, int particle_n, int state_n, double x_cov, double y_cov, double x_std, double y_std, double xr, double yr);

    void prediction(double (*px)(VectorXd), double (*py)(VectorXd));

    void update();

    void resample();
    double smse = 0;

};

// RPF(Regularized Particle Filter)
class RPF:public PF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void resample();
//    double ker(double )
};

#endif //PF_PF_H
