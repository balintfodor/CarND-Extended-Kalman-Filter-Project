#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateFromY(const Eigen::VectorXd &y)
{
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;
    P_ = (MatrixXd::Identity(4, 4) - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z)
{
    MatrixXd y = z - H_ * x_;
    UpdateFromY(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
    VectorXd hx(3);
    
    double px = x_[0];
	double py = x_[1];
	double vx = x_[2];
	double vy = x_[3];

    if (abs(py) < 1e-4) {
        cerr << "KalmanFilter::UpdateEKF: py is too small\n";
        return;
    }

    hx <<
        sqrt(px * px + py * py),
        atan2(py, px),
        (px * vx + py * vy) / sqrt(px * px + py * py);

    VectorXd y = z - hx;
    // to [-pi, pi]
    y(1) = fmod(y(1), M_PI);
    UpdateFromY(y);
}
