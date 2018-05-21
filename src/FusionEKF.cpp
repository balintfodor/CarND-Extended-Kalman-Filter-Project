#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd::Zero(2, 4);
  H_laser_(0, 0) = 1;
  H_laser_(1, 1) = 1;
  // will be updated in ProcessMeasurement
  Hj_ = MatrixXd::Zero(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  ekf_.F_ = MatrixXd::Identity(4, 4);

  // actual values will be updated in ProcessMeasurement
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.R_ = MatrixXd(2, 2);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

Eigen::MatrixXd FusionEKF::CalculateQ(double dt, double ax, double ay)
{
  static MatrixXd G = MatrixXd::Zero(4, 2);
  G(0, 0) = G(1, 1) = 0.5 * dt * dt;
  G(2, 0) = G(3, 1) = dt;
  MatrixXd Qv = MatrixXd::Zero(2, 2);
  Qv(0, 0) = ax;
  Qv(1, 1) = ay;
  return G * Qv * G.transpose();
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  VectorXd isInvalid = measurement_pack.raw_measurements_.unaryExpr(
    [](double x) { return isnan(x) || isinf(x); });

  if (isInvalid.any()) {
    return;
  }

  static double noise_ax = 9;
  static double noise_ay = 9;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      double cos_phi = cos(phi);
      double sin_phi = sin(phi);
      double px = rho * cos_phi;
      double py = rho * sin_phi;
      // it is not a true velocity vector but
      // might be ok for initializing the state
      double vx = rho_dot * cos_phi;
      double vy = rho_dot * sin_phi;

      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    ekf_.P_ = CalculateQ(0, noise_ax, noise_ay);

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  double dt = 1e-6 * (measurement_pack.timestamp_ - previous_timestamp_);
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_(0, 2) = ekf_.F_(1, 3) = dt;
  ekf_.Q_ = CalculateQ(dt, noise_ax, noise_ay);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ =\n" << ekf_.x_ << endl;
  cout << "P_ =\n" << ekf_.P_ << endl;
}
