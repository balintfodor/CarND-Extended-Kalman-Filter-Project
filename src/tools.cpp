#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
	const vector<VectorXd> &ground_truth)
{
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		cerr << "Tools::CalculateRMSE: invalid input\n";
		return rmse;
	}

	for (size_t i = 0; i < estimations.size(); ++i) {
		VectorXd r = estimations[i] - ground_truth[i];
		r = r.array() * r.array();
		rmse += r;
	}
	rmse /= 1.0 * estimations.size();

	return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
	MatrixXd Hj(3, 4);

	double px = x_state[0];
	double py = x_state[1];
	double vx = x_state[2];
	double vy = x_state[3];
	
	double d1 = px * px + py * py;

	if (d1 < 1e-4) {
		cerr << "Tools::CalculateJacobian: denominator is too small\n";
		return Hj;
	}

	double d2 = sqrt(d1);
	double d3 = d2 * d2 * d2;
	double f1 = vx * py - vy * px;
	double f2 = -f1;

	Hj <<
		px / d2,      py / d2,      0,       0,
		- py / d1,    px / d1,      0,       0,
		py * f1 / d3, px * f2 / d3, px / d2, py / d2;

	return Hj;
}
