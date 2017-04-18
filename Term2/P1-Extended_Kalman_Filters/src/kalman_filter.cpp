#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {

  x_ = x_in; // state vector
  P_ = P_in; // state covariance matrix
  F_ = F_in; // state transition matrix
  H_ = H_in; // measurement matrix
  R_ = R_in; // measurement covariance matrix
  Q_ = Q_in; // process covariance matrix
}

void KalmanFilter::Predict() {
	/**
	TODO:
	* predict the state
	*/
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	/**
	TODO:
	* update the state by using Kalman Filter equations
	*/
	// KF measurement update step
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;

	// new estimation
	x_ = x_ + K*y;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K*H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	/**
	TODO:
	* update the state by using Extended Kalman Filter equations
	*/
	//KF measurement update step
	double px = x_(0);
	double py = x_(1);
	double vx = x_(2);
	double vy = x_(3);

	// Map state into measurement space
	// In this case, cartesian to polar coordinates
	double rho = sqrt(px*px + py*py);
	double phi = atan2(py,px);
	
	if (rho < 0.0001) rho = 0.0001;
	double rho_dot = (px*vx + py*vy) / rho;

	VectorXd z_pred(3);
	z_pred << rho, phi, rho_dot;
	VectorXd y = z - z_pred;

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_*P_*Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_*Ht*Si;
	
	// new estimation
	x_ = x_ + (K*y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K*H_)*P_;
}
