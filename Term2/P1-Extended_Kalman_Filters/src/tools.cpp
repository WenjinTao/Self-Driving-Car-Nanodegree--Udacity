#include <iostream>
using namespace std;
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  int N = estimations.size();

  if (N != ground_truth.size()
	  || N == 0) {
	  cout << "Invalid estimation or ground_truth data" << endl;
	  return rmse;
  }

  //accumulate squared residuals
  for (int i=0; i<N; ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;

  }

  rmse = rmse/N;
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4); 
  //recover state statement
  double px = x_state(0);
  double py = x_state(1);   
  double vx = x_state(2);
  double vy = x_state(3);
  
  // pre-compute a set of terms to avoid repeated calculation
  double c1 = px*px + py*py;
  if (fabs(c1) < 0.00001) c1 = 0.00001;
  double c2 = sqrt(c1);
  double c3 = c1*c2; 

  // compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
	  -(py/c1), (px/c1), 0, 0,
	  (py*(vx*py - vy*px)/c3), (px*(vy*px - vx*py)/c3), (px/c2), (py/c2);
  return Hj;

}
