#include "FusionEKF.h"
#include "tools.h"
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

	// Initializing matrices
	R_laser_ = MatrixXd(2, 2); // px, py
	R_radar_ = MatrixXd(3, 3); // rho, phi, rho_dot
	H_laser_ = MatrixXd(2, 4); // px, py -> px, py, vx, vy
	Hj_ = MatrixXd(3, 4); // for radar, rho, phi, rho_dot -> px, py, vx, vy

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;
	/**
	TODO:
	* Finish initializing the FusionEKF.
	* Set the process and measurement noises
	*/

	// H Laser
	H_laser_ << 1, 0, 0, 0,
		0, 1, 0, 0;

	// State covariance matrix/ uncertainty
	P_ = MatrixXd(4, 4);
	P_ << 1, 0, 0, 0,
		    0, 1, 0, 0,
		    0, 0, 1000, 0,
		    0, 0, 0, 1000;

	// State transition matrix
	F_ = MatrixXd(4, 4);
	F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;

	// Process covariance matrix
	Q_ = MatrixXd(4, 4);
	// state
	x_ = VectorXd(4);

	// Process noise terms in Q
	noise_ax = 9;
	noise_ay = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    //ekf_.x_ = VectorXd(4);
    //ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
		cout << "radar ok" << endl;
		ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);

		// get rho, phi, rho_dot
		double rho = measurement_pack.raw_measurements_(0);
		double phi = measurement_pack.raw_measurements_(1);
		double rho_dot = measurement_pack.raw_measurements_(2);
		// convert
		double px = rho*cos(phi);
		double py = rho*sin(phi);
					
		double vx = rho_dot*cos(phi);
		double vy = rho_dot*sin(phi);

		ekf_.x_ << px, py, vx, vy;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		cout << "laser ok" << endl;
		ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

		// initial state with zero velocity vx = vy = 0;
		ekf_.x_ << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
			
    }

    // done initializing, no need to predict or update
	previous_timestamp_ = measurement_pack.timestamp_;
	is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

   /**
   TODO:
   * Update the state transition matrix F according to the new elapsed time.
   - Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

	// Calculate dt: current_timestamp - previous_timestamp
	double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

	// F matrix
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	// Update the process noise covariance matrix Q
	double dt_2 = dt * dt;
	double dt_3 = dt_2 * dt;
	double dt_4 = dt_3 * dt;

	ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
		0, dt_4/4 * noise_ay, 0, dt_3/2 * noise_ay,
		dt_3/2 * noise_ax, 0, dt_2*noise_ax, 0,
		0, dt_3/2 * noise_ay, 0, dt_2*noise_ay;
	
	ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  // Using Jacobian matrix Hj instead of H
	  Hj_ = tools.CalculateJacobian(ekf_.x_);
	  ekf_.H_ = Hj_;
	  ekf_.R_ = R_radar_;
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
	  // Laser update	  
	  // Using basic KF equations for Laser updates
	  ekf_.H_ = H_laser_;
	  ekf_.R_ = R_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
