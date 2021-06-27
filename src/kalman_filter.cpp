#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // estimate
  x_ = x_in;
  // uncertainty covariance
  P_ = P_in;
  // state transition matrix
  F_ = F_in;
  // measurement function
  H_ = H_in;
  // measurement noise
  // relates to sensor error
  R_ = R_in;
  // process covariance matrix
  // relates to acceleration
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * DONE: predict the state
   */
  // x' = Fx + v
  // no given velocity so v is 0
  x_ = F_ * x_ ;
  
  // P' = FPFt + Q
  // increased uncertainty as time goes on
  MatrixXd Ft = F_.transpose();
  P_ = (F_ * P_ * Ft) + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * DONE: update the state by using Kalman Filter equations
   */
  // y = z - Hx'
  // z is the measurement, compare to the prediction
  // compute error
  VectorXd y = z - (H_ * x_);
  
  // S = HP'Ht + R
  // map error into matrix
  MatrixXd Ht = H_.transpose();
  MatrixXd S = (H_ * P_ * Ht) + R_;
  
  // K = P'HtS-1
  // compute kalman gain
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  
  // x = x' + Ky
  // update our estimate
  x_ = x_ + (K * y);
  
  // P = (I - KH)P'
  // estimate new uncertainty
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * DONE: update the state by using Extended Kalman Filter equations
   */
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  // radial distance from origin
  double rho = sqrt(px*px + py*py);
  // angle between p and x
  double theta = atan2(py, px);
  // radial velocity
  double rho_dot = (px*vx + py*vy) / rho;
  
  // create measurement vector h
  // used to map measurement into state space
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
  VectorXd y = z - h;
  
  // normalize the values of EKF to be between PI and -PI
  // help from this forum - https://knowledge.udacity.com/questions/405935
  while (y[1] > M_PI) {
    y(1)-=2.*M_PI;
  }
  while (y[1] < -M_PI)
  {
    y(1)+=2.*M_PI;
  }
  
  // S = HP'Ht + R
  // map error into matrix
  MatrixXd Ht = H_.transpose();
  MatrixXd S = (H_ * P_ * Ht) + R_;
  
  // K = P'HtS-1
  // compute kalman gain
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  
  // x = x' + Ky
  // update our estimate
  x_ = x_ + (K * y);
  
  // P = (I - KH)P'
  // estimate new uncertainty
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
