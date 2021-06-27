#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * DONE: Calculate the RMSE here.
   * help from this lesson - https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/ec3054b9-9ffc-45c4-8523-485e2f7022da/concepts/c46a47f0-7cdc-4e49-b225-5134b438255a
   */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  // verify the size of the incoming vectors
  if (ground_truth.size() != estimations.size() || ground_truth.size() == 0){
    std::cout << "Invalid estimations (" << estimations.size() << ") or ground truths (" << ground_truth.size() << ")" << std::endl;
    exit(1);
  }
  
  // accumulate squared residuals
  for (unsigned int i=0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();
  
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * DONE:
   * Calculate a Jacobian here.
   * Mostly pulled from this lesson - https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/95d62426-4da9-49a6-9195-603e0f81d3f1/lessons/ec3054b9-9ffc-45c4-8523-485e2f7022da/concepts/08fc65c1-04d9-45d3-8a98-abf7bb072dc2
   */
  MatrixXd Hj(3,4);
  
  if (x_state.size() != 4) {
    std::cout << "CalculateJacobian Requires a vector with a length of 4";
    exit(1);
  }
  
  // get the state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);
  
  // check division by zero
  if (fabs(c1) < 0.0001) {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }
  
  // compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  
  return Hj;
}
