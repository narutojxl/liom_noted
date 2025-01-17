/**
* This file is part of LIO-mapping.
* 
* Copyright (C) 2019 Haoyang Ye <hy.ye at connect dot ust dot hk>,
* Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
* The Hong Kong University of Science and Technology
* 
* For more information please see <https://ram-lab.com/file/hyye/lio-mapping>
* or <https://sites.google.com/view/lio-mapping>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* LIO-mapping is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* LIO-mapping is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with LIO-mapping.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by hyye on 5/17/18.
//

#ifndef LIO_PIVOTPOINTPLANEFACTOR_H_
#define LIO_PIVOTPOINTPLANEFACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "utils/CircularBuffer.h"
#include "utils/Twist.h"
#include "utils/common_ros.h"
#include "utils/TicToc.h"
#include "utils/math_utils.h"
#include "utils/geometry_utils.h"

namespace lio {

using namespace mathutils;

class PivotPointPlaneFactor : public ceres::SizedCostFunction<1, 7, 7, 7> {//残差first

 public:
  PivotPointPlaneFactor(const Eigen::Vector3d &point,
                        const Eigen::Vector4d &coeff);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void Check(double **parameters);

  Eigen::Vector3d point_; //窗口内在自己帧下的点, p_raw
  Eigen::Vector4d coeff_; //残差对pi的雅克比，pi由p_raw转换而来

  // TODO: necessary?
//  static Eigen::Matrix3d sqrt_info;
  static double sum_t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

}

#endif //LIO_PIVOTPOINTPLANEFACTOR_H_
