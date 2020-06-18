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
// Created by hyye on 4/4/18.
//

#ifndef LIO_IMUFACTOR_H_
#define LIO_IMUFACTOR_H_

/// adapted from VINS-mono

#include <ceres/ceres.h>
#include "imu_processor/IntegrationBase.h"
#include "utils/math_utils.h"

namespace lio {

using namespace mathutils;

class ImuFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> { //TODO15,6,9,6,9

 public:
  ImuFactor() = delete;
  ImuFactor(std::shared_ptr<IntegrationBase> pre_integration) : pre_integration_{
      pre_integration} {
    // NOTE: g_vec_ is the gravity in laser's original frame
    g_vec_ = pre_integration_->g_vec_; //TODO应该是g_b0(0, 0, -g)
  }

  
  //窗口内存放的是b0_bk_(p,v,q); ba, bg
  // | .....     |<---- pre_integration_ ---->|               |                |    .......           |             
  //start        i                            j                                                      end  
  //start,end:窗口内第一帧,最后一帧  
  //i:需要优化的第一帧; j紧挨着i的下一个位姿
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); //第一个参数块

    Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
    Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]); //第二个参数块

    Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); //第三个参数块

    Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
    Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]); //第四个参数块

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = pre_integration_->Evaluate(Pi, Qi, Vi, Bai, Bgi,
                                          Pj, Qj, Vj, Baj, Bgj); 
    //i和j之间的imu预积分的残差项(15*1),见liom补充材料 "C IMU Residual", 
    //注意：其中预积分测量值(delta_p, delta_v, delta_q)需要先用(Bai - pre_integration_->linearized_ba_) 
    //  和(Bgi - pre_integration_->linearized_bg_)更新测量值，然后再带到残差表达式中
    //残差 r = [预积分位置测量残差项  预积分旋转测量残差项  预积分速度测量残差项   delta_ba   delta_bg]
    // r = [rp rq rv rba rbg]  
    //预积分paper作者给出的是残差对变量的增量的雅克比, 不是对状态的雅克比。


    Eigen::Matrix<double, 15, 15> sqrt_info =
        Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration_->covariance_.inverse()).matrixL().transpose();//预积分的方差作为残差的权重
    //Cholesky分解
    residual = sqrt_info * residual; 
    
    if (jacobians) {
      double sum_dt = pre_integration_->sum_dt_;
      Eigen::Matrix3d dp_dba = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BA); //位置预积分测量delta_p对ba雅克比
      Eigen::Matrix3d dp_dbg = pre_integration_->jacobian_.template block<3, 3>(O_P, O_BG); //位置预积分测量delta_p对bg雅克比

      Eigen::Matrix3d dq_dbg = pre_integration_->jacobian_.template block<3, 3>(O_R, O_BG); //旋转预积分测量delta_r对bg雅克比

      Eigen::Matrix3d dv_dba = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BA); //速度预积分测量delta_v对ba雅克比
      Eigen::Matrix3d dv_dbg = pre_integration_->jacobian_.template block<3, 3>(O_V, O_BG); //速度预积分测量delta_v对bg雅克比

      if (pre_integration_->jacobian_.maxCoeff() > 1e8 || pre_integration_->jacobian_.minCoeff() < -1e8) {
        ROS_DEBUG("numerical unstable in preintegration");
      }

      if (jacobians[0]) {//残差对第一个参数块(pi, qi)
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]); //TODO:15*6 ?
        jacobian_pose_i.setZero();

        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix(); //rp对pi
        jacobian_pose_i.block<3, 3>(O_P, O_R) = //rp对qi
            SkewSymmetric(Qi.inverse() * (-0.5 * g_vec_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt)); //右扰动

        Eigen::Quaterniond corrected_delta_q =
            pre_integration_->delta_q_ * DeltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
        jacobian_pose_i.block<3, 3>(O_R, O_R) = //rq对qi
            -(LeftQuatMatrix(Qj.inverse() * Qi) * RightQuatMatrix(corrected_delta_q)).topLeftCorner<3, 3>();
            //TODO按照右扰动我求出来的是: -(Qj.toRotationMatrix()).transpose() * (Qi.toRotationMatrix())
            //预积分文章中是对李代数求导，结果中含有右雅克比
        
        jacobian_pose_i.block<3, 3>(O_V, O_R) = //rv对qi
            SkewSymmetric(Qi.inverse() * (-g_vec_ * sum_dt + Vj - Vi));

        jacobian_pose_i = sqrt_info * jacobian_pose_i; //方差也作为了jacobian的权重

        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
          ROS_DEBUG("numerical unstable in preintegration");
        }
      }
      if (jacobians[1]) {//对第二个参数块(vi, bai, bgi)
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
        jacobian_speedbias_i.setZero();
        jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt; //r_p对vi
        jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba; //TODOr_p对bai,符号应该为正?
        jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg; //TODOr_p对bgi,符号应该为正?

        Eigen::Quaterniond corrected_delta_q =
            pre_integration_->delta_q_ * DeltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
        jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = //r_q对bgi
            -LeftQuatMatrix(Qj.inverse() * Qi * corrected_delta_q).topLeftCorner<3, 3>() * dq_dbg;
        //TODO这个该怎么推？

        jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix(); //r_v对vi
        jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba; //TODOr_v对bai,符号应该为正?
        jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg; //TODOr_v对bgi,符号应该为正?

        jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity(); //r_ba对bai

        jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity(); //r_bg对bgi

        jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

      }
      if (jacobians[2]) {//对第三个参数块(pj, qj)
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]); //TODO:15*6 ?
        jacobian_pose_j.setZero();

        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix(); //r_p对pj

        Eigen::Quaterniond corrected_delta_q =
            pre_integration_->delta_q_ * DeltaQ(dq_dbg * (Bgi - pre_integration_->linearized_bg_));
        jacobian_pose_j.block<3, 3>(O_R, O_R) = //r_q对qj
            LeftQuatMatrix(corrected_delta_q.inverse() * Qi.inverse() * Qj).topLeftCorner<3, 3>();
        //TODO按照右扰动我求出来的是： I

        jacobian_pose_j = sqrt_info * jacobian_pose_j;

      }
      if (jacobians[3]) {//对第四个参数块(vj, baj, bgj)
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
        jacobian_speedbias_j.setZero();
        
        //TODO缺少了rp对baj, rp对bgj

        //TODO缺少了r_q对bgj

        jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix(); //rv对vj
        //TODO缺少了rv对baj, rv对bgj

        jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity(); //rba对baj

        jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity(); //rbg对bgj

        jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

      }
    }

    return true;
  }

  std::shared_ptr<IntegrationBase> pre_integration_;
  Eigen::Vector3d g_vec_;

  const double eps_ = 10e-8;

};

} // namespace lio

#endif //LIO_IMUFACTOR_H_
