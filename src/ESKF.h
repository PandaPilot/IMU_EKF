#pragma once

#define EIGEN_NO_MALLOC
#define EIGEN_DONT_ALIGN_STATICALLY
#define EIGEN_DONT_VECTORIZE

#include <stdint.h>
#include <Eigen/Dense>  // Correct Eigen include for matrices
#include "Quaternion.h"
#include "utilities.h"

#define STATE_SIZE 15
#define MEASUREMENT_GYR_SIZE 3  // Corrected spelling of "MEASUREMENT"
#define MEASUREMENT_ACC_SIZE 3
#define MEASUREMENT_MAG_SIZE 3

// Define useful constants and macros
#ifndef SGN
#define SGN(X) ((X > 0) - (X < 0))
#endif

#ifndef RAD_TO_DEG
#define RAD_TO_DEG (180.0 / M_PI)
#endif

#ifndef DEG_TO_RAD
#define DEG_TO_RAD (M_PI / 180.0)
#endif

#ifndef MS2_TO_G
#define MS2_TO_G (1.0 / 9.81)
#endif

#ifndef G_TO_MS2
#define G_TO_MS2 (9.81)
#endif

namespace IMU_EKF
{
template <typename precision>
class ESKF
{
public:
    ESKF();
    void init();
    void initWithAcc(const float ax, const float ay, const float az);
    void initWithAccAndMag(const float ax, const float ay, const float az, const float mx, const float my, const float mz, const Eigen::Matrix<precision, 3, 3> &Winv, const Eigen::Matrix<precision, 3, 1> &V);
    void predict(precision dt);
    void correctGyr(const float gx, const float gy, const float gz);
    void correctAcc(const float ax, const float ay, const float az);
    void correctMag(const float mx, const float my, const float mz, const float incl, const float B, const Eigen::Matrix<precision, 3, 3> &W, const Eigen::Matrix<precision, 3, 1> &V);
    void reset();
    Eigen::Matrix<precision, 15, 1> getState() const;
    void getAttitude(float &roll, float &pitch, float &yaw) const;
    Quaternion<precision> getAttitude() const;
    void getAcceleration(float &x, float &y, float &z) const;

private:
    Eigen::Matrix<precision, 15, 1> x_;
    Eigen::Matrix<precision, 15, 15> P_;
    Eigen::Matrix<precision, 15, 15> Q_;
    Eigen::Matrix<precision, 3, 3> R_Gyr_;
    Eigen::Matrix<precision, 3, 3> R_Acc_;
    Eigen::Matrix<precision, 3, 3> R_Mag_;
    Quaternion<precision> qref_;
};

// Template function definitions moved from the .cpp file

template <typename precision>
ESKF<precision>::ESKF()
{
    init();
}

template <typename precision>
void ESKF<precision>::init()
{
    qref_ = Quaternion<precision>();
    x_.setZero();
    P_.setIdentity();
    Q_.setIdentity();
    
    // Set up covariance matrices (avoid hardcoding these values later)
    Q_(0, 0) = 0.05; Q_(1, 1) = 0.05; Q_(2, 2) = 0.05;
    Q_(3, 3) = 0.05; Q_(4, 4) = 0.05; Q_(5, 5) = 0.05;
    Q_(6, 6) = 0.025; Q_(7, 7) = 0.025; Q_(8, 8) = 0.025;
    Q_(9, 9) = 0.025; Q_(10, 10) = 0.025; Q_(11, 11) = 0.025;
    Q_(12, 12) = 0.01; Q_(13, 13) = 0.01; Q_(14, 14) = 0.01;

    R_Gyr_.setIdentity();
    R_Gyr_(0, 0) = 0.0000045494 * DEG_TO_RAD;
    R_Gyr_(1, 1) = 0.0000039704 * DEG_TO_RAD;
    R_Gyr_(2, 2) = 0.0000093844 * DEG_TO_RAD;

    R_Acc_.setIdentity();
    R_Acc_(0, 0) = 0.0141615383 * G_TO_MS2;
    R_Acc_(1, 1) = 0.0164647549 * G_TO_MS2;
    R_Acc_(2, 2) = 0.0100094303 * G_TO_MS2;

    R_Mag_.setIdentity();
    R_Mag_(0, 0) = 0.00004;
    R_Mag_(1, 1) = 0.00004;
    R_Mag_(2, 2) = 0.00004;
}

template <typename precision>
void ESKF<precision>::initWithAcc(const float ax, const float ay, const float az)
{
    init();
    precision roll = std::atan2(ay, SGN(-az) * std::sqrt(az * az + 0.01 * ax * ax));
    precision pitch = std::atan(-ax / std::sqrt(ay * ay + az * az));

    precision sr05 = std::sin(0.5 * roll);
    precision cr05 = std::cos(0.5 * roll);
    precision sp05 = std::sin(0.5 * pitch);
    precision cp05 = std::cos(0.5 * pitch);
    qref_ = Quaternion<precision>(sr05, 0, 0, cr05) * Quaternion<precision>(0, sp05, 0, cp05);
}

template <typename precision>
void ESKF<precision>::initWithAccAndMag(const float ax, const float ay, const float az, const float mx, const float my, const float mz, const Eigen::Matrix<precision, 3, 3> &Winv, const Eigen::Matrix<precision, 3, 1> &V)
{
    init();

    precision roll = std::atan2(ay, SGN(-az) * std::sqrt(az * az + 0.01 * ax * ax));
    precision pitch = std::atan(-ax / std::sqrt(ay * ay + az * az));

    precision sr = std::sin(roll);
    precision cr = std::cos(roll);
    precision sp = std::sin(pitch);
    precision cp = std::cos(pitch);

    Eigen::Matrix<precision, 3, 3> RxT;
    RxT << 1, 0, 0,
           0, cr, sr,
           0, -sr, cr;
    Eigen::Matrix<precision, 3, 3> RyT;
    RyT << cp, 0, -sp,
           0, 1, 0,
           sp, 0, cp;

    Eigen::Matrix<precision, 3, 1> Bp(mx, my, mz);
    Eigen::Matrix<precision, 3, 1> Bf = RyT * RxT * Winv * (Bp - V);

    precision yaw = -atan2(-Bf(1), Bf(0));

    precision sr05 = std::sin(0.5 * roll);
    precision cr05 = std::cos(0.5 * roll);
    precision sp05 = std::sin(0.5 * pitch);
    precision cp05 = std::cos(0.5 * pitch);
    precision sy05 = std::sin(0.5 * yaw);
    precision cy05 = std::cos(0.5 * yaw);

    qref_ = Quaternion<precision>(sr05, 0, 0, cr05) * Quaternion<precision>(0, sp05, 0, cp05) * Quaternion<precision>(0, 0, sy05, cy05);
}

template <typename precision>
void ESKF<precision>::predict(precision dt)
{
    Quaternion<precision> angular_velocity_quat(x_[12], x_[13], x_[14], 0);
    qref_ += dt * ((0.5 * angular_velocity_quat) * qref_);
    qref_.normalize();

    Eigen::Matrix<precision, 9, 9> A;
    A.setIdentity();
    A(0, 3) = dt;
    A(0, 6) = dt * dt * 0.5;
    A(1, 4) = dt;
    A(1, 7) = dt * dt * 0.5;
    A(2, 5) = dt;
    A(2, 8) = dt * dt * 0.5;
    A(3, 6) = dt;
    A(4, 7) = dt;
    A(5, 8) = dt;

    x_.segment(0, 9) = A * x_.segment(0, 9);
    P_.topLeftCorner(9, 9) = A * P_.topLeftCorner(9, 9) * A.transpose() + dt * Q_.topLeftCorner(9, 9);
}

template <typename precision>
void ESKF<precision>::correctGyr(const float gx, const float gy, const float gz)
{
    Eigen::Matrix<precision, 3, 1> z;
    z << gx * DEG_TO_RAD, gy * DEG_TO_RAD, gz * DEG_TO_RAD;

    Eigen::Matrix<precision, 3, 3> K = P_.bottomRightCorner(3, 3) * (P_.bottomRightCorner(3, 3) + R_Gyr_).inverse();
    x_.segment(12, 3) += K * (z - x_.segment(12, 3));
    P_.bottomRightCorner(3, 3) -= K * P_.bottomRightCorner(3, 3);
}

template <typename precision>
void ESKF<precision>::correctAcc(const float ax, const float ay, const float az)
{
    Eigen::Matrix<precision, 3, 1> z;
    z << ax * G_TO_MS2, ay * G_TO_MS2, az * G_TO_MS2;

    Eigen::Matrix<precision, 3, 1> gravity(0.0, 0.0, -9.81);

    if (std::abs(z.norm() - 9.81) < 0.7)
    {
        Eigen::Matrix<precision, 3, 3> K = P_.block(6, 6, 3, 3) * (P_.block(6, 6, 3, 3) + R_Acc_).inverse();
        x_.segment(6, 3) += K * (-x_.segment(6, 3));
        P_.block(6, 6, 3, 3) -= K * P_.block(6, 6, 3, 3);
    }
}

template <typename precision>
void ESKF<precision>::correctMag(const float mx, const float my, const float mz, const float incl, const float B, const Eigen::Matrix<precision, 3, 3> &W, const Eigen::Matrix<precision, 3, 1> &V)
{
    Eigen::Matrix<precision, 3, 1> z(mx, my, mz);
    Eigen::Matrix<precision, 3, 1> vi(std::cos(incl), 0, -std::sin(incl));
    vi = B * vi;

    Eigen::Matrix<precision, 3, 1> vb_pred = qref_.toRotationMatrix() * vi;

    Eigen::Matrix<precision, 3, 3> K = P_.block(9, 9, 3, 3) * W.transpose() * (W * P_.block(9, 9, 3, 3) * W.transpose() + R_Mag_).inverse();
    x_.segment(9, 3) += K * (z - vb_pred - W * x_.segment(9, 3));
    P_.block(9, 9, 3, 3) -= K * W * P_.block(9, 9, 3, 3);
}

template <typename precision>
void ESKF<precision>::reset()
{
    qref_ = Quaternion<precision>(x_(9), x_(10), x_(11), 2.0) * qref_;
    qref_.normalize();
    x_(9) = 0.0;
    x_(10) = 0.0;
    x_(11) = 0.0;
}

template <typename precision>
Eigen::Matrix<precision, 15, 1> ESKF<precision>::getState() const
{
    return x_;
}

template <typename precision>
void ESKF<precision>::getAttitude(float &roll, float &pitch, float &yaw) const
{
    Eigen::Matrix<precision, 3, 1> angles = qref_.toEulerAngles();
    roll = angles(0);
    pitch = angles(1);
    yaw = angles(2);
}

template <typename precision>
Quaternion<precision> ESKF<precision>::getAttitude() const
{
    return qref_;
}

template <typename precision>
void ESKF<precision>::getAcceleration(float &x, float &y, float &z) const
{
    x = x_(6);
    y = x_(7);
    z = x_(8);
}
} // namespace IMU_EKF

