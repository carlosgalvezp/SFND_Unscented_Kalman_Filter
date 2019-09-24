#include "measurement_model_lidar.h"

namespace
{
// Laser measurement noise standard deviation - px
constexpr double std_px = 0.15;  // [m]^2

// Laser measurement noise standard deviation - py
constexpr double std_py = 0.15;  // [m]^2
}  // namespace

MeasurementModelLidar::MeasurementModelLidar(std::size_t n_states):
    MeasurementModel(n_states),
    R_(Eigen::MatrixXd::Zero(n_observed_states_, n_observed_states_))
{
    R_(0, 0) = std_px * std_px;
    R_(1, 1) = std_py * std_py;
}

MeasurementModelLidar::~MeasurementModelLidar()
{
}

Eigen::VectorXd MeasurementModelLidar::computeInitialState(
        const Eigen::VectorXd& z) const
{
    Eigen::VectorXd x(n_states_);
    x << z(0), z(1), 0.0, 0.0, 0.0;

    return x;
}

Eigen::VectorXd MeasurementModelLidar::predictMeasurement(
    const Eigen::VectorXd& state) const
{
    return state.head(n_observed_states_);
}

Eigen::VectorXd MeasurementModelLidar::computeDifference(
    const Eigen::VectorXd& z_a,
    const Eigen::VectorXd& z_b) const
{
    return z_a - z_b;
}
