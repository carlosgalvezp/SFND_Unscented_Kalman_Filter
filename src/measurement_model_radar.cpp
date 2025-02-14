#include "measurement_model_radar.h"
#include "tools.h"
#include <cmath>

namespace
{
// Radar measurement noise standard deviation - range
constexpr double std_range      = 0.3;   // [m]

// Radar measurement noise standard deviation - bearing
constexpr double std_bearing    = 0.03;  // [rad]

// Radar measurement noise standard deviation - range-rate
constexpr double std_range_rate = 0.3;   // [m/s]
}  // namespace

MeasurementModelRadar::MeasurementModelRadar(const std::size_t n_states):
    MeasurementModel(n_states),
    R_(Eigen::MatrixXd::Zero(n_observed_states_, n_observed_states_))
{
    R_(0, 0) = std_range * std_range;
    R_(1, 1) = std_bearing * std_bearing;
    R_(2, 2) = std_range_rate * std_range_rate;
}

MeasurementModelRadar::~MeasurementModelRadar()
{
}

Eigen::VectorXd MeasurementModelRadar::computeInitialState(
        const Eigen::VectorXd& z) const
{
    const double rho = z(0);
    const double phi = z(1);

    const double px = rho * std::cos(phi);
    const double py = rho * std::sin(phi);

    Eigen::VectorXd x(n_states_);
    x << px, py, 0.0, 0.0, 0.0;

    return x;
}

Eigen::VectorXd MeasurementModelRadar::predictMeasurement(
    const Eigen::VectorXd& state) const
{
    Eigen::VectorXd z_hat = Eigen::VectorXd::Zero(n_observed_states_);

    const double px = state(0);
    const double py = state(1);
    const double v = state(2);
    const double yaw = state(3);

    const double vx = v * std::cos(yaw);
    const double vy = v * std::sin(yaw);

    const double sqrt_sum = std::sqrt(px * px + py * py);

    if (Tools::isNotZero(sqrt_sum))
    {
        z_hat << sqrt_sum,
              std::atan2(py, px),
              (px * vx + py * vy) / sqrt_sum;
    }

    return z_hat;
}

Eigen::VectorXd MeasurementModelRadar::computeDifference(
    const Eigen::VectorXd& z_a,
    const Eigen::VectorXd& z_b) const
{
    Eigen::VectorXd y = z_a - z_b;
    y(1) = Tools::normalizeAngle(y(1));

    return y;
}
