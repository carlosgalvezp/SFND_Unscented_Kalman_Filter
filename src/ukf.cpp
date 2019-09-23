#include "ukf.h"
#include "tools.h"

namespace
{

std::size_t computeNumberOfSigmaPoints(const std::size_t n_states)
{
    return 2U * n_states + 1U;
}

}  // namespace

UKF::UKF() :
    motion_model_(),
    sensor_model_lidar_(motion_model_.getStateVectorSize()),
    sensor_model_radar_(motion_model_.getStateVectorSize()),
    n_states_(motion_model_.getStateVectorSize()),
    x_(Eigen::VectorXd::Zero(n_states_)),
    P_(Eigen::MatrixXd::Identity(n_states_, n_states_) * kInitialUncertainty),
    x_sig_pred_(),
    lambda_(3.0 - static_cast<double>(motion_model_.getAugStateVectorSize())),
    weights_(computeNumberOfSigmaPoints(motion_model_.getAugStateVectorSize())),
    is_initialized_(false),
    NIS_lidar_(0.0),
    NIS_radar_(0.0)
{
    // Precompute weights
    const std::size_t n_states_augmented = motion_model_.getAugStateVectorSize();
    weights_[0] = lambda_ / (lambda_ + n_states_augmented);

    for (std::size_t i = 1U; i < weights_.size(); ++i)
    {
        weights_[i] = 0.5 / (lambda_ + n_states_augmented);
    }
}

UKF::~UKF(){}

void UKF::Prediction(const double delta_t)
{
    const Eigen::MatrixXd Q = motion_model_.getQ();
    const std::size_t n_augmented = motion_model_.getAugStateVectorSize();

    Eigen::VectorXd x_a = Eigen::VectorXd::Zero(n_augmented);
    Eigen::MatrixXd P_a = Eigen::MatrixXd::Zero(n_augmented, n_augmented);

    x_a.head(n_states_) = x_;
    P_a.topLeftCorner(n_states_, n_states_) = P_;
    P_a.bottomRightCorner(Q.rows(), Q.cols()) = Q;

    // Generate sigma points
    generateSigmaPoints(x_a, P_a, x_sig_pred_);

    // Predict them
    for (std::size_t i = 0U; i < x_sig_pred_.size(); ++i)
    {
        x_sig_pred_[i] = motion_model_.predict(x_sig_pred_[i], delta_t);
    }

    // Compute predicted mean
    x_.fill(0.0);

    for (std::size_t i = 0U; i < weights_.size(); ++i)
    {
        x_ += weights_[i] * x_sig_pred_[i];
    }

    // Compute predicted covariance
    P_.fill(0.0);

    for (std::size_t i = 0U; i < weights_.size(); ++i)
    {
        Eigen::VectorXd x_diff = x_sig_pred_[i] - x_;
        x_diff(3) = Tools::normalizeAngle(x_diff(3));

        P_ += weights_[i] * x_diff * x_diff.transpose();
    }
}

void UKF::initialize(const MeasurementPackage& measurement_pack)
{
    Eigen::VectorXd x0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        x0 = sensor_model_radar_.computeInitialState(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
        x0 = sensor_model_lidar_.computeInitialState(measurement_pack.raw_measurements_);
    }

    if (Tools::isNotZero(x0.norm()))
    {
        x_ = x0;
        is_initialized_ = true;
    }
}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package)
{
    if (!is_initialized_)
    {
        initialize(meas_package);
    }

    // Update
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        NIS_lidar_ = update(sensor_model_lidar_, meas_package.raw_measurements_);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        NIS_radar_ = update(sensor_model_radar_, meas_package.raw_measurements_);
    }
}

double UKF::update(const MeasurementModel& sensor_model,
                   const Eigen::VectorXd& z)
{
    const std::size_t n_observed = z.rows();
    double nis = 0.0;

    // Predict the measurement for each sigma point
    std::vector<Eigen::VectorXd> z_sig_pred(x_sig_pred_.size());

    for (std::size_t i = 0U; i < z_sig_pred.size(); ++i)
    {
        z_sig_pred[i] = sensor_model.predictMeasurement(x_sig_pred_[i]);
    }

    // Compute mean
    Eigen::VectorXd z_mean = Eigen::VectorXd::Zero(n_observed);

    for (std::size_t i = 0U; i < z_sig_pred.size(); ++i)
    {
        z_mean += weights_[i] * z_sig_pred[i];
    }

    // Compute covariance and cross-correlation
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_observed, n_observed);
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n_states_, n_observed);

    for (std::size_t i = 0U; i < z_sig_pred.size(); ++i)
    {
        Eigen::VectorXd z_diff = sensor_model.computeDifference(z_sig_pred[i], z_mean);

        S += weights_[i] * z_diff * z_diff.transpose();

        Eigen::VectorXd x_diff = x_sig_pred_[i] - x_;
        x_diff(3) = Tools::normalizeAngle(x_diff(3));

        T += weights_[i] * x_diff * z_diff.transpose();
    }

    S += sensor_model.getR();

    // Perform update step
    if (Tools::isNotZero(S.determinant()))
    {
        Eigen::MatrixXd K = T * S.inverse();

        Eigen::VectorXd z_diff = sensor_model.computeDifference(z, z_mean);

        x_ = x_ + K * z_diff;
        P_ = P_ - K * S * K.transpose();

        // Compute NIS
        nis = z_diff.transpose() * S.inverse() * z_diff;
    }

    return nis;
}

void UKF::generateSigmaPoints(const Eigen::VectorXd& x,
                              const Eigen::MatrixXd& P,
                              std::vector<Eigen::VectorXd>& x_sig)
{
    const std::size_t n_states = x.rows();
    const std::size_t n_sigma_pts = computeNumberOfSigmaPoints(n_states);

    const Eigen::MatrixXd P_sqrt = Tools::sqrt((lambda_ + n_states) * P);

    x_sig.resize(n_sigma_pts);

    x_sig[0] = x;

    for (std::size_t i = 0U; i < n_states; ++i)
    {
        x_sig[1U + i]             = x + P_sqrt.col(i);
        x_sig[1U + n_states + i]  = x - P_sqrt.col(i);
    }
}
