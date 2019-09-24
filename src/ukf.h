#ifndef UKF_H
#define UKF_H
#include <Eigen/Dense>
#include <vector>

#include "measurement_package.h"
#include "motion_model.h"
#include "measurement_model.h"
#include "measurement_model_lidar.h"
#include "measurement_model_radar.h"

static const double kInitialUncertainty = 1.0F;

/// \brief Implements the Unscented Kalman Filter (UKF)
class UKF
{
 public:
    /// \brief UKF
    /// \param motion_model reference to the motion model used
    UKF();

    ~UKF();

    /// \brief returns the current state
    /// \return the current state
    const Eigen::VectorXd& getState() const { return x_; }

    /// \brief sets the state to a new value
    /// \param x new state
    void setState(const Eigen::VectorXd& x) { x_ = x; }

    /// \brief Predicts sigma points, the state, and the state covariance matrix.
    ///
    /// \param delta_t Time between k and k+1 in s.
    void Prediction(double delta_t);

    /// \brief ProcessMeasurement
    ///
    /// \param meas_package The latest measurement data of either radar or laser.
    void ProcessMeasurement(const MeasurementPackage& meas_package);

 private:
    // Motion model
    MotionModel motion_model_;

    // Measurement models
    MeasurementModelLidar sensor_model_lidar_;
    MeasurementModelRadar sensor_model_radar_;

    // Timestamp of the last measurement
    std::size_t previous_timestamp_;

    // Dimension of the state vector
    std::size_t n_states_;

    // State vector: [pos_x pos_y velocity yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

    // State covariance matrix
    Eigen::MatrixXd P_;

    // Predicted sigma points
    std::vector<Eigen::VectorXd> x_sig_pred_;

    // Lambda parameter
    double lambda_;

    // Weights of sigma points
    std::vector<double> weights_;

    // Initialized flag
    bool is_initialized_;

    // If this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // If this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // NIS
    double NIS_lidar_;
    double NIS_radar_;

    void initialize(const MeasurementPackage& measurement_pack);

    void generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P,
                             std::vector<Eigen::VectorXd>& x_sig);

    double update(const MeasurementModel& sensor_model, const Eigen::VectorXd& z);
};

#endif /* UKF_H */
