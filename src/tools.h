#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"
#include <pcl/io/pcd_io.h>

#include "render/render.h"
#include "constants.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct lmarker
{
    double x, y;
    lmarker(double setX, double setY)
        : x(setX), y(setY)
    {}

};

struct rmarker
{
    double rho, phi, rho_dot;
    rmarker(double setRho, double setPhi, double setRhoDot)
        : rho(setRho), phi(setPhi), rho_dot(setRhoDot)
    {}

};

class Tools {
    public:
    /**
    * Constructor.
    */
    Tools();

    /**
    * Destructor.
    */
    virtual ~Tools();

    // Members
    std::vector<VectorXd> estimations;
    std::vector<VectorXd> ground_truth;

    double noise(double stddev, long long seedNum);
    lmarker lidarSense(Car& car, pcl::visualization::PCLVisualizer::Ptr& viewer, long long timestamp, bool visualize);
    rmarker radarSense(Car& car, Car ego, pcl::visualization::PCLVisualizer::Ptr& viewer, long long timestamp, bool visualize);
    void ukfResults(Car car, pcl::visualization::PCLVisualizer::Ptr& viewer, double time, int steps);
    /**
    * A helper method to calculate RMSE.
    */
    VectorXd CalculateRMSE(const std::vector<VectorXd> &estimations, const std::vector<VectorXd> &ground_truth);
    void savePcd(typename pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string file);
    pcl::PointCloud<pcl::PointXYZ>::Ptr loadPcd(std::string file);

    /// \brief Returns true if x is not zero, up to some predefined threshold
    /// \param x input
    /// \return true if x is greater than a small threshold
    static bool isNotZero(const double x);

    /// \brief Normalizes an angle to the range [-pi, pi)
    /// \param x angle to normalize
    /// \return the normalized angle
    static double normalizeAngle(const double x);

    /// \brief Computes the square root of a matrix
    /// \param x the input matrix
    /// \return the square root of x, A, such that x = A' * A
    static Eigen::MatrixXd sqrt(const Eigen::MatrixXd& x);
};

#endif /* TOOLS_H_ */
