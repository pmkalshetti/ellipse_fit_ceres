#include "random.h"

#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Vector2d = Eigen::Vector2d;
using Point = Vector2d;

struct Ellipse
{
    double a{};  // major axis
    double b{};  // minor axis
    double x0{}; // x translation
    double y0{}; // y translation
    double r{};  // rotation

    Ellipse(double a = 7.5, double b = 2,
            double x0 = 17, double y0 = 23, double r = 0.23)
        : a(a), b(b), x0(x0), y0(y0), r(r)
    {
    }

    Ellipse(const VectorXd &ellipse_params) : a(ellipse_params(0)),
                                              b(ellipse_params(1)),
                                              x0(ellipse_params(2)),
                                              y0(ellipse_params(3)),
                                              r(ellipse_params(4)) {}

    Ellipse(const double *ellipse_params) : a(ellipse_params[0]),
                                            b(ellipse_params[1]),
                                            x0(ellipse_params[2]),
                                            y0(ellipse_params[3]),
                                            r(ellipse_params[4]) {}

    Point evaluate(const double t) const
    {
        Point point;
        point(0) = x0 + a * cos(t) * cos(r) - b * sin(t) * sin(r);
        point(1) = y0 + a * cos(t) * sin(r) + b * sin(t) * cos(r);

        return point;
    }

    void set_params(const VectorXd &ellipse_params)
    {
        a = ellipse_params(0);
        b = ellipse_params(1);
        x0 = ellipse_params(2);
        y0 = ellipse_params(3);
        r = ellipse_params(4);
    }

    double dfx_dt(const double t) const
    {
        return a * cos(r) * sin(t) + b * sin(r) * cos(t);
    }

    double dfx_da(const double t) const
    {
        return -cos(t) * cos(r);
    }

    double dfx_db(const double t) const
    {
        return sin(t) * sin(r);
    }

    double dfx_dx0() const
    {
        return -1;
    }

    double dfx_dr(const double t) const
    {
        return a * cos(t) * sin(r) + b * sin(t) * cos(r);
    }

    double dfy_dt(const double t) const
    {
        return a * sin(r) * sin(t) - b * cos(r) * cos(t);
    }

    double dfy_da(const double t) const
    {
        return -cos(t) * sin(r);
    }

    double dfy_db(const double t) const
    {
        return -sin(t) * cos(r);
    }

    double dfy_dy0() const
    {
        return -1;
    }

    double dfy_dr(const double t) const
    {
        return -a * cos(t) * cos(r) + b * sin(t) * sin(r);
    }

    friend std::ostream &operator<<(std::ostream &out, const Ellipse &e)
    {
        out << "Ellipse Parameters: (a=" << e.a << ", b=" << e.b << ", x0=" << e.x0 << ", y0=" << e.y0 << ", r=" << e.r * 180 / EIGEN_PI << ")";
        return out;
    }

    std::shared_ptr<std::vector<Point>> generate_points_on_boundary(const int n_data_points, const double fraction_of_circumference)
    {
        double incr = fraction_of_circumference * 2 * M_PI / n_data_points;
        std::vector<Point> ellipse_points(n_data_points);
        Random rand;
        for (int i{0}; i < n_data_points; ++i)
        {
            double t = i * incr;
            ellipse_points[i] = evaluate(t);
            ellipse_points[i] += Vector2d{rand.normal(0.0, 0.01), rand.normal(0.0, 0.01)};
        }
        return std::make_shared<std::vector<Point>>(ellipse_points);
    }
};

struct Parameters
{
    VectorXd correspondences;
    VectorXd ellipse;

    Parameters(const std::vector<Point> &ellipse_points, const double fraction_of_circumference)
    : ellipse(5), correspondences(ellipse_points.size())
    {
        // init correspondences
        double incr = fraction_of_circumference * 2 * M_PI / ellipse_points.size();
        for (int i{0}; i < ellipse_points.size(); ++i)
        {
            correspondences(i) = i * incr;
        }

        // init ellipse parameters
        Eigen::Map<const MatrixXd> points_mat(&(ellipse_points[0](0)), ellipse_points.size(), 2);
        Point min_corner = points_mat.colwise().minCoeff();
        Point max_corner = points_mat.colwise().maxCoeff();
        ellipse(0) = 0.5 * (max_corner(0) - min_corner(0));
        ellipse(1) = 0.5 * (max_corner(1) - min_corner(1));
        ellipse(2) = 0.5 * (max_corner(0) + min_corner(0));
        ellipse(3) = 0.5 * (max_corner(1) + min_corner(1));
        ellipse(4) = 0;
    }
};

class DataTerm : public ceres::CostFunction
{
public:
    DataTerm(const Point data_point) : data_point(data_point)
    {
        // set parameter sizes
        // first parameter is correspondence
        // next paramteter is ellipse params
        mutable_parameter_block_sizes()->push_back(1); // 1d parametric position in curve
        mutable_parameter_block_sizes()->push_back(5); // 5 ellipse parameters

        // set residual size
        set_num_residuals(2); // x and y position distance
    }

    virtual bool Evaluate(const double *const *x, double *residuals, double **jacobians) const
    {
        const double t = x[0][0];
        const double *ellipse_params = x[1];
        Ellipse ellipse(ellipse_params);
        Point diff = data_point - ellipse.evaluate(x[0][0]);
        residuals[0] = diff(0);
        residuals[1] = diff(1);

        if (jacobians == nullptr)
            return true;

        // derivative wrt correspondence
        if (jacobians[0] != nullptr)
        {
            jacobians[0][0] = ellipse.dfx_dt(t);
            jacobians[0][1] = ellipse.dfy_dt(t);
        }

        // derivative wrt ellipse params
        if (jacobians[1] != nullptr)
        {
            ceres::MatrixRef(jacobians[1], 2, 5).setZero();

            jacobians[1][0] = ellipse.dfx_da(t);
            jacobians[1][1] = ellipse.dfx_db(t);
            jacobians[1][2] = ellipse.dfx_dx0();
            jacobians[1][4] = ellipse.dfx_dr(t);

            jacobians[1][5] = ellipse.dfy_da(t);
            jacobians[1][6] = ellipse.dfy_db(t);
            jacobians[1][8] = ellipse.dfy_dy0();
            jacobians[1][9] = ellipse.dfy_dr(t);
        }

        return true;
    }

    static CostFunction *create(Point data_point)
    {
        return new DataTerm(data_point);
    }

private:
    const Point data_point;
};

class EllipseParameterization : public ceres::LocalParameterization
{
public:
    EllipseParameterization(const int size) : size(size) {}

    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override
    {
        ceres::VectorRef(x_plus_delta, size) = ceres::ConstVectorRef(x, size) + ceres::ConstVectorRef(delta, size);
        return true;
    }

    bool ComputeJacobian(const double *x, double *jacobian) const override
    {
        ceres::MatrixRef(jacobian, size, size).setIdentity();
        return true;
    }

    int GlobalSize() const override { return size; }
    int LocalSize() const override { return size; }

private:
    const int size;
};

// this is not required for this problem, it is used to understand its usage
void set_local_parameterization(ceres::Problem *problem, Parameters *parameters)
{
    EllipseParameterization *ellipse_parameterization = new EllipseParameterization(parameters->ellipse.size());
    problem->SetParameterization(parameters->ellipse.data(), ellipse_parameterization);
}

void construct_problem(const std::vector<Point> &data, Parameters *parameters, ceres::Problem *problem)
{
    // define parameter blocks
    std::vector<double *> parameter_blocks(1 + 1); // 1 for correspondence, 1 for ellipse params
    parameter_blocks[1] = parameters->ellipse.data();

    for (int i{0}; i < data.size(); ++i)
    {
        parameter_blocks[0] = &parameters->correspondences[i];
        problem->AddResidualBlock(DataTerm::create(data[i]), nullptr, parameter_blocks);
    }

    // set local parameterization for ellipse params
    set_local_parameterization(problem, parameters);

}

void define_ordering(ceres::Solver::Options *options, Parameters *parameters)
{
    ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering;
    for (int i{0}; i < parameters->correspondences.size(); ++i)
        ordering->AddElementToGroup(&(parameters->correspondences(i)), 0);

    ordering->AddElementToGroup(&(parameters->ellipse(0)), 1);
    
    options->linear_solver_ordering.reset(ordering);
}


int main()
{
    // generate data
    constexpr int n_data_points{100};
    constexpr double fraction_of_circumference{1.3};
    Ellipse ellipse_gt;
    std::shared_ptr<std::vector<Point>> data_ptr = ellipse_gt.generate_points_on_boundary(n_data_points, fraction_of_circumference);

    // init parameters (correspondences and ellipse params)
    Parameters parameters(*data_ptr, fraction_of_circumference);
    VectorXd initial_ellipse_params = parameters.ellipse;

    // construct problem
    ceres::Problem problem;
    construct_problem(*data_ptr, &parameters, &problem);

    // options
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    define_ordering(&options, &parameters);

    // solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << std::setprecision(4);
    std::cout << "initial params\tfinal params\tgroundtruth params\n"
              << initial_ellipse_params(0) << "\t\t" << parameters.ellipse(0) << "\t\t" << ellipse_gt.a  << "\n"
              << initial_ellipse_params(1) << "\t\t" << parameters.ellipse(1) << "\t\t" << ellipse_gt.b  << "\n"
              << initial_ellipse_params(2) << "\t\t" << parameters.ellipse(2) << "\t\t" << ellipse_gt.x0 << "\n"
              << initial_ellipse_params(3) << "\t\t" << parameters.ellipse(3) << "\t\t" << ellipse_gt.y0 << "\n"
              << initial_ellipse_params(4) << "\t\t" << parameters.ellipse(4) << "\t\t" << ellipse_gt.r  << "\n";

    return 0;
}
