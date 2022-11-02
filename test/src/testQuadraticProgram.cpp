#define BOOST_TEST_MODULE testQuadraticProgram
#define BOOST_TEST_DYN_LINK
#include <cmath>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/quadraticProgram.hpp"

/**
 * Test module for quadratic programming solver functions.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/2/2022
 */
using namespace Eigen;

/**
 * Nocedal and Wright, Example 16.2 (an equality-constrained convex quadratic
 * program).
 */
BOOST_AUTO_TEST_CASE(TEST_EQUALITY_CONSTRAINED_CONVEX_QP_1)
{
    MatrixXd G(3, 3); 
    G << 6, 2, 1,
         2, 5, 2,
         1, 2, 4;
    VectorXd c(3);
    c << -8, -3, -3;
    MatrixXd A(2, 3);
    A << 1, 0, 1,
         0, 1, 1;
    VectorXd b(2); 
    b << 3, 0;

    VectorXd solution = solveEqualityConstrainedConvexQuadraticProgram<double>(G, c, A, b);
    const double tol = 1e-8;
    VectorXd correct_solution(3); 
    correct_solution << 2, -1, 1;
    BOOST_TEST((solution - correct_solution).norm() < tol);
}

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
BOOST_AUTO_TEST_CASE(TEST_CONVEX_QP_1)
{
    // Objective = (x(0) - 1)^2 + (x(1) - 2.5)^2
    MatrixXd G = 2 * MatrixXd::Identity(2, 2);
    VectorXd c(2);
    c << -2, -5;

    // Constraints are given by:
    //  x(0) - 2 * x(1) >= -2
    // -x(0) - 2 * x(1) >= -6
    // -x(0) + 2 * x(1) >= -2
    //  x(0) >= 0
    //  x(1) >= 0
    MatrixXd A(5, 2); 
    A <<  1, -2,
         -1, -2,
         -1,  2,
          1,  0,
          0,  1;
    VectorXd b(5); 
    b << -2, -6, -2, 0, 0;

    // Initialize solver at (0, 0)
    VectorXd x_init = VectorXd::Zero(2);
    Polytopes::LinearConstraints<double>* constraints = new Polytopes::LinearConstraints<double>(
        Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
    );
    const double tol = 1e-8; 
    const int max_iter = 10; 
    VectorXd solution = solveConvexQuadraticProgram<double>(G, c, constraints, x_init, tol, max_iter);
    VectorXd correct_solution(2); 
    correct_solution << 1.4, 1.7;
    BOOST_TEST((solution - correct_solution).norm() < tol);

    delete constraints;
}

