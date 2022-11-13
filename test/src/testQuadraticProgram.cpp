#include <cmath>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../../include/quadraticProgram.hpp"

/**
 * Test module for quadratic programming solver functions.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/13/2022
 */
using namespace Eigen;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType;

/**
 * Nocedal and Wright, Example 16.2 (an equality-constrained convex quadratic
 * program).
 */
template <typename T>
Matrix<T, Dynamic, 1> equalityConstrainedConvexQP1()
{
    Matrix<T, Dynamic, Dynamic> G(3, 3); 
    G << 6, 2, 1,
         2, 5, 2,
         1, 2, 4;
    Matrix<T, Dynamic, 1> c(3);
    c << -8, -3, -3;
    Matrix<T, Dynamic, Dynamic> A(2, 3);
    A << 1, 0, 1,
         0, 1, 1;
    Matrix<T, Dynamic, 1> b(2); 
    b << 3, 0;

    return solveEqualityConstrainedConvexQuadraticProgram<T>(G, c, A, b);
}

TEST_CASE("Equality-constrained convex QP with doubles")
{
    VectorXd solution = equalityConstrainedConvexQP1<double>(); 
    const double tol = 1e-8;
    VectorXd correct_solution(3); 
    correct_solution << 2, -1, 1;
    REQUIRE((solution - correct_solution).norm() < tol); 
}

TEST_CASE("Equality-constrained convex QP with MPFR floats")
{
    Matrix<PreciseType, Dynamic, 1> solution = equalityConstrainedConvexQP1<PreciseType>(); 
    const PreciseType tol = 1e-50;
    Matrix<PreciseType, Dynamic, 1> correct_solution(3); 
    correct_solution << 2, -1, 1;
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Equality-constrained convex QP with GMP rationals")
{
    Matrix<mpq_rational, Dynamic, 1> solution = equalityConstrainedConvexQP1<mpq_rational>(); 
    Matrix<mpq_rational, Dynamic, 1> correct_solution(3); 
    correct_solution << 2, -1, 1;
    REQUIRE(solution == correct_solution);
}

/**
 * Nocedal and Wright, Example 16.4 (a convex quadratic program).
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 1>, bool> convexQP1(const T tol, const int max_iter)
{
    // Objective = (x - 1)^2 + (y - 2.5)^2 = x^2 - 2x + 1 + y^2 - 5y + 6.25
    Matrix<T, Dynamic, Dynamic> G = 2 * Matrix<T, Dynamic, Dynamic>::Identity(2, 2);
    Matrix<T, Dynamic, 1> c(2);
    c << -2, -5;

    // Constraints are given by:
    //  x - 2y >= -2
    // -x - 2y >= -6
    // -x + 2y >= -2
    //  x >= 0
    //  x >= 0
    Matrix<T, Dynamic, Dynamic> A(5, 2); 
    A <<  1, -2,
         -1, -2,
         -1,  2,
          1,  0,
          0,  1;
    Matrix<T, Dynamic, 1> b(5); 
    b << -2, -6, -2, 0, 0;

    // Initialize solver at (2, 0)
    Matrix<T, Dynamic, 1> x_init(2); 
    x_init << 2, 0;

    return solveConvexQuadraticProgram<T>(G, c, A, b, x_init, tol, max_iter, false);
}

TEST_CASE("Convex QP 1 with doubles")
{
    const double tol = 1e-8; 
    const int max_iter = 100;
    auto result = convexQP1<double>(tol, max_iter);
    VectorXd solution = result.first.head(2); 
    VectorXd correct_solution(2); 
    correct_solution << 1.4, 1.7;
    REQUIRE(result.second);
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 1 with MPFR floats")
{
    const PreciseType tol = 1e-50;
    const int max_iter = 100;
    auto result = convexQP1<PreciseType>(tol, max_iter);
    Matrix<PreciseType, Dynamic, 1> solution = result.first.head(2);
    Matrix<PreciseType, Dynamic, 1> correct_solution(2);
    correct_solution << PreciseType("1.4"), PreciseType("1.7");
    REQUIRE(result.second);
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 1 with GMP rationals")
{
    const mpq_rational tol = 0; 
    const int max_iter = 100;
    auto result = convexQP1<mpq_rational>(tol, max_iter);
    Matrix<mpq_rational, Dynamic, 1> solution = result.first.head(2); 
    Matrix<mpq_rational, Dynamic, 1> correct_solution(2); 
    mpq_rational a(14, 10), b(17, 10); 
    correct_solution << a, b;
    REQUIRE(result.second);
    REQUIRE(solution == correct_solution);
}

/**
 * Example from CGAL documentation (Section 3.1).
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 1>, bool> convexQP2(const T tol, const int max_iter)
{
    // Objective = x^2 + 4y^2 - 32y
    Matrix<T, Dynamic, Dynamic> G(2, 2); 
    G << 1, 0,
         0, 4;
    G *= 2; 
    Matrix<T, Dynamic, 1> c(2); 
    c << 0, -32;

    // Constraints are given by: 
    //  x +  y <= 7  (or -x - y >= -7)
    // -x + 2y <= 4  (or x - 2y >= -4)
    //       y <= 4  (or -y >= -4)
    Matrix<T, Dynamic, Dynamic> A(3, 2);
    A << -1, -1,
          1, -2,
          0, -1;
    Matrix<T, Dynamic, 1> b(3); 
    b << -7, -4, -4;

    // Initialize solver at (2, 0)
    Matrix<T, Dynamic, 1> x_init(2); 
    x_init << 2, 0;
    
    return solveConvexQuadraticProgram<T>(G, c, A, b, x_init, tol, max_iter, false);
}

TEST_CASE("Convex QP 2 with doubles")
{
    const double tol = 1e-8; 
    const int max_iter = 100;
    auto result = convexQP2<double>(tol, max_iter);
    VectorXd solution = result.first.head(2);
    VectorXd correct_solution(2); 
    correct_solution << 2, 3;
    REQUIRE(result.second);
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 2 with MPFR floats")
{
    const PreciseType tol = 1e-50; 
    const int max_iter = 100;
    auto result = convexQP2<PreciseType>(tol, max_iter);
    Matrix<PreciseType, Dynamic, 1> solution = result.first.head(2);
    Matrix<PreciseType, Dynamic, 1> correct_solution(2); 
    correct_solution << 2, 3;
    REQUIRE(result.second);
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 2 with GMP rationals")
{
    const mpq_rational tol = 0; 
    const int max_iter = 100;
    auto result = convexQP2<mpq_rational>(tol, max_iter);
    Matrix<mpq_rational, Dynamic, 1> solution = result.first.head(2);
    Matrix<mpq_rational, Dynamic, 1> correct_solution(2); 
    correct_solution << 2, 3;
    REQUIRE(result.second);
    REQUIRE(solution == correct_solution);
}

/**
 * Numerical example from Northwestern University Process Optimization Open
 * Textbook (https://optimization.mccormick.northwestern.edu/index.php/Quadratic_programming).
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 1>, bool> convexQP3(const T tol, const int max_iter)
{
    // Objective = 3x^2 + y^2 + 2xy + x + 6y + 2
    Matrix<T, Dynamic, Dynamic> G(2, 2);
    G << 3, 1,
         1, 1;
    G *= 2;
    Matrix<T, Dynamic, 1> c(2);
    c << 1, 6;

    // Constraints are given by: 
    // 2x + 3y >= 4
    // x >= 0
    // y >= 0
    Matrix<T, Dynamic, Dynamic> A(3, 2); 
    A << 2, 3,
         1, 0,
         0, 1;
    Matrix<T, Dynamic, 1> b(3); 
    b << 4, 0, 0;

    // Initialize solver at (2, 0)
    Matrix<T, Dynamic, 1> x_init(2);
    x_init << 2, 0;
    
    return solveConvexQuadraticProgram<T>(G, c, A, b, x_init, tol, max_iter, false);
}

TEST_CASE("Convex QP 3 with doubles")
{
    const double tol = 1e-8; 
    const int max_iter = 100;
    auto result = convexQP3<double>(tol, max_iter);
    VectorXd solution = result.first.head(2);
    VectorXd correct_solution(2); 
    correct_solution << 0.5, 1;
    REQUIRE(result.second); 
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 3 with MPFR floats")
{
    const PreciseType tol = 1e-50; 
    const int max_iter = 100;
    auto result = convexQP3<PreciseType>(tol, max_iter);
    Matrix<PreciseType, Dynamic, 1> solution = result.first.head(2);
    Matrix<PreciseType, Dynamic, 1> correct_solution(2); 
    correct_solution << PreciseType("0.5"), 1;
    REQUIRE(result.second);
    REQUIRE((solution - correct_solution).norm() < tol);
}

TEST_CASE("Convex QP 3 with GMP rationals")
{
    const mpq_rational tol = 0;
    const int max_iter = 100;
    auto result = convexQP3<mpq_rational>(tol, max_iter);  
    Matrix<mpq_rational, Dynamic, 1> solution = result.first.head(2);
    Matrix<mpq_rational, Dynamic, 1> correct_solution(2); 
    correct_solution << mpq_rational(1, 2), 1;
    REQUIRE(result.second);
    REQUIRE(solution == correct_solution);
}
