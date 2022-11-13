/**
 * Helper class for representing linear constraints of the form `A * x <=/==/>= b`
 * with rational scalars.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/13/2022
 */

#ifndef LINEAR_CONSTRAINTS_RATIONAL_COEFFICIENTS_HPP
#define LINEAR_CONSTRAINTS_RATIONAL_COEFFICIENTS_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <CGAL/Gmpzf.h>
#include <boost/multiprecision/gmp.hpp>
#include "quadraticProgram.hpp"

using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;
using boost::multiprecision::mpq_rational;
typedef Matrix<mpq_rational, Dynamic, Dynamic> MatrixXr;
typedef Matrix<mpq_rational, Dynamic, 1>       VectorXr;

namespace Polytopes {

enum InequalityType
{
    LessThanOrEqualTo,
    GreaterThanOrEqualTo,
    IsEqualTo
};

CGAL::Comparison_result cgalInequalityType(const InequalityType type)
{
    if (type == LessThanOrEqualTo)
        return CGAL::SMALLER; 
    else if (type == GreaterThanOrEqualTo)
        return CGAL::LARGER; 
    else 
        return CGAL::EQUAL; 
}

/**
 * A class that implements a set of linear constraints among a set of
 * variables, `A * x <=/==/>= b`.
 */
class LinearConstraints
{
    protected:
        InequalityType type;              /** Inequality type.                   */
        int D;                            /** Number of variables.               */ 
        int N;                            /** Number of constraints.             */
        MatrixXr A;                       /** Matrix of constraint coefficients. */ 
        VectorXr b;                       /** Matrix of constraint values.       */ 
        Program* approx_nearest_L2;       /** Quadratic program for approximate nearest-point queries. */

        /**
         * Update the internal nearest-point-by-L2-distance quadratic program
         * with coefficients given by the current constraint matrix and vector. 
         */
        void updateApproxNearestL2()
        {
            // Convert all coefficients to double scalars 
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->approx_nearest_L2->set_a(j, i, static_cast<double>(this->A(i, j)));
                this->approx_nearest_L2->set_b(i, static_cast<double>(this->b(i)));
                this->approx_nearest_L2->set_r(i, cgalInequalityType(this->type)); 
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->approx_nearest_L2->set_d(i, i, 2);
                this->approx_nearest_L2->set_c(i, 0);
            }
            this->approx_nearest_L2->set_c0(0);
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * This is the internal protected method used in `parse(filename)` and
         * `parse(filename, type)`. 
         *
         * @param filename Path to file containing the polytope constraints.
         * @param type     Inequality type of the constraints in the file. 
         *                 (If `type` does not match `this->type`, then the
         *                 constraints are converted to `this->type`.)
         */
        void __parse(const std::string filename, const InequalityType type)
        {
            unsigned D = 0;
            unsigned N = 0;
            MatrixXr A(N, D);
            VectorXr b(N);
            
            std::string line;
            std::ifstream infile(filename); 
            if (infile.is_open())
            {
                while (std::getline(infile, line))
                {
                    // Accumulate the entries in each line ...
                    std::stringstream ss(line);
                    std::string token;
                    std::vector<mpq_rational> row;
                    N++;
                    while (std::getline(ss, token, ' '))
                    {
                        // Parse each line as a double, then convert to the
                        // scalar type of choice
                        row.push_back(mpq_rational(token));
                    }

                    // If this is the first row being parsed, get the number 
                    // of columns in constraint matrix 
                    if (D == 0)
                        D = row.size() - 1;

                    // Add the new constraint, with column 0 specifying the 
                    // constant term and the remaining columns specifying the
                    // linear coefficients:
                    //
                    // a0 + a1*x1 + a2*x2 + ... + aN*xN <=/==/>= 0
                    //
                    A.conservativeResize(N, D);
                    b.conservativeResize(N);
                    if (type != this->type)
                    {
                        for (unsigned i = 1; i < row.size(); ++i)
                            A(N-1, i-1) = -row[i];
                        b(N-1) = row[0];
                    }
                    else 
                    {
                        for (unsigned i = 1; i < row.size(); ++i)
                            A(N-1, i-1) = row[i]; 
                        b(N-1) = -row[0];
                    }
                }
                infile.close();
            }
            else
                throw std::invalid_argument("Specified file does not exist");

            this->A = A;
            this->b = b;
            this->N = N;
            this->D = D;
            this->updateApproxNearestL2();
        }

    public:
        /**
         * Trivial constructor that specifies only the inequality type.
         *
         * @param type Inequality type. 
         */
        LinearConstraints(const InequalityType type)
        {
            this->type = type; 
            this->N = 0;
            this->D = 0;
            this->A = MatrixXr::Zero(0, 0);
            this->b = VectorXr::Zero(0);
            this->approx_nearest_L2 = new Program(cgalInequalityType(type), false, 0.0, false, 0.0);
        }

        /**
         * Constructor that sets each variable to between the given lower 
         * and upper bounds.
         *
         * @param type  Inequality type. 
         * @param D     Number of variables. 
         * @param lower Lower bound for all variables. 
         * @param upper Upper bound for all variables. 
         */
        LinearConstraints(const InequalityType type, const int D, const mpq_rational lower,
                          const mpq_rational upper)
        {
            // Each variable has two constraints, one specifying a lower bound
            // and another specifying an upper bound
            this->type = type;  
            this->N = 2 * D;
            this->D = D;
            this->A.resize(this->N, this->D);
            this->b.resize(this->N);
            for (unsigned i = 0; i < this->D; ++i)
            {
                VectorXr v = VectorXr::Zero(this->D);
                v(i) = 1;
                if (this->type == LessThanOrEqualTo)
                {
                    this->A.row(i) = -v.transpose();
                    this->A.row(this->D + i) = v.transpose();
                    this->b(i) = -lower;
                    this->b(this->D + i) = upper;
                }
                else
                {
                    this->A.row(i) = v.transpose();
                    this->A.row(this->D + i) = -v.transpose();
                    this->b(i) = lower;
                    this->b(this->D + i) = -upper;
                }
            }
            this->approx_nearest_L2 = new Program(cgalInequalityType(type), false, 0.0, false, 0.0);
            this->updateApproxNearestL2();
        } 

        /**
         * Constructor with matrix and vector specifying the constraints.
         *
         * @param type Inequality type. 
         * @param A    Left-hand matrix in the constraints.
         * @param b    Right-hand vector in the constraints.  
         */
        LinearConstraints(const InequalityType type, const Ref<const MatrixXr>& A,
                          const Ref<const VectorXr>& b)
        {
            this->type = type; 
            this->A = A;
            this->b = b;
            this->N = this->A.rows();
            this->D = this->A.cols();
            if (this->b.size() != this->N)
            {
                std::stringstream ss;
                ss << "Dimensions of A and b do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << this->b.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            this->approx_nearest_L2 = new Program(cgalInequalityType(type), false, 0.0, false, 0.0);
            this->updateApproxNearestL2();
        }

        /**
         * Trivial destructor.
         */
        ~LinearConstraints()
        {
            delete this->approx_nearest_L2; 
        }

        /**
         * Set `this->type` to the given inequality type. 
         *
         * This method *does not update* `this->A` or `this->b`, but does 
         * update the nearest-point-by-L2-distance quadratic program to 
         * reflect the new inequality type.
         *
         * @param type Inequality type.  
         */
        void setInequalityType(const InequalityType type) 
        {
            this->type = type;
            this->updateApproxNearestL2(); 
        }

        /**
         * Switch the inequality type of the stored constraints.
         *
         * This negates both `this->A` and `this->b`, and also updates the 
         * nearest-point-by-L2-distance quadratic program.  
         */
        void switchInequalityType()
        {
            if (this->type == InequalityType::GreaterThanOrEqualTo)
                this->type = InequalityType::LessThanOrEqualTo; 
            else if (this->type == InequalityType::LessThanOrEqualTo)
                this->type = InequalityType::GreaterThanOrEqualTo;
            this->A *= -1; 
            this->b *= -1; 
            this->updateApproxNearestL2(); 
        }

        /**
         * Update `this->A` and `this->b` to the given matrix and vector. 
         *
         * @param A Left-hand matrix in the constraints. 
         * @param b Right-hand matrix in the constraints. 
         */
        void setAb(const Ref<const MatrixXr>& A, const Ref<const VectorXr>& b)
        {
            this->A = A;
            this->b = b;
            this->N = A.rows();
            this->D = A.cols();
            if (this->b.size() != this->N)
            {
                std::stringstream ss;
                ss << "Dimensions of A and b do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << this->b.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            this->updateApproxNearestL2(); 
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * @param filename Path to file containing the polytope constraints.
         */
        void parse(const std::string filename)
        {
            try
            {
                this->__parse(filename, this->type);
            }
            catch (const std::invalid_argument& e) 
            {
                throw; 
            }
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * @param filename Path to file containing the polytope constraints.
         * @param type     Inequality type of the constraints in the file. 
         *                 (If `type` does not match `this->type`, then the
         *                 constraints are converted to `this->type`.)
         */
        void parse(const std::string filename, const InequalityType type)
        {
            try
            {
                this->__parse(filename, type);
            }
            catch (const std::invalid_argument& e)
            {
                throw; 
            }
        }

        /**
         * Return `this->type`.
         */
        InequalityType getInequalityType()
        {
            return this->type; 
        }

        /**
         * Return `this->A`.
         */
        MatrixXr getA()
        {
            return this->A;
        }

        /**
         * Return `this->b`.
         */
        VectorXr getb()
        {
            return this->b;
        }

        /**
         * Return `this->N`.
         */
        int getN()
        {
            return this->N; 
        }

        /**
         * Return `this->D`. 
         */
        int getD()
        {
            return this->D; 
        }

        /** 
         * Return true if the constraints were satisfied (that is, if
         * `this->A * x <=/==/>= this->b`) by the given query vector.
         *
         * @param x Query vector.
         * @returns True if `this->A * x <=/==/>= this->b`, false otherwise.  
         */
        bool query(const Ref<const VectorXr>& x)
        {
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            if (this->type == LessThanOrEqualTo)
                return ((this->A * x).array() <= (this->b).array()).all();
            else if (this->type == GreaterThanOrEqualTo)
                return ((this->A * x).array() >= (this->b).array()).all();
            else 
                return ((this->A * x).array() == (this->b).array()).all(); 
        }

        /**
         * Remove the `k`-th constraint.
         *
         * @param k Index (`0 <= k <= this->N - 1`) of constraint to be removed. 
         */
        void removeConstraint(const int k)
        {
            std::vector<int> indices; 
            for (unsigned i = 0; i < k; ++i)
                indices.push_back(i);
            for (unsigned i = k + 1; i < this->N; ++i)
                indices.push_back(i); 
            this->A = this->A(indices, Eigen::all).eval();
            this->b = this->b(indices).eval();
            this->N--;    
        }

        /**
         * Return a boolean vector indicating which constraints are active
         * (i.e., which constraints are satisfied as equalities) for the 
         * given query vector.
         *
         * @param x Query vector.
         * @returns Boolean vector indicating which constraints are active. 
         */
        Matrix<bool, Dynamic, 1> active(const Ref<const VectorXr>& x)
        {
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() == (this->b).array()).matrix();
        }

        /**
         * Return the approximate nearest point to the given query vector,
         * with respect to L2 (Euclidean) distance, that satisfies the
         * constraints.
         *
         * The input vector may have scalar type other than `mpq_rational`,
         * and the output has type `mpq_rational`.
         * 
         * @param x Query vector. 
         * @returns Vector approximately nearest to query that satisfies the
         *          constraints.  
         */
        template <typename U>
        VectorXr approxNearestL2(const Ref<const Matrix<U, Dynamic, 1> >& x)
        {
            // First check that x itself satisfies the constraints
            VectorXr x_ = x.template cast<mpq_rational>();
            if (this->query(x_)) return x_;

            // Otherwise, solve the quadratic program for the nearest point to x
            // (Must update linear part of objective)
            for (unsigned i = 0; i < this->D; ++i)
                this->approx_nearest_L2->set_c(i, -2.0 * static_cast<double>(x(i)));
            Solution solution = CGAL::solve_quadratic_program(*this->approx_nearest_L2, ET());
            if (solution.is_infeasible())
                throw std::runtime_error("Quadratic program is infeasible");
            else if (solution.is_unbounded())
                throw std::runtime_error("Quadratic program is unbounded");
            else if (!solution.is_optimal())
                throw std::runtime_error("Failed to compute optimal solution");

            // Collect the values of the solution with the desired scalar type
            VectorXr y = VectorXr::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = static_cast<mpq_rational>(CGAL::to_double(*it));
                i++;
            }
            return y;
        }

        /**
         * Return the approximate nearest point to the given query vector,
         * with respect to L2 (Euclidean) distance, that satisfies the
         * constraints.
         *
         * The input vector may have scalar type `U` other than `mpq_rational`,
         * and the output has type `U` as well.
         * 
         * @param x        Query vector.
         * @param x_init   "Initial" vector that satisfies constraints. 
         * @param tol      Tolerance for assessing whether a stepsize is zero.
         * @param max_iter Maximum number of iterations when solving the
         *                 quadratic program.
         * @returns Vector approximately nearest to query that satisfies the
         *          constraints.  
         */
        template <typename U>
        Matrix<U, Dynamic, 1> approxNearestL2(const Ref<const Matrix<U, Dynamic, 1> >& x,
                                              const Ref<const Vector<U, Dynamic, 1> >& x_init,
                                              const U tol, const int max_iter)
        {
            // First check that x itself satisfies the constraints
            VectorXr x_ = x.template cast<mpq_rational>();
            if (this->query(x_)) return x_;

            // Otherwise, solve the quadratic program for the nearest point to x
            // (Must update linear part of objective)
            //
            // The calculations are performed with type U, not mpq_rational
            Matrix<U, Dynamic, Dynamic> G = 2 * Matrix<U, Dynamic, Dynamic>::Identity(this->D, this->D); 
            Matrix<U, Dynamic, 1> c = -2 * x; 
            Matrix<U, Dynamic, Dynamic> A_ = (this->A).template cast<U>();
            Matrix<U, Dynamic, 1> b_ = (this->b).template cast<U>();
            auto result = solveConvexQuadraticProgram<U>(G, c, A_, b_, x_init, tol, max_iter);
            if (!result.second)
            {
                std::stringstream ss;
                ss << "Quadratic program did not converge within given maximum"
                      "number of iterations (" << max_iter << ")" << std::endl;
                throw std::runtime_error(ss.str()); 
            }
            
            return result.first.head(this->D);
        }

        /**
         * Determine whether the `k`-th stored constraint is redundant.
         *
         * If the inequality type is `IsEqualTo`, this is determined by checking
         * if the `k`-th constraint lies in the row space of the matrix given 
         * by the remaining constraints.
         *
         * If the inequality type is not `IsEqualTo`, this is determined by
         * setting up the linear program excluding the `k`-th constraint with
         * the objective function set to the (right-hand-side of the) constraint.
         * If the constraint is redundant, then the optimal objective value should
         * satisfy the constraint.
         *
         * More specifically, if the original constraints are >=, then we 
         * try to *minimize* the `k`-th constraint subject to the others; if 
         * the original constraints are <=, then we try to *maximize* the `k`-th
         * constraint subject to the others.  
         *
         * @param k Index (`0 <= k <= this->N - 1`) of constraint to be tested
         *          for redundancy. 
         * @returns True if the constraint is redundant, false otherwise.   
         */
        bool isRedundant(const int k)
        {
            bool is_redundant = false; 
            if (this->type == IsEqualTo)
            {
                MatrixXr A_(this->N - 1, this->D + 1);
                VectorXr b_(this->D + 1);
                A_(Eigen::seq(0, k - 1), Eigen::seqN(0, this->D)) = this->A(Eigen::seq(0, k - 1), Eigen::all);
                A_(Eigen::seq(k + 1, this->N - 1), Eigen::seqN(0, this->D)) = this->A(Eigen::seq(k + 1, this->N - 1), Eigen::all);
                A_(Eigen::seq(0, k - 1), this->D) = this->b.head(k);
                A_(Eigen::seq(k + 1, this->N - 1), this->D) = this->b.tail(this->N - 1 - k);
                b_.head(this->D) = this->A.row(k);
                b_(this->D) = this->b(k);
                VectorXr solution = A_.transpose().fullPivLu().solve(b_);
                is_redundant = (A_.transpose() * solution == b_); 
            }
            else
            {
                // ----------------------------------------------------------- //
                // Checking feasibility of the original constraints ...        //
                // ----------------------------------------------------------- //
                // Instantiate the linear program with all constraints included ... 
                Program* program = new Program(cgalInequalityType(this->type), false, 0.0, false, 0.0);
                for (unsigned i = 0; i < this->N; ++i)
                {
                    for (unsigned j = 0; j < this->D; ++j)
                        program->set_a(j, i, static_cast<double>(this->A(i, j)));
                    program->set_b(i, static_cast<double>(this->b(i)));
                }
                
                // ... and with zero objective function 
                for (unsigned i = 0; i < this->D; ++i)
                    program->set_c(i, 0); 
                program->set_c0(0);

                // Solve the linear program and check that the program is feasible 
                // to begin with 
                Solution solution = CGAL::solve_quadratic_program(*program, ET());
                delete program; 
                if (solution.is_infeasible())
                    return false; 

                // ----------------------------------------------------------- //
                // Checking redundancy of the k-th constraint ...
                // ----------------------------------------------------------- // 
                // Now instantiate the linear program with the k-th constraint excluded
                Program* subprogram = new Program(cgalInequalityType(this->type), false, 0.0, false, 0.0); 
                for (unsigned i = 0; i < k; ++i)
                {
                    for (unsigned j = 0; j < this->D; ++j)
                        subprogram->set_a(j, i, static_cast<double>(this->A(i, j)));
                    subprogram->set_b(i, static_cast<double>(this->b(i)));
                }
                for (unsigned i = k + 1; i < this->N; ++i)
                {
                    for (unsigned j = 0; j < this->D; ++j)
                        subprogram->set_a(j, i - 1, static_cast<double>(this->A(i, j)));
                    subprogram->set_b(i - 1, static_cast<double>(this->b(i)));
                }

                // If the constraints are >=, set the objective function to the
                // k-th constraint and minimize 
                if (this->type == InequalityType::GreaterThanOrEqualTo)
                {
                    for (unsigned i = 0; i < this->D; ++i)
                        subprogram->set_c(i, static_cast<double>(this->A(k, i)));
                    subprogram->set_c0(0);
                    solution = CGAL::solve_quadratic_program(*subprogram, ET());
                    is_redundant = (
                        !solution.is_infeasible() && solution.is_optimal() &&
                        solution.objective_value() >= static_cast<double>(this->b(k))
                    ); 
                }
                // If the constraints are <=, set the objective function to the
                // *negative* of the k-th constraint and minimize
                else
                {
                    for (unsigned i = 0; i < this->D; ++i)
                        subprogram->set_c(i, -static_cast<double>(this->A(k, i)));
                    subprogram->set_c0(0);
                    solution = CGAL::solve_quadratic_program(*subprogram, ET());
                    is_redundant = (
                        !solution.is_infeasible() && solution.is_optimal() &&
                        solution.objective_value() >= static_cast<double>(-this->b(k))
                    ); 
                }
                delete subprogram;
            }
            return is_redundant;  
        }

        /**
         * Remove all redundant constraints by iterating through them in the 
         * order they are given in `this->A` and `this->b`.   
         */
        void removeRedundantConstraints()
        {
            unsigned i = 0; 
            while (i < this->N) 
            {
                // If the i-th constraint is redundant, remove it and reset 
                // i to zero (start from the beginning of the reduced system)
                if (this->isRedundant(i))
                {
                    this->removeConstraint(i); 
                    i = 0; 
                }
                // Otherwise, keep going 
                else 
                {
                    i++; 
                }
            }
            this->updateApproxNearestL2(); 
        }
};

}   // namespace Polytopes

#endif
