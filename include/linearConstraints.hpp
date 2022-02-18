#ifndef LINEAR_CONSTRAINTS_HPP
#define LINEAR_CONSTRAINTS_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
#include <CGAL/Gmpzf.h>

/**
 * Helper class for representing linear constraints of the form `A * x >= b`.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/18/2022
 */
using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

/**
 * A class that implements a set of linear constraints among a set of
 * variables, `A * x >= b`.
 */
class LinearConstraints
{
    private:
        int D;                /** Number of variables.               */ 
        int N;                /** Number of constraints.             */
        MatrixXd A;           /** Matrix of constraint coefficients. */ 
        VectorXd b;           /** Matrix of constraint values.       */ 
        Program nearest_L2;   /** Quadratic program for nearest point queries. */

    public:
        /**
         * Empty constructor. 
         */
        LinearConstraints() : nearest_L2(CGAL::LARGER, false, 0.0, false, 0.0) 
        {
            this->N = 0;
            this->D = 0;
            this->A = MatrixXd::Zero(0, 0);
            this->b = VectorXd::Zero(0);
        }

        /**
         * Constructor that sets each variable to between the given lower 
         * and upper bounds.
         *
         * @param D     Number of variables. 
         * @param lower Lower bound for all variables. 
         * @param upper Upper bound for all variables. 
         */
        LinearConstraints(const int D, const double lower, const double upper)
            : nearest_L2(CGAL::LARGER, false, 0.0, false, 0.0)  
        {
            // Each variable has two constraints, one specifying a lower bound
            // and another specifying an upper bound  
            this->N = 2 * D;
            this->D = D;
            this->A.resize(this->N, this->D);
            this->b.resize(this->N);
            for (unsigned i = 0; i < this->D; ++i)
            {
                VectorXd v = VectorXd::Zero(this->D);
                v(i) = 1.0;
                this->A.row(i) = v.transpose();
                this->A.row(this->D + i) = -v.transpose();
                this->b(i) = lower;
                this->b(this->D + i) = -upper;
            }

            // Update internal quadratic program with inequalities implied by 
            // the given bounds 
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2.set_a(j, i, this->A(i, j));
                this->nearest_L2.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2.set_d(i, i, 2.0);
                this->nearest_L2.set_c(i, 0.0);
            }
            this->nearest_L2.set_c0(0.0);
        } 

        /**
         * Constructor with matrix and vector specifying the constraints.
         *
         * @param A Left-hand matrix in the constraints.
         * @param b Right-hand vector in the constraints.  
         */
        LinearConstraints(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
            : nearest_L2(CGAL::LARGER, false, 0.0, false, 0.0) 
        {
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

            // Update internal quadratic program with given matrix and vector
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2.set_a(j, i, this->A(i, j));
                this->nearest_L2.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2.set_d(i, i, 2.0);
                this->nearest_L2.set_c(i, 0.0);
            }
            this->nearest_L2.set_c0(0.0);
        }

        /**
         * Trivial destructor.
         */
        ~LinearConstraints()
        {
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * @param filename Path to file containing the polytope constraints. 
         */
        void parse(const std::string filename)
        {
            unsigned D = 0;
            unsigned N = 0;
            MatrixXd A(N, D);
            VectorXd b(N);
            
            std::string line;
            std::ifstream infile(filename); 
            if (infile.is_open())
            {
                while (std::getline(infile, line))
                {
                    // Accumulate the entries in each line ...
                    std::stringstream ss(line);
                    std::string token;
                    std::vector<double> row;
                    N++;
                    while (std::getline(ss, token, ' '))
                        row.push_back(std::stod(token));

                    // If this is the first row being parsed, get the number 
                    // of columns in constraint matrix 
                    if (D == 0) D = row.size() - 1;

                    // Add the new constraint, with column 0 specifying the 
                    // constant term and the remaining columns specifying the
                    // linear coefficients:
                    //
                    // a0 + a1*x1 + a2*x2 + ... + aN*xN >= 0
                    //
                    A.conservativeResize(N, D);
                    b.conservativeResize(N);
                    for (unsigned i = 1; i < row.size(); ++i)
                        A(N-1, i-1) = row[i];
                    b(N-1) = -row[0];
                }
                infile.close();
            }
            else
                throw std::invalid_argument("Specified file does not exist");

            // Update internal quadratic program with given matrix and vector
            this->A = A;
            this->b = b;
            this->N = N;
            this->D = D;
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2.set_a(j, i, this->A(i, j));
                this->nearest_L2.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2.set_d(i, i, 2.0);
                this->nearest_L2.set_c(i, 0.0);
            }
            this->nearest_L2.set_c0(0.0);
        }

        /**
         * Update `this->A` and `this->b` to the given matrix and vector. 
         *
         * @param A Left-hand matrix in the constraints. 
         * @param b Right-hand matrix in the constraints. 
         */
        void setAb(const Ref<const MatrixXd>& A, const Ref<const VectorXd>& b)
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

            // Update internal quadratic program with given matrix and vector
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    this->nearest_L2.set_a(j, i, this->A(i, j));
                this->nearest_L2.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
            {
                this->nearest_L2.set_d(i, i, 2.0);
                this->nearest_L2.set_c(i, 0.0);
            }
            this->nearest_L2.set_c0(0.0);
        }

        /**
         * Return `this->A`.
         */
        MatrixXd getA()
        {
            return this->A;
        }

        /**
         * Return `this->b`.
         */
        VectorXd getb()
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
         * `this->A * x >= this->b`) by the given query vector.
         *
         * @param x Query vector.
         * @returns True if `this->A * x >= this->b`, false otherwise.  
         */
        bool check(const Ref<const VectorXd>& x)
        {
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() >= (this->b).array()).all();
        }

        /**
         * Return a boolean vector indicating which constraints are active
         * (i.e., which constraints are satisfied as equalities) for the 
         * given query vector.
         *
         * @param x Query vector.
         * @returns Boolean vector indicating which constraints are active. 
         */
        Matrix<bool, Dynamic, 1> active(const Ref<const VectorXd>& x)
        {
            if (x.size() != this->D)
            {
                std::stringstream ss;
                ss << "Dimensions of A and x do not match: ("
                   << this->A.rows() << "," << this->A.cols()
                   << ") vs. " << x.size() << std::endl;
                throw std::invalid_argument(ss.str());
            }
            return ((this->A * x).array() == (this->b).array());
        }

        /**
         * Return the nearest point to the given query vector, with respect 
         * to L2 (Euclidean) distance, that satisfies the constraints.
         *
         * @param x Query vector. 
         * @returns Vector nearest to query that satisfies the constraints.  
         */
        VectorXd nearestL2(const Ref<const VectorXd>& x)
        {
            // First check that x itself satisfies the constraints
            if (this->check(x)) return x;

            // Otherwise, solve the quadratic program for the nearest point to x
            for (unsigned i = 0; i < this->D; ++i)
                this->nearest_L2.set_c(i, -2.0 * x(i));
            Solution solution = CGAL::solve_quadratic_program(this->nearest_L2, ET());
            if (solution.is_infeasible())
                throw std::runtime_error("Quadratic program is infeasible");
            else if (solution.is_unbounded())
                throw std::runtime_error("Quadratic program is unbounded");
            else if (!solution.is_optimal())
                throw std::runtime_error("Failed to compute optimal solution");

            // Collect the values of the solution into a VectorXd
            VectorXd y = VectorXd::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = CGAL::to_double(*it);
                i++;
            }
            return y;
        }

        /**
         * Return the solution to the given linear program, with the feasible 
         * region given by the stored constraints (`this->A * x >= this->b`).
         *
         * The linear program seeks to *minimize* the given objective function. 
         *
         * @param obj Vector of length `this->D` encoding the objective function.
         * @param c0  Constant term of the objective function.
         * @returns   Vector solution to the given linear program.  
         */
        VectorXd solveLinearProgram(const Ref<const VectorXd>& obj, const double c0)
        {
            // Instantiate the linear program ... 
            Program program(CGAL::LARGER, false, 0.0, false, 0.0);
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program.set_a(j, i, this->A(i, j));
                program.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
                program.set_c(i, obj(i)); 
            program.set_c0(c0);

            // ... and (try to) solve it ... 
            Solution solution = CGAL::solve_quadratic_program(program, ET());
            if (solution.is_infeasible())
                throw std::runtime_error("Quadratic program is infeasible");
            else if (solution.is_unbounded())
                throw std::runtime_error("Quadratic program is unbounded");
            else if (!solution.is_optimal())
                throw std::runtime_error("Failed to compute optimal solution");

            // ... and return the solution vector
            VectorXd y = VectorXd::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = CGAL::to_double(*it);
                i++;
            }
            return y;
        }

        /**
         * Determine whether the k-th stored constraint is redundant.
         *
         * @param k Index (`0 <= k <= this->N - 1`) of constraint to be tested
         *          for redundancy. 
         * @returns True if the constraint is redundant, false otherwise.   
         */
        bool isRedundant(const int k)
        {
            // Instantiate the linear program (exclude the k-th constraint and 
            // set the (negative of the) k-th constraint as the objective
            // function to be minimized) ... 
            Program program(CGAL::LARGER, false, 0.0, false, 0.0);
            for (unsigned i = 0; i < k; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program.set_a(j, i, this->A(i, j));
                program.set_b(i, this->b(i));
            }
            for (unsigned i = k + 1; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program.set_a(j, i, this->A(i, j));
                program.set_b(i, this->b(i));
            }
            for (unsigned i = 0; i < this->D; ++i)
                program.set_c(i, -this->A(k, i)); 
            program.set_c0(0);

            // ... and (try to) solve it 
            Solution solution = CGAL::solve_quadratic_program(program, ET());

            // Return whether a feasible solution could not be found 
            return (solution.is_infeasible() || solution.is_unbounded() || !solution.is_optimal());
        }
};

#endif
