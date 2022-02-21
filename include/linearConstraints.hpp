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
 * Helper class for representing linear constraints of the form `A * x <=/>= b`.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/21/2022
 */
using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

namespace Polytopes {

enum InequalityType
{
    LessThanOrEqualTo,
    GreaterThanOrEqualTo
};

/**
 * A class that implements a set of linear constraints among a set of
 * variables, `A * x <=/>= b`.
 */
class LinearConstraints
{
    protected:
        InequalityType type;    /** Inequality type.                   */
        int D;                  /** Number of variables.               */ 
        int N;                  /** Number of constraints.             */
        MatrixXd A;             /** Matrix of constraint coefficients. */ 
        VectorXd b;             /** Matrix of constraint values.       */ 
        Program nearest_L2;     /** Quadratic program for nearest-point queries. */

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * This is the internal protected method used in `parse(filename)` and
         * `parse(filename, type)`. 
         *
         * @param filename Path to file containing the polytope constraints.
         * @param type     Inequality type (not denoted in the file).
         */
        void __parse(const std::string filename, const InequalityType type)
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
                    if (D == 0)
                        D = row.size() - 1;

                    // Add the new constraint, with column 0 specifying the 
                    // constant term and the remaining columns specifying the
                    // linear coefficients:
                    //
                    // a0 + a1*x1 + a2*x2 + ... + aN*xN <=/>= 0
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
            this->type = type; 
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

    public:
        /**
         * Trivial constructor that specifies only the inequality type.
         *
         * @param type Inequality type. 
         */
        LinearConstraints(const InequalityType type)
            : nearest_L2((type == LessThanOrEqualTo ? CGAL::SMALLER : CGAL::LARGER),
                         false, 0.0, false, 0.0) 
        {
            this->type = type; 
            this->N = 0;
            this->D = 0;
            this->A = MatrixXd::Zero(0, 0);
            this->b = VectorXd::Zero(0);
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
        LinearConstraints(const InequalityType type, const int D,
                          const double lower, const double upper)
            : nearest_L2((type == LessThanOrEqualTo ? CGAL::SMALLER : CGAL::LARGER),
                         false, 0.0, false, 0.0)  
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
         * @param type Inequality type. 
         * @param A    Left-hand matrix in the constraints.
         * @param b    Right-hand vector in the constraints.  
         */
        LinearConstraints(const InequalityType type, const Ref<const MatrixXd>& A,
                          const Ref<const VectorXd>& b)
            : nearest_L2((type == LessThanOrEqualTo ? CGAL::SMALLER : CGAL::LARGER),
                         false, 0.0, false, 0.0) 
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
         * Update `this->type`. 
         *
         * @param type Inequality type. 
         */
        void setInequalityType(const InequalityType type)
        {
            this->type = type; 
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
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * @param filename Path to file containing the polytope constraints.
         */
        void parse(const std::string filename)
        {
            this->parse(filename, this->type); 
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces
         * (inequalities), read in the constraint matrix and vector.
         *
         * @param filename Path to file containing the polytope constraints.
         * @param type     Inequality type (not denoted in the file).
         */
        void parse(const std::string filename, const InequalityType type)
        {
            this->parse(filename, type); 
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
         * `this->A * x <=/>= this->b`) by the given query vector.
         *
         * @param x Query vector.
         * @returns True if `this->A * x <=/>= this->b`, false otherwise.  
         */
        bool query(const Ref<const VectorXd>& x)
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
            else
                return ((this->A * x).array() >= (this->b).array()).all();
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
            if (this->query(x)) return x;

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
         * region given by the stored constraints (`this->A * x <=/>= this->b`).
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
            Program program(
                (this->type == LessThanOrEqualTo ? CGAL::SMALLER : CGAL::LARGER),
                false, 0.0, false, 0.0
            );
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
         * Determine whether the `k`-th stored constraint is redundant.
         *
         * @param k Index (`0 <= k <= this->N - 1`) of constraint to be tested
         *          for redundancy. 
         * @returns True if the constraint is redundant, false otherwise.   
         */
        bool isRedundant(const int k)
        {
            // Instantiate the linear program ...
            Program program(
                (this->type == LessThanOrEqualTo ? CGAL::SMALLER : CGAL::LARGER),
                false, 0.0, false, 0.0
            );
            // ... excluding the k-th constraint ... 
            for (unsigned i = 0; i < k; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program.set_a(j, i, this->A(i, j));
                program.set_b(i, this->b(i));
            }
            for (unsigned i = k + 1; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program.set_a(j, i - 1, this->A(i, j));
                program.set_b(i - 1, this->b(i));
            }
            // ... and if the constraints are less-than-or-equal-to, setting
            // the *negative* of the k-th constraint as the objective function ...
            if (this->type == LessThanOrEqualTo)
            {
                for (unsigned i = 0; i < this->D; ++i)
                    program.set_c(i, -this->A(k, i));
            }
            // ... and otherwise setting the k-th constraint as the objective 
            // function ...
            else
            {
                for (unsigned i = 0; i < this->D; ++i)
                    program.set_c(i, this->A(k, i)); 
            } 
            program.set_c0(0);

            // ... and (try to) solve it
            Solution solution = CGAL::solve_quadratic_program(program, ET());

            // If the solution is infeasible, then return false (*not redundant*)
            if (solution.is_infeasible() || solution.is_unbounded() || !solution.is_optimal())
                return false; 

            // If the solution is feasible, then check that the solution satisfies
            // the excised constraint
            VectorXd y = VectorXd::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = CGAL::to_double(*it);
                i++;
            }
            return (this->A.row(k) * y >= this->b(k)); 
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
        }
};

}   // namespace Polytopes

#endif
