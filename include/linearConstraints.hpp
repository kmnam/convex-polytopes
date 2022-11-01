/**
 * Helper class for representing linear constraints of the form `A * x <=/==/>= b`.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/1/2022
 */

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

using namespace Eigen;
typedef CGAL::Gmpzf ET;
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

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
template <typename T>
class LinearConstraints
{
    protected:
        InequalityType type;              /** Inequality type.                   */
        int D;                            /** Number of variables.               */ 
        int N;                            /** Number of constraints.             */
        Matrix<T, Dynamic, Dynamic> A;    /** Matrix of constraint coefficients. */ 
        Matrix<T, Dynamic, 1>       b;    /** Matrix of constraint values.       */ 
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
            Matrix<T, Dynamic, Dynamic> A(N, D);
            Matrix<T, Dynamic, 1> b(N);
            
            std::string line;
            std::ifstream infile(filename); 
            if (infile.is_open())
            {
                while (std::getline(infile, line))
                {
                    // Accumulate the entries in each line ...
                    std::stringstream ss(line);
                    std::string token;
                    std::vector<T> row;
                    N++;
                    while (std::getline(ss, token, ' '))
                    {
                        // Parse each line as a double, then convert to the
                        // scalar type of choice
                        row.push_back(static_cast<T>(std::stod(token)));
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
            this->updateNearestL2();
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
            this->A = Matrix<T, Dynamic, Dynamic>::Zero(0, 0);
            this->b = Matrix<T, Dynamic, 1>::Zero(0);
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
        LinearConstraints(const InequalityType type, const int D, const T lower,
                          const T upper)
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
                Matrix<T, Dynamic, 1> v = Matrix<T, Dynamic, 1>::Zero(this->D);
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
            this->updateNearestL2();
        } 

        /**
         * Constructor with matrix and vector specifying the constraints.
         *
         * @param type Inequality type. 
         * @param A    Left-hand matrix in the constraints.
         * @param b    Right-hand vector in the constraints.  
         */
        LinearConstraints(const InequalityType type, const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                          const Ref<const Matrix<T, Dynamic, 1> >& b)
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
            this->updateNearestL2();
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
            this->updateNearestL2(); 
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
            this->type = type;
            this->updateNearestL2(); 
        }

        /**
         * Update `this->A` and `this->b` to the given matrix and vector. 
         *
         * @param A Left-hand matrix in the constraints. 
         * @param b Right-hand matrix in the constraints. 
         */
        void setAb(const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                   const Ref<const Matrix<T, Dynamic, 1> >& b)
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
            this->updateNearestL2(); 
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
        Matrix<T, Dynamic, Dynamic> getA()
        {
            return this->A;
        }

        /**
         * Return `this->b`.
         */
        Matrix<T, Dynamic, 1> getb()
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
        bool query(const Ref<const Matrix<T, Dynamic, 1> >& x)
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
        Matrix<bool, Dynamic, 1> active(const Ref<const Matrix<T, Dynamic, 1> >& x)
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
         * Return the approximate nearest point to the given query vector,
         * with respect to L2 (Euclidean) distance, that satisfies the
         * constraints.
         *
         * Note that, while the returned vector has scalar type `T`, the 
         * nearest-point quadratic program is solved with double-precision
         * arithmetic, meaning that the returned vector may be slightly
         * inaccurate.
         *
         * The input vector may have scalar type other than `T` or double,
         * but its coordinates are converted to doubles for the quadratic
         * program.
         * 
         * @param x Query vector. 
         * @returns Vector approximately nearest to query that satisfies the
         *          constraints.  
         */
        template <typename U>
        Matrix<T, Dynamic, 1> approxNearestL2(const Ref<const Matrix<U, Dynamic, 1> >& x)
        {
            // First check that x itself satisfies the constraints
            Matrix<T, Dynamic, 1> x_ = x.template cast<T>(); 
            if (this->query(x_)) return x_;

            // Otherwise, solve the quadratic program for the nearest point to x
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
            Matrix<T, Dynamic, 1> y = Matrix<T, Dynamic, 1>::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }
            return y;
        }

        /**
         * Return the solution to the given linear program, with the feasible 
         * region given by the stored constraints (`this->A * x <=/==/>= this->b`).
         *
         * The linear program seeks to *minimize* the given objective function.
         *
         * Note that, while the returned vector has scalar type `T`, the 
         * linear program is solved with double-precision arithmetic, meaning
         * that the returned vector may be slightly inaccurate. 
         *
         * @param obj Vector of length `this->D` encoding the objective function.
         * @param c0  Constant term of the objective function.
         * @returns   Vector solution to the given linear program.  
         */
        Matrix<T, Dynamic, 1> solveLinearProgram(const Ref<const Matrix<T, Dynamic, 1> >& obj,
                                                 const T c0)
        {
            // Instantiate the linear program ... 
            Program* program = new Program(CGAL::SMALLER, false, 0.0, false, 0.0);
            for (unsigned i = 0; i < this->N; ++i)
            {
                for (unsigned j = 0; j < this->D; ++j)
                    program->set_a(j, i, static_cast<double>(this->A(i, j)));
                program->set_b(i, static_cast<double>(this->b(i)));
                program->set_r(i, cgalInequalityType(this->type)); 
            }
            for (unsigned i = 0; i < this->D; ++i)
                program->set_c(i, static_cast<double>(obj(i))); 
            program->set_c0(static_cast<double>(c0));

            // ... and (try to) solve it ... 
            Solution solution = CGAL::solve_quadratic_program(*program, ET());
            if (solution.is_infeasible())
                throw std::runtime_error("Quadratic program is infeasible");
            else if (solution.is_unbounded())
                throw std::runtime_error("Quadratic program is unbounded");
            else if (!solution.is_optimal())
                throw std::runtime_error("Failed to compute optimal solution");

            // ... and return the solution vector
            Matrix<T, Dynamic, 1> y = Matrix<T, Dynamic, 1>::Zero(this->D);
            unsigned i = 0;
            for (auto it = solution.variable_values_begin(); it != solution.variable_values_end(); ++it)
            {
                y(i) = static_cast<T>(CGAL::to_double(*it));
                i++;
            }
            delete program; 
            return y;
        }

        /**
         * Determine whether the `k`-th stored constraint is redundant.
         *
         * This is determined here by setting up the linear program excluding
         * the `k`-th constraint with the objective function set to the 
         * (right-hand-side of the) constraint. If the constraint is redundant,
         * then the optimal objective value should satisfy the constraint.
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
            bool is_redundant = false; 
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
            this->updateNearestL2(); 
        }
};

}   // namespace Polytopes

#endif
