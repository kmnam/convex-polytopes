#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "linearConstraints.hpp"

/**
 * A basic implementation of Avis & Fukuda's algorithm for vertex enumeration
 * in convex polytopes, from:
 *
 * Avis & Fukuda, A pivoting algorithm for convex hulls and vertex enumeration
 * of arrangements and polyhedra, Discrete Comput Geom, 8: 295-313 (1992).  
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/20/2022
 */
using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpq_rational; 
using boost::multiprecision::mpfr_float_backend;

namespace Polytopes {

class PolyhedralDictionarySystem : public LinearConstraints
{
    private:
        // Number of rows in the core linear system; set to this->N + 1
        int nrows;

        // Number of columns in the core linear system; set to this->N + this->D + 2
        int ncols;

        // Boolean vector indicating the current basis, which should always 
        // have size = this->N; initialized to 0, ..., this->N - 1
        Matrix<bool, Dynamic, 1> basis;

        // Core linear system whose dictionaries are to be enumerated; has 
        // size (this->N + 1, this->N + this->D + 2)  
        MatrixXd core_A;

        // Dictionary coefficient matrix corresponding to the current basis;
        // has size (this-> 
        MatrixXd dict_coefs;

        /**
         * Private method to be called by all constructors, defining the 
         * core linear system, basis, cobasis, and dictionary coefficient 
         * matrix from the given set of linear constraints.  
         */
        void update()
        {
            this->nrows = this->N + 1;  
            this->ncols = this->N + this->D + 2; 

            // Construct the dictionary matrix ...
            this->core_A = MatrixXd::Zero(this->nrows, this->ncols);
            auto basis_seq = Eigen::seqN(0, this->N); 
            auto cobasis_seq = Eigen::seqN(this->N, this->D);
            auto last_N_rows_seq = Eigen::seqN(1, this->N); 
            this->core_A(0, basis_seq) = VectorXd::Ones(this->D); 
            this->core_A(0, this->ncols - 2) = 1; 
            this->core_A(last_N_rows_seq, basis_seq) = MatrixXd::Identity(this->N, this->N); 
            this->core_A(last_N_rows_seq, cobasis_seq) = this->A;       // TODO Negative?  
            this->core_A(last_N_rows_seq, this->ncols - 1) = -this->b;  // TODO Negative?

            // ... and initialize the basis
            this->basis = Matrix<bool, Dynamic, 1>::Zero(this->N + this->D);  
            for (int i = 0; i < this->N; ++i)
                this->basis(i) = true;

            // Compute the dictionary coefficient matrix
            MatrixXd block_basis_inv = MatrixXd::Zero(this->nrows, this->nrows); 
            MatrixXd block_cobasis = MatrixXd::Zero(this->nrows, this->D + 1);
            block_basis_inv.block(1, 1, this->N, this->N) = MatrixXd::Identity(this->N, this->N);
            block_basis_inv.block(this->N, 1, 1, this->N) = -VectorXd::Ones(this->N); 
            block_basis_inv(this->N, 0) = 1;  
            block_cobasis(Eigen::all, Eigen::seqN(0, this->D)) = this->core_A(Eigen::all, cobasis_seq);
            block_cobasis.col(this->D) = this->core_A.col(this->N + this->D + 1);
            this->dict_coefs = -block_basis_inv * block_cobasis;  
        }

        /**
         * Perform the given pivot between the i-th variable (in the basis) 
         * and the j-th variable (in the cobasis) of the current dictionary
         * *without checking whether i and j are valid indices or whether the
         * pivot is primal and/or dual feasible*.
         *
         * @param i Index of basis variable. 
         * @param j Index of cobasis variable. 
         */
        void __pivot(const int i, const int j)
        {
            // Get all indices in the basis and cobasis 
            std::vector<int> basis_indices_except_ij, cobasis_indices_except_ij; 
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if (this->basis(k) && k != i)
                    basis_indices_except_ij.push_back(k); 
                else if (!this->basis(k) && k != j)
                    cobasis_indices_except_ij.push_back(k); 
            } 
            
            // Perform the pivot
            this->basis(i) = false; 
            this->basis(j) = true;
            VectorXd dict_row_i = this->dict_coefs.row(i); 
            VectorXd dict_col_j = this->dict_coefs.col(j);
            double dict_ij = dict_row_i(j); 
            this->dict_coefs(j, i) = 1 / dict_ij; 
            this->dict_coefs.col(i) = dict_col_j / dict_ij; 
            this->dict_coefs.row(j) = -dict_row_i / dict_ij;
            for (int k : basis_indices_except_ij)
            {
                for (int m : cobasis_indices_except_ij) 
                {
                    this->dict_coefs(k, m) -= (dict_col_j(k) * dict_row_i(m) / dict_ij);
                }
            }
        }

        /**
         * Determine if the given pivot is a reverse Bland pivot, by:
         * 
         * 1) performing the given pivot,
         * 2) determining the Bland pivot of the new dictionary,
         * 3) reversing the given pivot to recover the old dictionary, and 
         * 4) determining if this Bland pivot matches the reverse of the 
         *    given pivot.  
         *
         * @param s Index of basis variable of the given pivot. 
         * @param r Index of cobasis variable of the given pivot.
         * @returns True if the pivot is a reverse Bland pivot, false otherwise.
         * @throws std::runtime_error If the current dictionary is primal infeasible
         *                            (rethrown exception thrown in `findBland()`).
         */
        bool isReverseBlandPivot(const int s, const int r)
        {
            // Perform the given pivot
            this->__pivot(s, r);

            // Determine the Bland pivot of the obtained dictionary 
            int new_r, new_s; 
            try
            {
                std::tie(new_r, new_s) = this->findBland(); 
            }
            catch (const std::runtime_error& e) 
            {
                throw; 
            }

            // Reverse the given pivot to recover the old dictionary
            this->__pivot(r, s);

            // Determine if this Bland pivot matches the reverse of the 
            // given pivot 
            return (new_r == r && new_s == s); 
        }

        /**
         * Determine if the given pivot is a reverse criss-cross pivot, by:
         * 
         * 1) performing the given pivot,
         * 2) determining the criss-cross pivot of the new dictionary,
         * 3) reversing the given pivot to recover the old dictionary, and 
         * 4) determining if this criss-cross pivot matches the reverse of the 
         *    given pivot.  
         *
         * @param s Index of basis variable of the given pivot. 
         * @param r Index of cobasis variable of the given pivot.
         * @returns True if the pivot is a reverse criss-cross pivot, false
         *          otherwise.
         */
        bool isReverseCrissCrossPivot(const int s, const int r)
        {
            // Perform the given pivot
            this->__pivot(s, r);

            // Determine the criss-cross pivot of the obtained dictionary 
            int new_r, new_s; 
            std::tie(new_r, new_s) = this->findBland(); 

            // Reverse the given pivot to recover the old dictionary
            this->__pivot(r, s);

            // Determine if this criss-cross pivot matches the reverse of the 
            // given pivot 
            return (new_r == r && new_s == s); 
        }

    public:
        /**
         * Trivial constructor that specifies only the inequality type.
         *
         * @param type Inequality type. 
         */
        PolyhedralDictionarySystem(const InequalityType type) : LinearConstraints(type)
        {
            this->update();
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
        PolyhedralDictionarySystem(const InequalityType type, const int D,
                                   const double lower, const double upper)
            : LinearConstraints(type, D, lower, upper)
        {
            this->update();
        }

        /**
         * Constructor with matrix and vector specifying the constraints. 
         *
         * @param type Inequality type.
         * @param A    Left-hand matrix in the constraints.
         * @param b    Right-hand vector in the constraints. 
         */
        PolyhedralDictionarySystem(const InequalityType type,
                                   const Ref<const MatrixXd>& A,
                                   const Ref<const VectorXd>& b)
            : LinearConstraints(type, A, b) 
        {
            this->update(); 
        } 

        /**
         * Empty destructor. 
         */
        ~PolyhedralDictionarySystem()
        {
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces 
         * (inequalities), read in the constraint matrix and vector, remove 
         * redundant constraints, and overwrite the dictionary system. 
         *
         * @param filename Path to file containing the polytope constraints.
         * @param type     Inequality type (not denoted in the file).  
         */
        void parse(const std::string filename, const InequalityType type)
        {
            LinearConstraints::parse(filename, type); 
            this->removeRedundantConstraints(); 
            this->update(); 
        }

        /**
         * Return the left-hand matrix of the core linear system. 
         *
         * @returns Left-hand matrix of the core linear system. 
         */
        MatrixXd getCoreSystem()
        {
            return this->core_A; 
        }

        /**
         * Return the boolean vector indicating the current choice of basis
         * (and cobasis) indices. 
         *
         * @returns Boolean vector indicating current basis and cobasis indices.
         */
        Matrix<bool, Dynamic, 1> getBasis()
        {
            return this->basis; 
        }

        /**
         * Return the current dictionary coefficient matrix. 
         *
         * @return Current dictionary coefficient matrix.
         */
        MatrixXd getDictCoefs()
        {
            return this->dict_coefs; 
        }

        /**
         * Return true if the i-th variable is primal feasible in the current
         * dictionary.
         *
         * @param i Index of variable of interest. 
         * @returns True if the variable is primal feasible, false otherwise. 
         */ 
        bool isPrimalFeasible(const int i)
        {
            return (
                i >= 0 && i < this->N + this->D && this->basis(i) &&
                this->dict_coefs(i, this->N + this->D + 1) >= 0
            ); 
        }

        /**
         * Return true if the i-th variable is dual feasible in the current
         * dictionary.
         *
         * @param i Index of variable of interest. 
         * @returns True if the variable is dual feasible, false otherwise. 
         */ 
        bool isDualFeasible(const int i)
        {
            return (
                i >= 0 && i < this->N + this->D && !this->basis(i) &&
                this->dict_coefs(this->N + this->D, i) <= 0
            ); 
        }

        /**
         * Perform the given pivot between the i-th variable (in the basis) 
         * and the j-th variable (in the cobasis) of the current dictionary.
         *
         * @param i Index of basis variable. 
         * @param j Index of cobasis variable. 
         * @returns 1 if the pivot is primal feasible, 2 if the pivot is dual 
         *          feasible, 3 if the pivot is both, 0 otherwise.
         * @throws std::invalid_argument If `i` is not in the current basis or 
         *                               `j` is not in the current cobasis. 
         */
        int pivot(const int i, const int j)
        {
            // Check whether the i-th variable is indeed in the basis 
            if (i < 0 || i >= this->N + this->D || !this->basis(i))
                throw std::invalid_argument("Undefined pivot: variable i is not in basis");

            // Check whether the j-th variable is indeed in the cobasis 
            if (j < 0 || j >= this->N + this->D || this->basis(j))
                throw std::invalid_argument("Undefined pivot: variable j is not in cobasis");

            // Check whether the dictionary is primal and/or dual feasible
            bool all_primal_curr = this->isPrimalFeasible(i);
            bool all_dual_curr = this->isDualFeasible(j);
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if (all_primal_curr && this->basis(k) && !this->isPrimalFeasible(k))
                    all_primal_curr = false; 
                else if (all_dual_curr && !this->basis(k) && !this->isDualFeasible(k))
                    all_dual_curr = false; 
            }

            // Perform the given pivot 
            this->__pivot(i, j);  

            // Now check whether the new dictionary is primal feasible and/or 
            // dual feasible
            bool all_primal_next = this->isPrimalFeasible(j);
            bool all_dual_next = this->isDualFeasible(i); 
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if (all_primal_next && this->basis(k) && !this->isPrimalFeasible(k))
                    all_primal_next = false; 
                else if (all_dual_next && !this->basis(k) && !this->isDualFeasible(k))
                    all_dual_next = false; 
            }
            
            // Return an indicator for whether the previous and/or new dictionaries
            // are primal and/or dual feasible 
            if (all_primal_curr && all_primal_next && all_dual_curr && all_dual_next)
                return 3; 
            else if (all_dual_curr && all_dual_next) 
                return 2; 
            else if (all_primal_curr && all_primal_next) 
                return 1; 
            else 
                return 0;  
        }

        /**
         * Find the pivot determined by Bland's rule on the current dictionary
         * and return the indices being switched.
         * 
         * @returns The basis index and cobasis index being switched by the 
         *          Bland pivot, *or* the pair `[-1, -1]` if the dictionary is 
         *          dual feasible. 
         * @throws std::runtime_error If the current dictionary is primal infeasible.
         */
        std::pair<int, int> findBland()
        {
            // Get all indices in the basis and cobasis 
            std::vector<int> basis_indices, cobasis_indices; 
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if (this->basis(k))
                    basis_indices.push_back(k); 
                else if (!this->basis(k))
                    cobasis_indices.push_back(k); 
            } 

            // Check whether the dictionary is primal feasible 
            for (const int k : basis_indices) 
            {
                if (!this->isPrimalFeasible(k))
                {
                    throw std::runtime_error(
                        "Bland's rule cannot be performed on primal infeasible dictionary"
                    ); 
                }
            } 

            // Find the least index in the cobasis such that the corresponding
            // variable is dual infeasible 
            int s = -1; 
            for (const int k : cobasis_indices) 
            {
                if (!this->isDualFeasible(k))
                {
                    s = k;
                    break;
                } 
            }

            // If no such index exists, then return [-1, -1]
            if (s == -1) 
                return std::make_pair(-1, -1); 

            // Find the least index in the basis obtaining the minimum value of
            // -(dict_coefs(i, N+D+1) / dict_coefs(i, s))
            int r = -1;
            double lambda = std::numeric_limits<double>::infinity(); 
            for (const int k : basis_indices) 
            {
                double value = -(this->dict_coefs(k, this->N + this->D + 1) / this->dict_coefs(k, s));
                if (lambda < value) 
                {
                    lambda = value;
                    r = k;  
                } 
            }

            // Return the chosen indices 
            return std::make_pair(r, s); 
        }

        /**
         * Perform the pivot determined by Bland's rule on the current dictionary.
         *
         * @throws std::runtime_error If the current dictionary is primal infeasible
         *                            (rethrown exception thrown in `findBland()`).
         */
        void pivotBland()
        {
            int r, s;
            try
            { 
                std::tie(r, s) = this->findBland();
            }
            catch (const std::runtime_error& e) 
            {
                throw; 
            }

            // Perform the pivot only if it is actually allowed 
            if (r != -1) 
                this->__pivot(r, s);
        }

        /**
         * Find the pivot determined by the criss-cross rule on the current
         * dictionary and return the indices being switched.  
         * 
         * @returns The basis index and cobasis index being switched by the 
         *          criss-cross pivot, *or* the pair `[-1, -1]` if the
         *          dictionary is primal or dual feasible. 
         */
        std::pair<int, int> findCrissCross()
        {
            // Get all indices in the basis and cobasis 
            std::vector<int> basis_indices, cobasis_indices; 
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if (this->basis(k))
                    basis_indices.push_back(k); 
                else if (!this->basis(k))
                    cobasis_indices.push_back(k); 
            } 

            // Find the least index such that the corresponding variable is
            // primal or dual infeasible 
            int i = -1; 
            for (int k = 0; k < this->N + this->D; ++k)
            {
                if ((this->basis(k) && !this->isPrimalFeasible(k)) ||
                    (!this->basis(k) && !this->isDualFeasible(k)))
                {
                    i = k;
                    break;
                } 
            }

            // If no such index exists, then return [-1, -1]
            if (i == -1) 
                return std::make_pair(-1, -1); 

            // If i is in the basis ...
            int r, s; 
            if (this->basis(i))
            {
                // Set r = i and set s to the least (cobasis) index such that
                // dict_coefs(r, s) > 0
                r = i;
                for (const int k : cobasis_indices) 
                {
                    if (this->dict_coefs(r, k) > 0)
                    {
                        s = k; 
                        break; 
                    } 
                }
            }
            // Otherwise ... 
            {
                // Set s = i and set r to the least (basis) index such that
                // dict_coefs(r, s) < 0
                s = i; 
                for (const int k : basis_indices)
                {
                    if (this->dict_coefs(k, s) < 0)
                    {
                        r = k; 
                        break; 
                    }
                }
            }

            // Return the chosen indices 
            return std::make_pair(r, s); 
        }

        /**
         * Perform the pivot determined by the criss-cross rule on the current
         * dictionary. 
         */
        void pivotCrissCross()
        {
            int r, s;
            std::tie(r, s) = this->findCrissCross();

            // Perform the pivot only if it is actually allowed 
            if (r != -1) 
                this->__pivot(r, s);
        }

        /**
         * Return the basic solution to the current dictionary, along with an 
         * indicator for whether it is degenerate. 
         *
         * @returns The basic solution to the current dictionary, along with 
         *          an indicator for whether the solution is degenerate.  
         */ 
        std::pair<VectorXd, bool> solve()
        {
            VectorXd cobasic = VectorXd::Zero(this->D + 1); 
            cobasic(this->D) = 1; 
            VectorXd basic = this->dict_coefs * cobasic;
            return std::make_pair(basic, (basic.array() == 0).any());  
        }
}; 

}   // namespace Polytopes 
