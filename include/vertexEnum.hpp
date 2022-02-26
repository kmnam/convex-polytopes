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
 * - Avis & Fukuda, A pivoting algorithm for convex hulls and vertex enumeration
 *   of arrangements and polyhedra, Discrete Comput Geom, 8: 295-313 (1992).
 * - Avis, A C implementation of the reverse search vertex enumeration algorithm,
 *   RIMS Kokyuroku 872 (H Imai, ed.), Kyoto University (1994).
 * - Avis, A revised implementation of the reverse search vertex enumeration 
 *   algorithm (2000).
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/26/2022
 */
using namespace Eigen;
using boost::multiprecision::number; 
using boost::multiprecision::mpq_rational; 
using boost::multiprecision::mpfr_float_backend;
typedef Matrix<bool, Dynamic, 1>               VectorXb; 
typedef Matrix<mpq_rational, Dynamic, Dynamic> MatrixXr; 
typedef Matrix<mpq_rational, Dynamic, 1>       VectorXr; 

namespace Polytopes {

/** Exception to be thrown when inversion of a singular matrix is attempted. */
class SingularMatrixException    : public std::runtime_error
{
    public:
        SingularMatrixException(const std::string& what = "")    : std::runtime_error(what) {}
}; 
/** Exception to be thrown when a pivot could not be performed.              */
class InvalidPivotException      : public std::runtime_error
{
    public:
        InvalidPivotException(const std::string& what = "")      : std::runtime_error(what) {}
};
/**
 * Exception to be thrown when a dictionary is primal infeasible when it should
 * not be (e.g., when applying Bland's rule).
 */ 
class PrimalInfeasibleException  : public std::runtime_error
{
    public:
        PrimalInfeasibleException(const std::string& what = "")  : std::runtime_error(what) {}
};
/**
 * Exception to be thrown when a dictionary is dual feasible when it should 
 * not be (e.g., when applying Bland's rule). 
 */ 
class DualFeasibleException      : public std::runtime_error
{
    public:
        DualFeasibleException(const std::string& what = "")      : std::runtime_error(what) {}
};
/**
 * Exception to be thrown when a dictionary is optimal when it should not be
 * (e.g., when applying the criss-cross pivot rule). 
 */
class OptimalDictionaryException : public std::runtime_error
{
    public:
        OptimalDictionaryException(const std::string& what = "") : std::runtime_error(what) {}
}; 

/**
 * Apply one iteration of the Faddeev-LeVerrier algorithm on the given N-by-N
 * matrix `A`, to obtain the i-th auxiliary matrix and (N-i)-th scalar 
 * from the (i-1)-th auxiliary matrix `M` and (N-i+1)-th scalar `c`.
 *
 * @param A Matrix whose inverse is ultimately desired. 
 * @param M Auxiliary matrix obtained from previous Faddeev-LeVerrier iteration(s).
 * @param c Constant obtained from previous Faddevv-LeVerrier iteration(s). 
 * @param i Iteration number.
 * @returns Next matrix and constant from the Faddeev-LeVerrier algorithm.
 * @throws  boost::wrapexcept<std::overflow_error> If the Faddeev-LeVerrier
 *                                                 algorithm fails due to
 *                                                 division-by-zero.  
 */
std::pair<MatrixXr, mpq_rational> faddeevLeVerrier(const Ref<const MatrixXr>& A,
                                                   const Ref<const MatrixXr>& M,
                                                   const mpq_rational c,
                                                   const int i)
{
    MatrixXr M_next = A * M + c * MatrixXr::Identity(A.rows(), A.rows());
    mpq_rational c_next; 
    try
    {
        c_next = -(A * M_next).trace() / i;
    }
    catch (const boost::wrapexcept<std::overflow_error>& e)
    {
        throw; 
    } 
    return std::make_pair(M_next, c_next); 
}

/**
 * Invert the given square rational matrix using the Faddeev-LeVerrier algorithm.
 *
 * @param A Input rational matrix. 
 * @returns Inverse of `A`, with rational scalars.
 * @throws  SingularMatrixException If the Faddeev-LeVerrier algorithm fails.  
 */
MatrixXr invertRational(const Ref<const MatrixXr>& A)
{
    int N = A.rows(); 
    MatrixXr M = MatrixXr::Zero(N, N); 
    mpq_rational c = 1; 
    for (int i = 1; i <= N; ++i)
    {
        std::pair<MatrixXr, mpq_rational> next;
        try
        { 
            next = faddeevLeVerrier(A, M, c, i);
        }
        catch (const boost::wrapexcept<std::overflow_error>& e) 
        {
            throw SingularMatrixException(
                "Matrix inversion failed due to division-by-zero encountered "
                "during Faddeev-LeVerrier algorithm"
            ); 
        } 
        M = next.first; 
        c = next.second;
    }

    MatrixXr inv;
    try
    {
        inv = -M / c;
    }
    catch (boost::wrapexcept<std::overflow_error>& e)
    {
        throw SingularMatrixException(
            "Matrix inversion failed due to division-by-zero encountered "
            "during Faddeev-LeVerrier algorithm"
        ); 
    }
    return inv; 
}

class PolyhedralDictionarySystem : public LinearConstraints<mpq_rational>
{
    protected:
        /**
         * Boolean vector indicating the current basis indices (and cobasis
         * indices), initialized to `this->N + 1` ones followed by `this->D + 1`
         * zeros. 
         */
        VectorXb in_basis; 

        /**
         * Integer vector containing the current basis indices, initialized to
         * `0`, ..., `this->N`.
         */ 
        VectorXi basis;
        
        /**
         * Integer vector containing the current cobasis indices, initialized to
         * `this->N + 1`, ..., `this->N + this->D + 1`.
         */
        VectorXi cobasis;  

        /**
         * Core linear system whose dictionaries are to be enumerated; has
         * size `(this->N + 1, this->N + this->D + 2)`.
         */
        MatrixXr core_A; 

        /**
         * Dictionary coefficient matrix corresponding to the current basis.
         */
        MatrixXr dict_coefs; 
        MatrixXr basis_inv_coefs; 

        /**
         * Basic solution of the current dictionary; has size `this->N + this->D + 2`. 
         */
        VectorXr basic_solution;

        /**
         * Indicator for whether the current basic solution is degenerate. 
         */
        bool is_degenerate;  

        /**
         * Update the core linear system from the current set of linear constraints. 
         */
        void updateCore()
        {
            // Initialize the basis and cobasis, as: 
            //
            // basis   = {0, 1, ..., this->N}
            // cobasis = {this->N + 1, ..., this->N + this->D, this->N + this->D + 1}
            //
            // 0 always remains in the basis and this->N + this->D + 1 always 
            // remains in the cobasis
            this->in_basis = VectorXb::Zero(this->N + this->D + 2);
            this->in_basis.head(this->N + 1) = VectorXb::Ones(this->N + 1); 
            this->basis = VectorXi::Zero(this->N + 1); 
            for (int i = 0; i <= this->N; ++i)
                this->basis(i) = i;
            this->cobasis = VectorXi::Zero(this->D + 1); 
            for (int i = 0; i <= this->D; ++i)
                this->cobasis(i) = this->N + 1 + i;

            // Construct the core linear system
            this->core_A = MatrixXr::Zero(this->N + 1, this->N + this->D + 2); 
            this->core_A(0, this->cobasis.head(this->D)) = VectorXr::Ones(this->D); 
            this->core_A(0, 0) = 1;
            this->core_A(Eigen::seqN(1, this->N), this->basis.tail(this->N)) = MatrixXr::Identity(this->N, this->N);
            this->core_A(Eigen::seqN(1, this->N), this->cobasis.head(this->D)) = this->A; 
            this->core_A(Eigen::seqN(1, this->N), this->N + this->D + 1) = -this->b;
            // TODO Introduce options for different inequality types
        }

        /**
         * Update the dictionary from the current core linear system and choice
         * of basis and cobasis.
         *
         * @throws SingularMatrixException If the submatrix of the core matrix
         *                                 corresponding to the basis columns  
         *                                 is not invertible (rethrown exception
         *                                 from `invertRational()`).  
         */
        void updateDictCoefs()
        {
            try
            {
                this->basis_inv_coefs = invertRational(this->core_A(Eigen::all, this->basis));
            }
            catch (const SingularMatrixException& e) 
            {
                throw;
            } 
            this->dict_coefs = -this->basis_inv_coefs * this->core_A(Eigen::all, this->cobasis); 
        }

        /**
         * Update the basic solution corresponding to the current dictionary.
         */
        void updateBasicSolution()
        {
            VectorXr v = VectorXr::Zero(this->D + 1); 
            v(this->D) = 1; 
            this->basic_solution = this->dict_coefs * v;
            this->is_degenerate = (this->basic_solution.array() == 0).any();  
        }

        /**
         * Perform the given pivot between `this->basis(i)` and `this->cobasis(j)`
         * *without checking whether `this->basis(i)` and `this->cobasis(j)
         * are valid indices or whether the pivot is primal and/or dual
         * feasible*.
         *
         * @param i Position of basis variable in the current basis. 
         * @param j Position of cobasis variable in the current cobasis.
         * @returns The basis position (`i` if basis index is `this->basis(i)`)
         *          and cobasis position (`j` if cobasis index is `this->cobasis(j)`)
         *          in the *new* dictionary obtained after the pivot.
         * @throws InvalidPivotException If the pivot cannot be performed.  
         */
        std::pair<int, int> __pivot(const int i, const int j)
        {
            const int r = this->basis(i); 
            const int s = this->cobasis(j);

            // Switch r and s between the basis and cobasis 
            this->in_basis(r) = false; 
            this->in_basis(s) = true;
            int p = 0; 
            int q = 0; 
            for (int k = 0; k < this->N + this->D + 2; ++k)
            {
                if (this->in_basis(k))
                {
                    this->basis(p) = k; 
                    p++; 
                }
                else 
                {
                    this->cobasis(q) = k; 
                    q++; 
                }
            }

            // Find the indices of s in the new basis and r in the new cobasis 
            int new_i, new_j; 
            for (int k = 1; k <= this->N; ++k)
            {
                if (this->basis(k) == s)
                {
                    new_i = k; 
                    break; 
                }
            }
            for (int k = 0; k < this->D; ++k)
            {
                if (this->cobasis(k) == r)
                {
                    new_j = k;
                    break; 
                }
            }

            // Update the dictionary coefficient matrix and basic solution
            try
            {
                this->updateDictCoefs();
            }
            // If the submatrix of the core matrix corresponding to the basis
            // variables is not invertible, then the pivot cannot be performed
            catch (const SingularMatrixException& e)
            {
                // Switch r and s *back* between the basis and cobasis 
                this->in_basis(r) = true; 
                this->in_basis(s) = false; 
                int p = 0; 
                int q = 0; 
                for (int k = 0; k < this->N + this->D + 2; ++k)
                {
                    if (this->in_basis(k))
                    {
                        this->basis(p) = k; 
                        p++; 
                    }
                    else 
                    {
                        this->cobasis(q) = k; 
                        q++; 
                    }
                }

                // Throw exception to signal that the pivot failed
                throw InvalidPivotException("Specified pivot is invalid"); 
            } 
            this->updateBasicSolution();

            // Return the indices of s in the new basis and r in the new cobasis
            return std::make_pair(new_i, new_j);  
        }

        /**
         * Determine if the given pivot is a reverse Bland pivot.
         *
         * @param j Position of basis variable in the current basis. 
         * @param i Position of cobasis variable in the current cobasis. 
         * @returns True if the pivot is a reverse Bland pivot, false otherwise.
         * @throws InvalidPivotException If the reverse pivot cannot be performed
         *                               or reversed.
         * @throws PrimalInfeasibleException If the Bland pivot of the newly
         *                                   obtained dictionary is not defined
         *                                   due to primal infeasibility.
         * @throws DualFeasibleException If the Bland pivot of the newly obtained
         *                               dictionary is not defined due to dual
         *                               feasibility. 
         */
        bool isReverseBlandPivot(const int j, const int i) 
        {
            // Perform the reverse pivot
            //
            // this->basis(j) is sent to the cobasis and becomes this->cobasis(new_i)
            // this->cobasis(i) is sent to the basis and becomes this->basis(new_j)
            //
            // If the reverse pivot could not be performed, then simply return false 
            int new_j, new_i;
            try
            { 
                std::tie(new_j, new_i) = this->__pivot(j, i);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }

            // Now find the pivot corresponding to Bland's rule for this 
            // new dictionary 
            int bland_i, bland_j;
            try
            { 
                std::tie(bland_i, bland_j) = this->findBland();
            }
            catch (const PrimalInfeasibleException& e) 
            {
                throw; 
            }
            catch (const DualFeasibleException& e)
            {
                throw; 
            }

            // Reverse the reverse pivot to obtain the original dictionary 
            try
            { 
                this->__pivot(new_j, new_i);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }

            // Is this reverse of the reverse pivot the same as the Bland pivot? 
            return (new_j == bland_i && new_i == bland_j);  
        }

        /**
         * Determine if the given pivot is a reverse criss-cross pivot.
         *
         * @param j Position of basis variable in the current basis. 
         * @param i Position of cobasis variable in the current cobasis. 
         * @returns True if the pivot is a reverse criss-cross pivot, false
         *          otherwise.
         * @throws InvalidPivotException If the reverse pivot cannot be performed
         *                               or reversed.
         * @throws OptimalDictionaryException If the criss-cross pivot of the newly
         *                                    obtained dictionary is not defined
         *                                    due to optimality. 
         */
        bool isReverseCrissCrossPivot(const int j, const int i)
        {
            // Perform the reverse pivot
            //
            // this->basis(j) is sent to the cobasis and becomes this->cobasis(new_i)
            // this->cobasis(i) is sent to the basis and becomes this->basis(new_j)
            //
            // If the reverse pivot could not be performed, then simply return false 
            int new_j, new_i;
            try
            { 
                std::tie(new_j, new_i) = this->__pivot(j, i);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }

            // Now find the criss-cross pivot for this new dictionary
            int cc_i, cc_j; 
            try
            { 
                std::tie(cc_i, cc_j) = this->findCrissCross(); 
            }
            catch (const OptimalDictionaryException& e) 
            {
                throw; 
            }

            // Reverse the reverse pivot to obtain the original dictionary 
            try
            { 
                this->__pivot(new_j, new_i);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }

            // Is this reverse of the reverse pivot the same as the criss-cross pivot? 
            return (new_j == cc_i && new_i == cc_j); 
        }

    public:
        /**
         * Trivial constructor that specifies only the inequality type.
         *
         * @param type Inequality type. 
         */
        PolyhedralDictionarySystem(const InequalityType type) : LinearConstraints(type)
        {
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
                                   const mpq_rational lower, const mpq_rational upper)
            : LinearConstraints(type, D, lower, upper)
        {
            this->updateCore();
            this->updateDictCoefs();
            this->updateBasicSolution();  
        }

        /**
         * Constructor with matrix and vector specifying the constraints. 
         *
         * @param type Inequality type.
         * @param A    Left-hand matrix in the constraints.
         * @param b    Right-hand vector in the constraints. 
         */
        PolyhedralDictionarySystem(const InequalityType type,
                                   const Ref<const MatrixXr>& A,
                                   const Ref<const VectorXr>& b)
            : LinearConstraints(type, A, b) 
        {
            this->updateCore();
            this->updateDictCoefs();
            this->updateBasicSolution(); 
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
         */
        void parse(const std::string filename)
        {
            this->__parse(filename, this->type);
            this->removeRedundantConstraints();
            this->updateCore();
            this->updateDictCoefs();
            this->updateBasicSolution(); 
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
            this->__parse(filename, type); 
            this->removeRedundantConstraints(); 
            this->updateCore(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }

        /**
         * Return the left-hand matrix of the core linear system. 
         *
         * @returns Left-hand matrix of the core linear system. 
         */
        MatrixXr getCoreMatrix()
        {
            return this->core_A; 
        }

        /**
         * Return the vector of indices in the current choice of basis. 
         *
         * @returns Vector of indices in the current basis. 
         */
        VectorXi getBasis()
        {
            return this->basis; 
        }

        /**
         * Return the vector of indices in the current choice of cobasis. 
         *
         * @returns Vector of indices in the current cobasis.
         */
        VectorXi getCobasis()
        {
            return this->cobasis; 
        }

        /**
         * Return the current dictionary coefficient matrix. 
         *
         * @returns Current dictionary coefficient matrix.
         */
        MatrixXr getDictCoefs()
        {
            return this->dict_coefs; 
        }

        /**
         * Return the inverse of the basis submatrix of the current dictionary
         * coefficient matrix. 
         *
         * @returns Inverse of basis submatrix of current dictionary coefficient
         *          matrix 
         */
        MatrixXr getBasisInvCoefs()
        {
            return this->basis_inv_coefs; 
        }

        /**
         * Return true if `this->basis(i)` is primal feasible in the current 
         * dictionary. 
         *
         * @param i Position of variable in the current basis. 
         * @returns True if the variable is primal feasible, false otherwise.
         * @throws  std::invalid_argument If `i` is not a valid basis index.  
         */ 
        bool isPrimalFeasible(const int i)
        {
            if (i == 0) 
                throw std::invalid_argument("0 is undefined regarding primal feasibility"); 
            else if (i < 1 || i > this->N)
                throw std::invalid_argument("Invalid basis index specified"); 

            return (this->dict_coefs(i, this->D) >= 0);
        }

        /**
         * Return true if `this->cobasis(i)` is dual feasible in the current 
         * dictionary. 
         *
         * @param i Position of variable in the current cobasis. 
         * @returns True if the variable is dual feasible, false otherwise.
         * @throws  std::invalid_argument If `i` is not a valid cobasis index. 
         */ 
        bool isDualFeasible(const int i)
        {
            if (i == this->D)
                throw std::invalid_argument("N + D + 1 is undefined regarding dual feasibility"); 
            else if (i < 0 || i > this->D - 1)
                throw std::invalid_argument("Invalid cobasis index specified"); 

            return (this->dict_coefs(0, i) <= 0);
        }

        /**
         * Perform the given pivot between `this->basis(i)` and `this->cobasis(j)`
         *
         * @param i Position of basis variable in the current basis. 
         * @param j Position of cobasis variable in the current cobasis. 
         * @returns A triple containing: 
         *          - The basis position (`k` if basis index is `this->basis(k)`)
         *            in the *new* dictionary obtained after the pivot
         *          - The cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *            in the *new* dictionary obtained after the pivot 
         *          - An indicator set to 1 if the pivot is primal feasible and 
         *            dual infeasible, 2 if the pivot is dual feasible and primal
         *            infeasible, 3 if the pivot is primal and dual feasible, and
         *            0 otherwise.
         * @throws std::invalid_argument If `i` is not a valid basis index or 
         *                               `j` is not a valid cobasis index.
         * @throws InvalidPivotException If the pivot could not be performed 
         *                               (rethrown exception from `__pivot()`). 
         */
        std::tuple<int, int, int> pivot(const int i, const int j)
        {
            // Check whether i is indeed a valid basis index 
            if (i == 0)
                throw std::invalid_argument("Undefined pivot: basis must always contain 0"); 
            else if (i < 0 || i > this->N)
                throw std::invalid_argument("Undefined pivot: invalid basis index specified");

            // Check whether j is indeed a valid cobasis index
            if (j == this->D)
                throw std::invalid_argument("Undefined pivot: cobasis must always contain N + D + 1"); 
            else if (j < 0 || j > this->D)
                throw std::invalid_argument("Undefined pivot: invalid cobasis index specified"); 

            // Check whether the dictionary is primal and/or dual feasible
            bool all_primal_curr = true; 
            bool all_dual_curr = true; 
            for (int k = 1; k <= this->N; ++k)
            {
                if (all_primal_curr && !this->isPrimalFeasible(k))
                {
                    all_primal_curr = false;
                    break; 
                } 
            }
            for (int k = 0; k < this->D; ++k)
            {
                if (all_dual_curr && !this->isDualFeasible(k))
                { 
                    all_dual_curr = false;
                    break;
                } 
            }

            // Perform the given pivot
            int new_i, new_j;
            try
            { 
                std::tie(new_i, new_j) = this->__pivot(i, j);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }

            // Now check whether the new dictionary is primal feasible and/or 
            // dual feasible
            bool all_primal_next = true; 
            bool all_dual_next = true; 
            for (int k = 1; k <= this->N; ++k)
            {
                if (all_primal_next && !this->isPrimalFeasible(k))
                {
                    all_primal_next = false;
                    break;
                }
            }
            for (int k = 0; k < this->D; ++k)
            {
                if (all_dual_next && !this->isDualFeasible(k))
                {
                    all_dual_next = false;
                    break;
                }
            }
            
            // Return an indicator for whether the previous and/or new dictionaries
            // are primal and/or dual feasible 
            if (all_primal_curr && all_primal_next && all_dual_curr && all_dual_next)
                return std::make_tuple(new_i, new_j, 3); 
            else if (all_dual_curr && all_dual_next) 
                return std::make_tuple(new_i, new_j, 2); 
            else if (all_primal_curr && all_primal_next) 
                return std::make_tuple(new_i, new_j, 1); 
            else 
                return std::make_tuple(new_i, new_j, 0);  
        }

        /**
         * Find the pivot determined by Bland's rule on the current dictionary
         * and return the indices being switched.
         * 
         * @returns The basis position (`k` if basis index is `this->basis(k)`)
         *          and cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *          being switched by the Bland pivot. 
         * @throws PrimalInfeasibleException If the current dictionary is primal
         *                                   infeasible.
         * @throws DualFeasibleException If the current dictionary is dual feasible.
         */
        std::pair<int, int> findBland()
        {
            // Check whether the dictionary is primal feasible 
            for (int k = 1; k <= this->N; ++k)
            {
                if (!this->isPrimalFeasible(k))
                {
                    throw PrimalInfeasibleException(
                        "Bland's rule cannot be performed on primal infeasible "
                        "dictionary"
                    ); 
                }
            } 

            // Find the least index in the cobasis such that the corresponding
            // variable is dual infeasible 
            int j = -1;
            for (int k = 0; k < this->D; ++k)
            {
                if (!this->isDualFeasible(k))
                {
                    j = k;
                    break; 
                } 
            }

            // If no such index exists, then return [-1, -1]
            if (j == -1)
                throw DualFeasibleException(
                    "Bland's rule cannot be performed on dual feasible dictionary"
                ); 

            // Find the least index in the basis (other than 0) obtaining the
            // minimum value of -(dict_coefs(i, N+D+1) / dict_coefs(i, s))
            int i = -1;
            mpq_rational lambda = std::numeric_limits<mpq_rational>::max(); 
            for (int k = 1; k <= this->N; ++k)
            {
                mpq_rational denom = this->dict_coefs(k, j); 
                if (denom < 0)
                {
                    mpq_rational value = -this->dict_coefs(k, this->D) / denom; 
                    if (lambda > value) 
                    {
                        lambda = value;
                        i = k;
                    }
                }
            }

            // Return the chosen indices 
            return std::make_pair(i, j); 
        }

        /**
         * Perform the pivot determined by Bland's rule on the current dictionary.
         *
         * @returns The basis position (`k` if basis index is `this->basis(k)`)
         *          and cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *          in the *new* dictionary obtained after the pivot. 
         * @throws PrimalInfeasibleException If the current dictionary is primal
         *                                   infeasible (rethrown exception from
         *                                   `findBland()`).
         * @throws DualFeasibleException If the current dictionary is dual feasible
         *                               (rethrown exception from `findBland()`).
         * @throws InvalidPivotException If the pivot could not be performed 
         *                               (rethrown exception from `__pivot()`). 
         */
        std::pair<int, int> pivotBland()
        {
            int i, j; 
            try
            { 
                std::tie(i, j) = this->findBland();
            }
            catch (const PrimalInfeasibleException& e)
            {
                throw; 
            }
            catch (const DualFeasibleException& e)
            {
                throw; 
            }

            // Perform the pivot only if it is actually allowed 
            try
            {
                return this->__pivot(i, j);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }
        }

        /**
         * Find the pivot determined by the criss-cross rule on the current
         * dictionary and return the indices being switched.  
         * 
         * @returns The basis index and cobasis index being switched by the 
         *          criss-cross pivot.
         * @throws OptimalDictionaryException If the dictionary is primal and 
         *                                    dual feasible. 
         */
        std::pair<int, int> findCrissCross()
        {
            // Find the least index such that the corresponding variable is
            // primal or dual infeasible 
            int i = -1;
            int b = 0; 
            int c = 0;
            bool ith_var_in_basis; 
            for (int k = 0; k < this->N + this->D + 2; ++k)
            {
                if (k != 0 && this->in_basis(k))    // Is a basis variable other than 0
                {
                    if (!this->isPrimalFeasible(b))
                    {
                        i = b;
                        ith_var_in_basis = true; 
                        break;
                    } 
                    b++; 
                }
                else if (k == 0)
                {
                    b++; 
                }                                   // Is a cobasis variable other than N+D+1
                else if (k != this->N + this->D + 1 && !this->in_basis(k))
                {
                    if (!this->isDualFeasible(c))
                    {
                        i = c;
                        ith_var_in_basis = false; 
                        break; 
                    }
                    c++;
                }
                else    // k == this->N + this->D + 1
                {
                    c++; 
                }
            }

            // If there is no such index, then the dictionary is optimal 
            if (i == -1)
                throw OptimalDictionaryException(
                    "Criss-cross pivot is undefined for optimal dictionary"
                ); 

            // Otherwise, if i is in the basis ...
            if (ith_var_in_basis)
            {
                // Find the least cobasis index, this->cobasis(j), such that
                // dict_coefs(i, j) > 0
                for (int j = 0; j < this->D; ++j)
                {
                    if (this->dict_coefs(i, j) > 0)
                        return std::make_pair(i, j); 
                }   // Note that there must exist at least one such index 
            }
            // Otherwise ... 
            else
            {
                // Find the least basis index, this->basis(j), such that 
                // dict_coefs(j, i) < 0
                for (int j = 1; j <= this->N; ++j)
                {
                    if (this->dict_coefs(j, i) < 0)
                        return std::make_pair(j, i); 
                }   // Note that there must exist at least one such index 
            }
        }

        /**
         * Perform the pivot determined by the criss-cross rule on the current
         * dictionary.
         *
         * @returns The basis position (`k` if basis index is `this->basis(k)`)
         *          and cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *          in the *new* dictionary obtained after the pivot.
         * @throws OptimalDictionaryException If the dictionary is primal and 
         *                                    dual feasible (rethrown exception
         *                                    from `findCrissCross()`).
         * @throws InvalidPivotException If the pivot could not be performed 
         *                               (rethrown exception from `__pivot()`). 
         */
        std::pair<int, int> pivotCrissCross()
        {
            int i, j;
            try
            {
                std::tie(i, j) = this->findCrissCross();
            }
            catch (const OptimalDictionaryException& e)
            {
                throw; 
            }

            // Perform the pivot only if it is actually allowed 
            try
            { 
                return this->__pivot(i, j);
            }
            catch (const InvalidPivotException& e)
            {
                throw; 
            }
        }

        /**
         * 
         */
        void search()
        {
            int i = 1;
            int j = 0;
            int n = 0;  
            do
            {
                // Keep incrementing (i, j) until we find a reverse Bland or 
                // criss-cross pivot
                bool is_reverse_bland = this->isReverseBlandPivot(i, j); 
                bool is_reverse_cc = this->isReverseCrissCrossPivot(i, j); 
                while (i <= this->N && !is_reverse_bland && !is_reverse_cc)
                {
                    j++;
                    if (j == this->D + 1)
                    {
                        j = 0; 
                        i++; 
                    }
                    is_reverse_bland = this->isReverseBlandPivot(i, j); 
                    is_reverse_cc = this->isReverseCrissCrossPivot(i, j); 
                }

                // If we have found a reverse Bland or criss-cross pivot ...
                if (i <= this->N)
                {
                    // ... then perform the pivot
                    std::tie(i, j) = this->__pivot(i, j);
                    if (i == -1)
                        throw std::runtime_error("What's happening here????"); 

                    // If the corresponding basic solution is degenerate, check
                    // whether it corresponds to a lex-min basis 
                    if (this->is_degenerate)
                    {
                        bool lexmin = true; 
                        for (int k = 1; k <= this->N; ++k)
                        {
                            for (int m = 0; m < this->D; ++m)
                            {
                                if (this->dict_coefs(k, this->D) == 0 && this->dict_coefs(k, m) != 0)
                                {
                                    lexmin = false; 
                                    break; 
                                }
                            }
                            if (!lexmin)
                                break; 
                        }
                        if (lexmin)
                            std::cout << this->basic_solution.transpose() << std::endl; 
                    }
                    else 
                        std::cout << this->basic_solution.transpose() << std::endl; 

                    // Reset i and j
                    i = 1; 
                    j = 0; 
                }
                // Otherwise ... 
                else 
                {
                    // ... then find and perform the Bland pivot corresponding
                    // to the current dictionary
                    std::tie(i, j) = this->pivotBland();
                    if (i == -1)
                        throw std::runtime_error("What's happening here????"); 
                    
                    // Increment i and j
                    j++;
                    if (j == this->D)
                    {
                        j = 0; 
                        i++; 
                    }
                }
                n++;
                if (n == 5)
                    break;  
            }
            while (i <= this->N || this->basis(this->N) != this->N); 
        }
}; 

}   // namespace Polytopes
