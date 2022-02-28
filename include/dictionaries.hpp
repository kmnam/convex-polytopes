#include <iostream>
#include <stdexcept>
#include <vector>
#include <stack>
#include <utility>
#include <unordered_set>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>

/**
 * A basic implementation of a "dictionary system," which constitutes the 
 * machinery underlying Avis & Fukuda's algorithm for vertex enumeration
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
 *     2/28/2022
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
 * Exception to be thrown when a dictionary is primal feasible when it should 
 * not be (e.g., when applying dual Bland's rule). 
 */
class PrimalFeasibleException    : public std::runtime_error
{
    public:
        PrimalFeasibleException(const std::string& what = "")    : std::runtime_error(what) {}
};
/**
 * Exception to be thrown when a dictionary is dual infeasible when it should
 * not be (e.g., when applying dual Bland's rule). 
 */
class DualInfeasibleException    : public std::runtime_error
{
    public:
        DualInfeasibleException(const std::string& what = "")    : std::runtime_error(what) {}
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

class DictionarySystem
{
    protected:
        /**
         * Dimensions of the core linear system.
         */ 
        int rows;   /** Number of rows in the core linear system.    */  
        int cols;   /** Number of columns in the core linear system. */
        int f;      /** Index that always lies in the basis.         */ 
        int g;      /** Index that always lies in the cobasis.       */
        int fi;     /** Position of `f` in the basis (`this->f = this->basis(this->fi)`).     */
        int gi;     /** Position of `g` in the cobasis (`this->g = this->cobasis(this->gi)`). */

        /**
         * Boolean vector indicating the current basis and cobasis indices.
         */
        VectorXb in_basis; 

        /**
         * Integer vector containing the current basis indices.
         */ 
        VectorXi basis;
        
        /**
         * Integer vector containing the current cobasis indices.
         */
        VectorXi cobasis;  

        /**
         * Core linear system whose dictionaries are to be enumerated.
         */
        MatrixXr core_A; 

        /**
         * Dictionary coefficient matrix corresponding to the current basis.
         */
        MatrixXr dict_coefs; 
        MatrixXr basis_inv_coefs; 

        /**
         * Basic solution of the current dictionary. 
         */
        VectorXr basic_solution;

        /**
         * Indicator for whether the current basic solution is degenerate. 
         */
        bool is_degenerate;  

        /**
         * Update `this->basis` and `this->cobasis` from `this->in_basis`.
         */
        void updateBasis()
        {
            int b = 0; 
            int c = 0;
            for (int k = 0; k < this->cols; ++k) 
            {
                if (this->in_basis(k))
                {
                    this->basis(b) = k;
                    if (k == this->f)
                        this->fi = b; 
                    b++;
                }
                else 
                {
                    this->cobasis(c) = k;
                    if (k == this->g)
                        this->gi = c;  
                    c++; 
                }
            }
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
            VectorXr v = VectorXr::Zero(this->cols - this->rows); 
            v(this->gi) = 1; 
            this->basic_solution = this->dict_coefs * v;
            this->is_degenerate = (this->basic_solution.array() == 0).any();  
        }

        /**
         * Update the basis to the given vector of indices *without checking 
         * whether the basis is valid*. 
         *
         * The vector of indices should contain `this->rows` distinct entries,
         * including `this->f` and excluding `this->g`.
         *
         * @param basis Given basis indices.
         */
        void __setBasis(const Ref<const VectorXi>& basis)
        {
            // Update the basis and cobasis ...
            this->in_basis = VectorXb::Zero(this->cols); 
            for (int i = 0; i < this->rows; ++i)
                this->in_basis(basis(i)) = true;
            this->updateBasis(); 

            // ... and the dictionary coefficient matrix and basic solution
            this->updateDictCoefs();
            this->updateBasicSolution();  
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
            this->updateBasis(); 

            // Find the indices of s in the new basis and r in the new cobasis 
            int new_i, new_j; 
            for (int k = 0; k < this->rows; ++k)
            {
                if (this->basis(k) == s)
                {
                    new_i = k; 
                    break; 
                }
            }
            for (int k = 0; k < this->cols; ++k)
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
                this->updateBasis(); 

                // Throw exception to signal that the pivot failed
                throw InvalidPivotException("Specified pivot is invalid"); 
            } 
            this->updateBasicSolution();

            // Return the indices of s in the new basis and r in the new cobasis
            return std::make_pair(new_i, new_j);  
        }

    public:
        /**
         * Empty constructor. 
         */
        DictionarySystem()
        {
            // Assume 2 columns and 1 row at the very least 
            this->rows = 1; 
            this->cols = 2;
            this->f = 0;
            this->g = 1;
            this->fi = 0; 
            this->gi = 0;
            this->in_basis = VectorXb::Zero(2); 
            this->in_basis(0) = true;
            this->basis.resize(1); 
            this->cobasis.resize(1);  
            this->updateBasis(); 
            this->updateDictCoefs(); 
            this->updateBasicSolution(); 
        }

        /**
         * Constructor specifying the core system, with:
         * - `this->f` set to 0,
         * - `this->g` set to `this->cols - 1`, and
         * - basis set to 0, ..., `this->rows - 1`.
         *
         * @param A Input core matrix.
         */
        DictionarySystem(const Ref<const MatrixXr>& A)
        {
            this->core_A = A; 
            this->rows = A.rows(); 
            this->cols = A.cols();
            this->f = 0;
            this->g = this->cols - 1; 
            this->in_basis = VectorXb::Zero(this->cols);
            for (int i = 0; i < this->rows; ++i)
                this->in_basis(i) = true;
            this->basis.resize(this->rows); 
            this->cobasis.resize(this->cols - this->rows);  
            this->updateBasis(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }

        /**
         * Constructor specifying the core system and the fixed basis and 
         * cobasis indices, with the basis set to `this->f` plus the first 
         * `this->rows - 1` non-fixed indices in the system. 
         *
         * @param A Input core matrix.
         * @param f Fixed basis index. 
         * @param g Fixed cobasis index.
         * @throws std::invalid_argument If `f` or `g` are invalid (if `f` or 
         *                               `g` do not fall between 0 and
         *                               `this->cols - 1`, or if `f == g`). 
         */
        DictionarySystem(const Ref<const MatrixXr>& A, const int f, const int g)
        {
            this->core_A = A; 
            this->rows = A.rows(); 
            this->cols = A.cols();

            // Check that f and g are valid indices and are not equal
            if (f < 0 || f > this->cols - 1 || g < 0 || g > this->cols - 1 || f == g)
                throw std::invalid_argument("Invalid choices of fixed indices");  

            this->f = f; 
            this->g = g;

            // Initialize the basis to f plus the first (this->rows - 1) 
            // non-fixed indices in the system  
            this->in_basis = VectorXb::Zero(this->cols);
            this->in_basis(f) = true;  
            int i = 0; 
            int n = 1; 
            while (n < this->rows)
            {
                if (i != f)
                {
                    this->in_basis(i) = true;
                    n++;
                }
                i++; 
            }
            this->basis.resize(this->rows); 
            this->cobasis.resize(this->cols - this->rows);  
            this->updateBasis(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }

        /**
         * Constructor specifying the core system, the fixed basis and 
         * cobasis indices, and the initial basis. 
         *
         * @param A     Input core matrix.
         * @param f     Fixed basis index. 
         * @param g     Fixed cobasis index.
         * @param basis Input basis. 
         * @throws std::invalid_argument If `f` or `g` are invalid (if `f` or 
         *                               `g` do not fall between 0 and
         *                               `this->cols - 1`, or if `f == g`), or
         *                               if `basis` is invalid (see `setBasis()`). 
         */
        DictionarySystem(const Ref<const MatrixXr>& A, const int f, const int g,
                         const Ref<const VectorXi>& basis)
        {
            this->core_A = A; 
            this->rows = A.rows(); 
            this->cols = A.cols();

            // Check that f and g are valid indices and are not equal
            if (f < 0 || f > this->cols - 1 || g < 0 || g > this->cols - 1 || f == g)
                throw std::invalid_argument("Invalid choices of fixed indices");  

            this->f = f; 
            this->g = g;

            // Set the basis to the given indices
            this->in_basis.resize(this->cols); 
            this->basis.resize(this->rows); 
            this->cobasis.resize(this->cols - this->rows);  
            this->setBasis(basis);  
        } 

        /**
         * Empty destructor. 
         */
        ~DictionarySystem()
        {
        }

        /**
         * Return the core matrix.
         *
         * @returns Core matrix. 
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
         * Update the basis to the given vector of indices. 
         *
         * @param basis Given basis indices.
         * @throws std::invalid_argument If the given basis is invalid (has
         *                               wrong length, contains invalid indices,
         *                               contains duplicates, does not contain
         *                               `this->f`, or contains `this->g`). 
         */
        void setBasis(const Ref<const VectorXi>& basis)
        {
            // Check that the basis contains f, does not contain g, and
            // consists of (this->rows) distinct indices
            if (basis.size() != this->rows)
                throw std::invalid_argument("Invalid basis specified"); 
            std::unordered_set<int> indices; 
            for (int i = 0; i < this->rows; ++i)
            {
                if (basis(i) < 0 || basis(i) >= this->cols || basis(i) == this->g)
                    throw std::invalid_argument("Specified basis contains invalid index"); 
                indices.insert(basis(i));
            }
            if (indices.size() != this->rows)
                throw std::invalid_argument("Specified basis contains duplicate indices"); 
            if (indices.find(this->f) == indices.end())
                throw std::invalid_argument("Specified basis does not contain 0");

            this->__setBasis(basis); 
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
            if (i == this->fi) 
                throw std::invalid_argument("f is undefined regarding primal feasibility"); 
            else if (i < 0 || i >= this->rows)
                throw std::invalid_argument("Invalid basis index specified"); 

            return (this->dict_coefs(i, this->gi) >= 0); 
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
            if (i == this->gi)
                throw std::invalid_argument("g is undefined regarding dual feasibility"); 
            else if (i < 0 || i >= this->cols - this->rows)
                throw std::invalid_argument("Invalid cobasis index specified"); 

            return (this->dict_coefs(this->fi, i) <= 0);
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
            if (i == this->fi)
                throw std::invalid_argument("Undefined pivot: basis must always contain f"); 
            else if (i < 0 || i >= this->rows)
                throw std::invalid_argument("Undefined pivot: invalid basis index specified");

            // Check whether j is indeed a valid cobasis index
            if (j == this->gi)
                throw std::invalid_argument("Undefined pivot: cobasis must always contain g"); 
            else if (j < 0 || j >= this->cols - this->rows)
                throw std::invalid_argument("Undefined pivot: invalid cobasis index specified"); 

            // Check whether the dictionary is primal and/or dual feasible
            bool all_primal_curr = true; 
            bool all_dual_curr = true; 
            for (int k = 0; k < this->rows; ++k)
            {
                if (all_primal_curr && k != this->fi && !this->isPrimalFeasible(k))
                {
                    all_primal_curr = false;
                    break; 
                } 
            }
            for (int k = 0; k < this->cols - this->rows; ++k)
            {
                if (all_dual_curr && k != this->gi && !this->isDualFeasible(k))
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
            for (int k = 0; k < this->rows; ++k)
            {
                if (all_primal_next && k != this->fi && !this->isPrimalFeasible(k))
                {
                    all_primal_next = false;
                    break;
                }
            }
            for (int k = 0; k < this->cols - this->rows; ++k)
            {
                if (all_dual_next && k != this->gi && !this->isDualFeasible(k))
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
            for (int k = 0; k < this->rows; ++k)
            {
                if (k != this->fi && !this->isPrimalFeasible(k))
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
            for (int k = 0; k < this->cols - this->rows; ++k)
            {
                if (k != this->gi && !this->isDualFeasible(k))
                {
                    j = k;
                    break; 
                } 
            }

            // If no such index exists, then throw an exception
            if (j == -1)
                throw DualFeasibleException(
                    "Bland's rule cannot be performed on dual feasible dictionary"
                ); 

            // Find the least index in the basis (other than f) obtaining the
            // minimum value of -(dict_coefs(i, g) / dict_coefs(i, j))
            int i = -1;
            mpq_rational lambda = static_cast<mpq_rational>(std::numeric_limits<int>::max()); 
            for (int k = 0; k < this->rows; ++k)
            {
                if (k != this->fi)
                {
                    mpq_rational denom = this->dict_coefs(k, j); 
                    if (denom < 0)
                    {
                        mpq_rational value = -this->dict_coefs(k, this->gi) / denom; 
                        if (lambda > value) 
                        {
                            lambda = value;
                            i = k;
                        }
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

            // Perform the pivot only if it is actually allowed (here, the 
            // pivot should always be allowed)
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
         * Find the pivot determined by dual Bland's rule on the current
         * dictionary and return the indices being switched.
         * 
         * @returns The basis position (`k` if basis index is `this->basis(k)`)
         *          and cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *          being switched by the dual Bland pivot. 
         * @throws DualInfeasibleException If the current dictionary is dual
         *                                 infeasible.
         * @throws PrimalFeasibleException If the current dictionary is primal
         *                                 feasible.
         */
        std::pair<int, int> findDualBland()
        {
            // Check whether the dictionary is dual feasible 
            for (int k = 0; k < this->cols - this->rows; ++k)
            {
                if (k != this->gi && !this->isDualFeasible(k))
                {
                    throw DualInfeasibleException(
                        "Dual Bland's rule cannot be performed on dual infeasible "
                        "dictionary"
                    ); 
                }
            } 

            // Find the least index in the basis such that the corresponding
            // variable is primal infeasible 
            int i = -1;
            for (int k = 0; k < this->rows; ++k)
            {
                if (k != this->fi && !this->isPrimalFeasible(k))
                {
                    i = k;
                    break; 
                } 
            }

            // If no such index exists, then throw an exception
            if (i == -1)
                throw PrimalFeasibleException(
                    "Dual Bland's rule cannot be performed on dual feasible "
                    "dictionary"
                ); 

            // Find the least index in the cobasis (other than g) obtaining
            // the minimum value of -(dict_coefs(f, j) / dict_coefs(i, j)) 
            int j = -1;
            mpq_rational lambda = static_cast<mpq_rational>(std::numeric_limits<int>::max()); 
            for (int k = 0; k < this->cols - this->rows; ++k)
            {
                if (k != this->gi)
                {
                    mpq_rational denom = this->dict_coefs(i, k); 
                    if (denom < 0)
                    {
                        mpq_rational value = -this->dict_coefs(this->fi, k) / denom; 
                        if (lambda > value) 
                        {
                            lambda = value;
                            j = k;
                        }
                    }
                }
            }

            // Return the chosen indices 
            return std::make_pair(i, j); 
        }

        /**
         * Perform the pivot determined by dual Bland's rule on the current
         * dictionary.
         *
         * @returns The basis position (`k` if basis index is `this->basis(k)`)
         *          and cobasis position (`k` if cobasis index is `this->cobasis(k)`)
         *          in the *new* dictionary obtained after the pivot. 
         * @throws PrimalFeasibleException If the current dictionary is primal
         *                                 feasible (rethrown exception from
         *                                 `findDualBland()`).
         * @throws DualInfeasibleException If the current dictionary is dual
         *                                 infeasible (rethrown exception from
         *                                 `findDualBland()`).
         * @throws InvalidPivotException If the pivot could not be performed 
         *                               (rethrown exception from `__pivot()`). 
         */
        std::pair<int, int> pivotDualBland()
        {
            int i, j; 
            try
            { 
                std::tie(i, j) = this->findDualBland();
            }
            catch (const PrimalFeasibleException& e)
            {
                throw; 
            }
            catch (const DualInfeasibleException& e)
            {
                throw; 
            }

            // Perform the pivot only if it is actually allowed (here, the 
            // pivot should always be allowed)
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
            bool ith_var_in_basis = false; 
            for (int k = 0; k < this->cols; ++k)
            {
                if (this->in_basis(k) && k != this->f)        // Is a basis variable other than f
                {
                    if (!this->isPrimalFeasible(b))
                    {
                        i = b;
                        ith_var_in_basis = true; 
                        break;
                    } 
                    b++; 
                }
                else if (k == this->f)
                {
                    b++; 
                }
                else if (!this->in_basis(k) && k != this->g)  // Is a cobasis variable other than g
                {
                    if (!this->isDualFeasible(c))
                    {
                        i = c;
                        ith_var_in_basis = false; 
                        break; 
                    }
                    c++;
                }
                else
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
                // Find the least cobasis index, this->cobasis(j), other 
                // than g such that dict_coefs(i, j) > 0
                for (int j = 0; j < this->cols - this->rows; ++j)
                {
                    if (j != this->gi && this->dict_coefs(i, j) > 0)
                        return std::make_pair(i, j); 
                }   // Note that there must exist at least one such index 
            }
            // Otherwise ... 
            else
            {
                // Find the least basis index, this->basis(j), other than
                // f such that dict_coefs(j, i) < 0
                for (int j = 0; j < this->rows; ++j)
                {
                    if (j != this->fi && this->dict_coefs(j, i) < 0)
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
         * Determine if the given pivot is a reverse Bland pivot.
         *
         * @param j Position of basis variable in the current basis. 
         * @param i Position of cobasis variable in the current cobasis. 
         * @returns True if the pivot is a reverse Bland pivot, false otherwise.
         * @throws InvalidPivotException If the reverse pivot cannot be performed
         *                               or reversed.
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
                bland_i = -1; 
            }
            catch (const DualFeasibleException& e)
            {
                bland_i = -1; 
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
            return (bland_i != -1 && new_j == bland_i && new_i == bland_j);  
        }

        /**
         * Determine if the given pivot is a reverse dual Bland pivot.
         *
         * @param j Position of basis variable in the current basis. 
         * @param i Position of cobasis variable in the current cobasis. 
         * @returns True if the pivot is a reverse dual Bland pivot, false
         *          otherwise.
         * @throws InvalidPivotException If the reverse pivot cannot be performed
         *                               or reversed.
         */
        bool isReverseDualBlandPivot(const int j, const int i) 
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

            // Now find the pivot corresponding to dual Bland's rule for this 
            // new dictionary 
            int dbland_i, dbland_j;
            try
            { 
                std::tie(dbland_i, dbland_j) = this->findDualBland();
            }
            catch (const PrimalFeasibleException& e) 
            {
                dbland_i = -1; 
            }
            catch (const DualInfeasibleException& e)
            {
                dbland_i = -1;
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

            // Is this reverse of the reverse pivot the same as the dual Bland
            // pivot? 
            return (dbland_i != -1 && new_j == dbland_i && new_i == dbland_j);  
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
                cc_i = -1;  
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
            return (cc_i != -1 && new_j == cc_i && new_i == cc_j); 
        }

        /**
         * Return the array of all possible reverse Bland pivots from the 
         * current dictionary, given in lexicographic order.
         *
         * The returned array has two columns, and each row contains the basis
         * and cobasis index forming each pivot.
         *
         * @returns Array of all possible reverse Bland pivots from the 
         *          current dictionary, given in lexicographic order. 
         */
        MatrixXi getReverseBlandPivots()
        {
            int n = 0; 
            MatrixXi reverse_bland(n, 2); 

            for (int i = 0; i < this->rows; ++i)
            {
                if (i != this->fi)
                {
                    for (int j = 0; j < this->cols - this->rows; ++j)
                    {
                        if (j != this->gi)
                        {
                            // Check whether (i, j) is a reverse Bland pivot
                            //
                            // Note that this can raise an InvalidPivotException,
                            // in which case (i, j) is not even a valid pivot to
                            // begin with 
                            bool is_reverse_bland;  
                            try
                            {
                                is_reverse_bland = this->isReverseBlandPivot(i, j); 
                            }
                            catch (const InvalidPivotException& e)
                            {
                                continue; 
                            }
                            if (is_reverse_bland)
                            {
                                n++; 
                                reverse_bland.conservativeResize(n, 2); 
                                reverse_bland(n-1, 0) = i; 
                                reverse_bland(n-1, 1) = j; 
                            }
                        }
                    }
                }
            }

            return reverse_bland;
        }

        /**
         * Return the array of all possible reverse criss-cross pivots from the 
         * current dictionary, given in lexicographic order.
         *
         * The returned array has two columns, and each row contains the basis
         * and cobasis index forming each pivot.
         *
         * @returns Array of all possible reverse criss-cross pivots from the 
         *          current dictionary, given in lexicographic order. 
         */
        MatrixXi getReverseCrissCrossPivots()
        {
            int n = 0; 
            MatrixXi reverse_cc(n, 2); 

            for (int i = 0; i < this->rows; ++i)
            {
                if (i != this->fi)
                {
                    for (int j = 0; j < this->cols - this->rows; ++j)
                    {
                        if (j != this->gi)
                        {
                            // Check whether (i, j) is a reverse criss-cross pivot
                            //
                            // Note that this can raise an InvalidPivotException,
                            // in which case (i, j) is not even a valid pivot to
                            // begin with 
                            bool is_reverse_cc; 
                            try
                            {
                                is_reverse_cc = this->isReverseCrissCrossPivot(i, j); 
                            }
                            catch (const InvalidPivotException& e)
                            {
                                continue; 
                            }
                            if (is_reverse_cc)
                            {
                                n++; 
                                reverse_cc.conservativeResize(n, 2); 
                                reverse_cc(n-1, 0) = i; 
                                reverse_cc(n-1, 1) = j; 
                            }
                        }
                    }
                }
            }

            return reverse_cc;
        }

        /**
         * Perform a depth-first search of all *primal feasible* dictionaries
         * accessible from the current dictionary via reverse Bland pivots,
         * which is assumed to be optimal.
         *
         * @returns Array of *primal feasible* bases in the order in which
         *          they were encountered.
         * @throws PrimalInfeasibleException If the initial dictionary is 
         *                                   primal infeasible. 
         * @throws DualInfeasibleException   If the initial dictionary is 
         *                                   dual infeasible. 
         */
        MatrixXi searchFromOptimalDictBland()
        {
            // Check that the search begins at an optimal dictionary
            for (int i = 0; i < this->rows; ++i)
            {
                if (i != this->fi && !this->isPrimalFeasible(i))
                    throw PrimalInfeasibleException("Search must begin at optimal dictionary");
            }
            for (int i = 0; i < this->cols - this->rows; ++i)
            {
                if (i != this->gi && !this->isDualFeasible(i))
                    throw DualInfeasibleException("Search must begin at optimal dictionary"); 
            }

            // Maintain a stack of bases and their possible reverse Bland pivots ... 
            std::stack<VectorXi> pivots;

            // ... and a stack of performed reverse Bland pivots
            std::stack<std::pair<int, int> > reverses_of_performed_pivots;  

            // Initialize the former stack with all possible reverse Bland pivots
            // from the initial optimal dictionary
            //
            // Add the reverse pivots in rev-lex-min order, so that they can 
            // be performed in lex-min order when popping off the stack 
            MatrixXi reverse_bland = this->getReverseBlandPivots();
            for (int i = reverse_bland.rows() - 1; i >= 0; --i)
            {
                VectorXi pivot(this->rows + 2);
                pivot.head(this->rows) = this->basis;  
                pivot.tail(2) = reverse_bland.row(i);
                pivots.push(pivot);  
            }

            // Initialize array of bases encountered during the search 
            int n = 1; 
            MatrixXi bases(n, this->rows); 
            bases.row(n-1) = this->basis; 

            // While the stack is non-empty ... 
            while (!pivots.empty())
            {
                // Pop the next reverse Bland pivot from the stack 
                VectorXi next = pivots.top();
                pivots.pop();

                // If the current basis does not match the basis corresponding 
                // to this next pivot, reverse as many performed pivots as 
                // needed to restore the latter basis
                while (this->basis != next.head(this->rows))
                {
                    std::pair<int, int> prev = reverses_of_performed_pivots.top(); 
                    reverses_of_performed_pivots.pop();
                    this->__pivot(prev.first, prev.second);  
                }

                // Perform the pivot
                int new_i, new_j; 
                std::tie(new_i, new_j) = this->__pivot(next(this->rows), next(this->rows+1));
                reverses_of_performed_pivots.emplace(std::make_pair(new_i, new_j));
                n++;
                bases.conservativeResize(n, this->rows);  
                bases.row(n-1) = this->basis; 

                // Collect all possible reverse Bland pivots from the current
                // dictionary
                //
                // Add the reverse pivots in rev-lex-min order, so that they can 
                // be performed in lex-min order when popping off the stack 
                reverse_bland = this->getReverseBlandPivots();
                for (int i = reverse_bland.rows() - 1; i >= 0; --i)
                {
                    VectorXi pivot(this->rows + 2);
                    pivot.head(this->rows) = this->basis;  
                    pivot.tail(2) = reverse_bland.row(i);
                    pivots.push(pivot);  
                } 
            }

            // Restore the original optimal dictionary by reversing all 
            // remaining performed pivots 
            while (!reverses_of_performed_pivots.empty())
            {
                std::pair<int, int> prev = reverses_of_performed_pivots.top(); 
                reverses_of_performed_pivots.pop(); 
                this->__pivot(prev.first, prev.second); 
            }

            return bases; 
        }

        /**
         * Perform a depth-first search of all *primal feasible* dictionaries
         * accessible from the current dictionary via reverse criss-cross pivots,
         * which is assumed to be optimal.
         *
         * @returns Array of bases in the order in which they were encountered.
         * @throws PrimalInfeasibleException If the initial dictionary is 
         *                                   primal infeasible. 
         * @throws DualInfeasibleException   If the initial dictionary is 
         *                                   dual infeasible. 
         */
        MatrixXi searchFromOptimalDictCrissCross()
        {
            // Check that the search begins at an optimal dictionary
            for (int i = 0; i < this->rows; ++i)
            {
                if (i != this->fi && !this->isPrimalFeasible(i))
                    throw PrimalInfeasibleException("Search must begin at optimal dictionary");
            }
            for (int i = 0; i < this->cols - this->rows; ++i)
            {
                if (i != this->gi && !this->isDualFeasible(i))
                    throw DualInfeasibleException("Search must begin at optimal dictionary"); 
            }

            // Maintain a stack of bases and their possible reverse criss-cross
            // pivots ... 
            std::stack<VectorXi> pivots;

            // ... and a stack of performed reverse criss-cross pivots
            std::stack<std::pair<int, int> > reverses_of_performed_pivots;  

            // Initialize the former stack with all possible reverse criss-cross
            // pivots from the initial optimal dictionary
            //
            // Add the reverse pivots in rev-lex-min order, so that they can 
            // be performed in lex-min order when popping off the stack 
            MatrixXi reverse_cc = this->getReverseCrissCrossPivots();
            for (int i = reverse_cc.rows() - 1; i >= 0; --i)
            {
                VectorXi pivot(this->rows + 2);
                pivot.head(this->rows) = this->basis;  
                pivot.tail(2) = reverse_cc.row(i);
                pivots.push(pivot);  
            }

            // Initialize array of bases encountered during the search 
            int n = 1; 
            MatrixXi bases(n, this->rows); 
            bases.row(n-1) = this->basis; 

            // While the stack is non-empty ... 
            while (!pivots.empty())
            {
                // Pop the next reverse criss-cross pivot from the stack 
                VectorXi next = pivots.top();
                pivots.pop();

                // If the current basis does not match the basis corresponding 
                // to this next pivot, reverse as many performed pivots as 
                // needed to restore the latter basis
                while (this->basis != next.head(this->rows))
                {
                    std::pair<int, int> prev = reverses_of_performed_pivots.top(); 
                    reverses_of_performed_pivots.pop();
                    this->__pivot(prev.first, prev.second);  
                }

                // Perform the pivot
                int new_i, new_j; 
                std::tie(new_i, new_j) = this->__pivot(next(this->rows), next(this->rows+1));
                reverses_of_performed_pivots.emplace(std::make_pair(new_i, new_j));
                n++;
                bases.conservativeResize(n, this->rows);  
                bases.row(n-1) = this->basis; 

                // Collect all possible reverse criss-cross pivots from the 
                // current dictionary
                //
                // Add the reverse pivots in rev-lex-min order, so that they can 
                // be performed in lex-min order when popping off the stack 
                reverse_cc = this->getReverseCrissCrossPivots();
                for (int i = reverse_cc.rows() - 1; i >= 0; --i)
                {
                    VectorXi pivot(this->rows + 2);
                    pivot.head(this->rows) = this->basis;  
                    pivot.tail(2) = reverse_cc.row(i);
                    pivots.push(pivot);  
                } 
            }

            // Restore the original optimal dictionary by reversing all 
            // remaining performed pivots 
            while (!reverses_of_performed_pivots.empty())
            {
                std::pair<int, int> prev = reverses_of_performed_pivots.top(); 
                reverses_of_performed_pivots.pop(); 
                this->__pivot(prev.first, prev.second); 
            }

            return bases; 
        }
}; 

}   // namespace Polytopes
