/**
 * A basic implementation of Avis & Fukuda's pivoting algorithm for vertex
 * enumeration in convex polytopes and hyperplane arrangements, from:
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
 *     11/13/2022
 */

#ifndef VERTEX_ENUM_AVIS_FUKUDA_HPP
#define VERTEX_ENUM_AVIS_FUKUDA_HPP 

#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "linearConstraints.hpp"
#include "dictionaries.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational; 

namespace Polytopes {

class PolyhedralDictionarySystem : public DictionarySystem, public LinearConstraints
{
    protected:
        /**
         * Update the core linear system from the current set of linear constraints. 
         */
        void updateCore()
        {
            this->rows = this->N + 1; 
            this->cols = this->N + this->D + 2;
            this->f = 0; 
            this->fi = 0; 
            this->g = this->N + this->D + 1;
            this->gi = this->D;

            // Initialize the basis and cobasis, as: 
            //
            // basis   = {0, 1, ..., this->N}
            // cobasis = {this->N + 1, ..., this->N + this->D, this->N + this->D + 1}
            //
            // this->f = 0 always remains in the basis
            //
            // this->g = this->N + this->D + 1 always remains in the cobasis
            this->in_basis = VectorXb::Zero(this->cols); 
            this->in_basis.head(this->rows) = VectorXb::Ones(this->rows);
            this->basis.resize(this->rows); 
            this->cobasis.resize(this->cols - this->rows); 
            this->updateBasis();

            // Construct the core linear system
            this->core_A = MatrixXr::Zero(this->rows, this->cols); 
            this->core_A(0, this->cobasis.head(this->cols - this->rows - 1)) = VectorXr::Ones(this->cols - this->rows - 1); 
            this->core_A(0, this->fi) = 1;
            this->core_A(Eigen::seqN(1, this->rows - 1), this->basis.tail(this->rows - 1))
                = MatrixXr::Identity(this->rows - 1, this->rows - 1);
            this->core_A(Eigen::seqN(1, this->rows - 1), this->cobasis.head(this->cols - this->rows - 1)) = this->A; 
            this->core_A(Eigen::seqN(1, this->rows - 1), this->cols - 1) = -this->b;
        }

    public:
        /**
         * Constructor specifying the polytope constraints. 
         *
         * @param type Inequality type. 
         * @param A    Left-hand matrix in polytope constraints.
         * @param b    Right-hand vector in polytope constraints. 
         */
        PolyhedralDictionarySystem(const InequalityType type) 
            : DictionarySystem(), LinearConstraints(type)
        {
            this->updateCore(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }
        
        /**
         * Constructor specifying the polytope constraints. 
         *
         * @param type Inequality type. 
         * @param A    Left-hand matrix in polytope constraints.
         * @param b    Right-hand vector in polytope constraints. 
         */
        PolyhedralDictionarySystem(const InequalityType type, 
                                   const Ref<const MatrixXr>& A, 
                                   const Ref<const VectorXr>& b)
            : DictionarySystem(), LinearConstraints(type, A, b) 
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
            try
            {
                this->__parse(filename, this->type);
            }
            catch (const std::invalid_argument& e)
            {
                throw; 
            }
            this->removeRedundantConstraints();
        }

        /**
         * Given a file specifying a convex polytope in terms of half-spaces 
         * (inequalities), read in the constraint matrix and vector, remove 
         * redundant constraints, and overwrite the dictionary system. 
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
            this->removeRedundantConstraints(); 
        }

        /**
         * Switch the inequality type of the stored constraints. 
         *
         * In addition to negating both `this->A` and `this->b` and updating 
         * the nearest-point-by-L2-distance quadratic program, this method
         * also updates the core matrix, dictionary coefficient matrix, and 
         * current basic solution. 
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
            this->updateCore();
            this->updateDictCoefs(); 
            this->updateBasicSolution(); 
        }

        /**
         * Remove all redundant constraints by iterating through them in the 
         * order they are given in `this->A` and `this->b`, then update 
         * `this->core_A`, `this->dict_coefs`, and `this->basic_solution`.  
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
            this->updateCore(); 
            this->updateDictCoefs(); 
            this->updateBasicSolution(); 
        }

        /**
         * Return the vertex associated with the current basic solution.
         *
         * @returns Vertex associated with the current basic solution.
         */
        VectorXr getVertex()
        {
            VectorXr solution_extended = VectorXr::Zero(this->cols); 
            solution_extended(this->basis) = this->basic_solution;
            return solution_extended(Eigen::seqN(this->N + 1, this->D)); 
        }

        /**
         * Enumerate the vertices of the convex polytope via the Avis-Fukuda 
         * algorithm, i.e., a depth-first-search traversal of all primal
         * feasible dictionaries of the core system.
         *
         * The traversal is performed via reverse Bland pivots. 
         *
         * @returns Matrix of vertex coordinates, with each row a distinct
         *          vertex of the polytope. 
         */
        MatrixXr enumVertices()
        {
            // Check that the initial dictionary is optimal ...
            bool is_optimal = true; 
            for (int i = 1; i <= this->N; ++i)
            {
                if (!this->isPrimalFeasible(i))
                {
                    is_optimal = false;
                    break; 
                }
            }
            for (int i = 0; i < this->D; ++i)
            {
                if (!this->isDualFeasible(i))
                {
                    is_optimal = false;
                    break;
                }
            }

            // ... and if it is *not* optimal, then change the basis to its
            // initial form: 
            //
            // basis   = {0, 1, ..., this->N}
            // cobasis = {this->N + 1, ..., this->N + this->D, this->N + this->D + 1}
            if (!is_optimal)
            {
                VectorXi new_basis(this->N + 1); 
                for (int i = 0; i <= this->N; ++i)
                    new_basis(i) = i; 
                this->__setBasis(new_basis); 
            }

            // Search for all optimal dictionaries accessible via reverse
            // dual Bland pivots from the current optimal dictionary 
            MatrixXi opt_bases = this->findOptimalDicts();

            // From each such optimal dictionary, run a separate depth-first
            // search via reverse Bland pivots
            int n = 0; 
            MatrixXr vertices(n, this->D); 
            for (int i = 0; i < opt_bases.rows(); ++i)
            {
                this->__setBasis(opt_bases.row(i));
                MatrixXi bases; 
                MatrixXr solutions; 
                std::tie(bases, solutions) = this->searchFromOptimalDictBland();
                for (int j = 0; j < solutions.rows(); ++j)
                {
                    VectorXr solution_extended = VectorXr::Zero(this->cols);
                    solution_extended(bases.row(j)) = solutions.row(j);
                    VectorXr vertex = solution_extended(Eigen::seqN(this->N + 1, this->D));

                    // Add the vertex to the matrix to be returned if it has not
                    // been encountered yet
                    bool vertex_prev_found = false; 
                    for (int k = 0; k < n; ++k)
                    {
                        if ((vertices.array().row(k) == vertex.transpose().array()).all())
                        {
                            vertex_prev_found = true;
                            break; 
                        }
                    }
                    if (!vertex_prev_found)
                    {
                        n++; 
                        vertices.conservativeResize(n, this->D); 
                        vertices.row(n-1) = vertex; 
                    }
                }
            }

            return vertices;  
        }
};

/**
 * TODO Write tests for this class
 */
class HyperplaneArrangement : public DictionarySystem, public LinearConstraints
{
    protected:
        /**
         * Update the core linear system from the current set of linear constraints. 
         */
        void updateCore()
        {
            this->rows = this->N - this->D + 1; 
            this->cols = this->D + 1; 
            this->f = this->N; 
            this->fi = this->N - this->D;
            this->g = this->N + 1;  
            this->gi = this->D;

            // Initialize the basis and cobasis, as: 
            //
            // basis   = {0, 1, ..., this->N - this->D - 1, this->N}
            // cobasis = {this->N - this->D, ..., this->N - 1, this->N + 1}
            //
            // this->f = this->N always remains in the basis
            //
            // this->g = this->N + 1 always remains in the cobasis
            this->in_basis = VectorXb::Zero(this->cols); 
            this->in_basis.head(this->rows - 1) = VectorXb::Ones(this->rows - 1);
            this->in_basis(this->f) = true; 
            this->basis.resize(this->rows); 
            this->cobasis.resize(this->cols - this->rows); 
            this->updateBasis();

            // Divide A and b into submatrices and subvectors according to its
            // basis and cobasis variables (other than f and g)
            MatrixXr A_basis = -this->A(Eigen::seqN(0, this->N - this->D), Eigen::all); 
            MatrixXr A_cobasis = -this->A(Eigen::seqN(this->N - this->D, this->D), Eigen::all); 
            VectorXr b_basis = this->b(Eigen::seqN(0, this->N - this->D)); 
            VectorXr b_cobasis = this->b(Eigen::seqN(this->N - this->D, this->D)); 

            // Assume that the (square) submatrix of A corresponding to the 
            // cobasis variables is invertible
            MatrixXr inv_A_cobasis; 
            try
            {
                inv_A_cobasis = invertRational(A_cobasis);      
            }
            catch (const SingularMatrixException& e)
            {
                throw; 
            }
            MatrixXr A_transformed = -A_basis * inv_A_cobasis;
            VectorXr b_transformed = b_basis + A_transformed * b_cobasis;

            // Construct the core linear system
            this->core_A = MatrixXr::Zero(this->rows, this->cols);
            this->core_A(Eigen::all, this->basis) = MatrixXr::Identity(this->rows, this->rows);
            this->core_A(Eigen::seqN(0, this->rows - 1), this->cobasis.head(this->D)) = A_transformed;  
            this->core_A(this->rows - 1, this->cobasis.head(this->D)) = VectorXr::Ones(this->D); 
            this->core_A(Eigen::seqN(0, this->rows - 1), this->g) = -b_transformed;
        }

    public:
        /**
         * Empty constructor.
         */
        HyperplaneArrangement()
            : DictionarySystem(), LinearConstraints(InequalityType::IsEqualTo) 
        {
            this->updateCore(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }

        /**
         * Constructor specifying the hyperplanes. 
         *
         * @param A Left-hand matrix in hyperplane arrangement.
         * @param b Right-hand vector in hyperplane arrangement.
         */
        HyperplaneArrangement(const Ref<const MatrixXr>& A, const Ref<const VectorXr>& b)
            : DictionarySystem(), LinearConstraints(InequalityType::IsEqualTo, A, b) 
        {
            this->updateCore(); 
            this->updateDictCoefs();
            this->updateBasicSolution(); 
        }

        /**
         * Empty destructor. 
         */
        ~HyperplaneArrangement()
        {
        }

        /**
         * Given a file specifying a hyperplane arrangement, read in the
         * constraint matrix and vector, remove redundant constraints, and
         * overwrite the dictionary system. 
         *
         * @param filename Path to file containing the polytope constraints.
         */
        void parse(const std::string filename)
        {
            std::cout << "are you here at least?\n";
            try
            {
                this->__parse(filename, InequalityType::IsEqualTo);
            }
            catch (const std::invalid_argument& e)
            {
                throw; 
            }
            std::cout << "what about here?\n";
            std::cout << this->D << std::endl;
            std::cout << this->N << std::endl;
            this->removeRedundantConstraints();
        }

        /**
         * Remove all redundant constraints by iterating through them in the 
         * order they are given in `this->A` and `this->b`, then update 
         * `this->core_A`, `this->dict_coefs`, and `this->basic_solution`.  
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
            this->updateCore(); 
            this->updateDictCoefs(); 
            this->updateBasicSolution(); 
        }

        /**
         * Return the vertex associated with the current basic solution.
         *
         * @returns Vertex associated with the current basic solution.
         */
        VectorXr getVertex()
        {
            VectorXr x = VectorXr::Zero(this->N);
            int b = 0; 
            for (int i = 0; i < this->N; ++i)
            {
                if (this->in_basis(i))
                {
                    x(i) = this->dict_coefs(b, this->gi); 
                    b++;
                }
            }

            MatrixXr M = invertRational(-this->A(Eigen::seqN(this->N - this->D, this->D), Eigen::all)); 
            MatrixXr y = x.tail(this->D) - this->b.tail(this->D); 
            return M * y; 
        }

        /**
         * Enumerate the vertices of the hyperplane arrangement via the
         * Avis-Fukuda algorithm, i.e., a depth-first-search traversal of all
         * dictionaries of the core system.
         *
         * The traversal is performed via reverse criss-cross pivots. 
         *
         * @returns Matrix of vertex coordinates, with each row a distinct
         *          vertex of the arrangement.  
         */
        MatrixXr enumVertices()
        {
            // Check that the initial dictionary is optimal ...
            bool is_optimal = true; 
            for (int i = 0; i < this->basis.size() - 1; ++i)
            {
                if (i != this->fi && !this->isPrimalFeasible(i))
                {
                    is_optimal = false;
                    break; 
                }
            }
            for (int i = 0; i < this->cobasis.size() - 1; ++i)
            {
                if (!this->isDualFeasible(i))
                {
                    is_optimal = false;
                    break;
                }
            }

            // ... and if it is *not* optimal, then change the basis to its 
            // initial form:
            //
            // basis   = {0, 1, ..., this->N - this->D - 1, this->N}
            // cobasis = {this->N - this->D, ..., this->N - 1, this->N + 1}
            if (!is_optimal)
            {
                VectorXi new_basis(this->N - this->D + 1); 
                for (int i = 0; i < this->N - this->D; ++i)
                    new_basis(i) = i;
                new_basis(this->N - this->D) = this->N; 
                this->__setBasis(new_basis); 
            }

            // Search for all optimal dictionaries accessible via reverse
            // dual Bland pivots from the current optimal dictionary 
            MatrixXi opt_bases = this->findOptimalDicts();

            // From each such optimal dictionary, run a separate depth-first
            // search via reverse criss-cross pivots
            int n = 0; 
            MatrixXr vertices(n, this->D); 
            for (int i = 0; i < opt_bases.rows(); ++i)
            {
                this->__setBasis(opt_bases.row(i));
                MatrixXi bases; 
                MatrixXr solutions; 
                std::tie(bases, solutions) = this->searchFromOptimalDictCrissCross();
                for (int j = 0; j < solutions.rows(); ++j)
                {
                    // Get the vertex associated with the j-th basic solution 
                    this->__setBasis(bases.row(j)); 
                    VectorXr vertex = this->getVertex();

                    // Add the vertex to the matrix to be returned if it has not
                    // been encountered yet
                    bool vertex_prev_found = false; 
                    for (int k = 0; k < n; ++k)
                    {
                        if ((vertices.array().row(k) == vertex.transpose().array()).all())
                        {
                            vertex_prev_found = true;
                            break; 
                        }
                    }
                    if (!vertex_prev_found)
                    {
                        n++; 
                        vertices.conservativeResize(n, this->D); 
                        vertices.row(n-1) = vertex; 
                    }
                }
            }

            return vertices;  
        }

}; 

}   // namespace Polytopes

#endif
