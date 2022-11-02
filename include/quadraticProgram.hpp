/**
 * A simple implementation of linear and quadratic programming with arbitrary
 * scalar types.
 *
 * **Authors:**
 *     Kee-Myoung Nam
 *
 * **Last updated:**
 *     11/2/2022
 */

#ifndef LINEAR_QUADRATIC_PROGRAMMING_SOLVER_HPP
#define LINEAR_QUADRATIC_PROGRAMMING_SOLVER_HPP

#include <iostream>
#include <cmath>
#include "linearConstraints.hpp"

using namespace Eigen; 

/**
 * Solve the given convex quadratic program with equality constraints by 
 * directly solving the associated Karush-Kuhn-Tucker system (Nocedal & Wright,
 * Eq. 16.5). 
 *
 * This function assumes that `G` is positive semidefinite, and that the
 * dimensions of `G`, `c`, `A`, and `b` match.
 *
 * @param G Symmetric matrix in quadratic program.
 * @param c Vector in quadratic program.
 * @param A Left-hand constraint matrix.
 * @param b Right-hand constraint vector. 
 */
template <typename T>
Matrix<T, Dynamic, 1> solveEqualityConstrainedConvexQuadraticProgram(const Ref<const Matrix<T, Dynamic, Dynamic> >& G,
                                                                     const Ref<const Matrix<T, Dynamic, 1> >& c,
                                                                     const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                                                     const Ref<const Matrix<T, Dynamic, 1> >& b)
{
    const int D = G.rows();    // Number of variables 
    const int N = A.rows();    // Number of constraints
    const int M = D + N;

    // Define and solve Karush-Kuhn-Tucker system (Nocedal & Wright, Eq. 16.4)
    //
    // The symmetric rewriting of the KKT system (Eq. 16.5) is not useful here
    // because the resulting matrix is generally indefinite and does not lend
    // itself to Cholesky factorization; thus, we solve the KKT system with
    // LU factorization 
    Matrix<T, Dynamic, Dynamic> A_kkt = Matrix<T, Dynamic, Dynamic>::Zero(M, M);
    Matrix<T, Dynamic, 1> b_kkt(M); 
    A_kkt(Eigen::seqN(0, D), Eigen::seqN(0, D)) = G; 
    A_kkt(Eigen::seqN(D, N), Eigen::seqN(0, D)) = A; 
    A_kkt(Eigen::seqN(0, D), Eigen::seqN(D, N)) = -A.transpose();
    b_kkt.head(D) = -c;
    b_kkt.tail(N) = b;
    Matrix<T, Dynamic, 1> sol_kkt = A_kkt.partialPivLu().solve(b_kkt);

    return sol_kkt.head(D);
}    

/**
 * Solve the convex quadratic program determined by the positive semidefinite
 * matrix `G` and vector `c`, i.e., minimize `(1/2) * x.T * G * x + c.T * x`,
 * subject to the given linear constraints. 
 *
 * @param G
 * @param c
 * @param constraints
 * @param tol
 * @param max_iter
 * @returns
 * @throws std::runtime_error If `G` is not positive semidefinite.
 */
template <typename ProgramType, typename ConstraintType = ProgramType>
Matrix<ProgramType, Dynamic, 1> solveConvexQuadraticProgram(const Ref<const Matrix<ProgramType, Dynamic, Dynamic> >& G,
                                                            const Ref<const Matrix<ProgramType, Dynamic, 1> >& c,
                                                            Polytopes::LinearConstraints<ConstraintType>* constraints,
                                                            const Ref<const Matrix<ProgramType, Dynamic, 1> >& x_init,
                                                            const ProgramType tol,
                                                            const int max_iter)
{
    const int D = constraints->getD();    // Number of variables 
    const int N = constraints->getN();    // Number of constraints
    Matrix<ProgramType, Dynamic, Dynamic> A(N, D); 
    Matrix<ProgramType, Dynamic, 1> b(N);
    Matrix<ConstraintType, Dynamic, Dynamic> A_ = constraints->getA(); 
    Matrix<ConstraintType, Dynamic, 1> b_ = constraints->getb(); 
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < D; ++j)
            A(i, j) = static_cast<ProgramType>(A_(i, j));
        b(i) = static_cast<ProgramType>(b_(i)); 
    }

    // Check that G is positive semidefinite by attempting a Cholesky decomposition
    LDLT<Matrix<ProgramType, Dynamic, Dynamic> > cholesky(G); 
    if (cholesky.info() == Eigen::NumericalIssue || cholesky.isNegative())
        throw std::runtime_error("G is not positive semidefinite");
   
    // Identify working set of active constraints at the initial iterate
    Matrix<ProgramType, Dynamic, 1> xk = x_init;
    Matrix<bool, Dynamic, 1> working = constraints->active(xk);
    
    for (int k = 0; k < max_iter; ++k)
    {
        int nw = working.count();

        // Define and solve the k-th equality-constrained convex QP subproblem
        // (Nocedal & Wright, Eq. 16.39)
        Matrix<ProgramType, Dynamic, Dynamic> Ak(nw, D);
        int i = 0;  
        for (int j = 0; j < N; ++j)
        {
            if (working(j))
            {
                Ak.row(i) = A.row(j);
                i++; 
            }
        }
        Matrix<ProgramType, Dynamic, 1> bk = Matrix<ProgramType, Dynamic, 1>::Zero(nw);
        Matrix<ProgramType, Dynamic, 1> gk = G * xk + c;
        Matrix<ProgramType, Dynamic, 1> solution
            = solveEqualityConstrainedConvexQuadraticProgram<ProgramType>(G, gk, Ak, bk);

        // If the solution to the k-th subproblem is (close to) zero ... 
        if (solution.norm() < tol)
        {
            // Compute corresponding Lagrange multipliers (Nocedal & Wright,
            // Eq. 16.42)
            Matrix<ProgramType, Dynamic, 1> lk = Ak.transpose().partialPivLu().solve(gk);

            // If all Lagrange multipliers are non-negative among all active 
            // constraints, then the current iterate is in fact a solution 
            // to the main QP 
            if ((lk.array() >= 0).all())
            {
                return xk;
            }
            // Otherwise, find the least constraint index for which the Lagrange
            // multiplier is smallest (most negative) and *remove* this constraint
            // from the working active set 
            else
            {
                Eigen::Index minidx; 
                lk.minCoeff(&minidx);
                i = 0;
                for (int j = 0; j < N; ++j)
                {
                    if (working(j))          // If the j-th constraint is active ...
                    {
                        if (i == minidx)    // ... and it is the active constraint to be removed ... 
                        {
                            working(j) = false;
                            break;
                        }
                        i++;
                    }
                }
            }
        }
        else    // If the solution to the k-th subproblem is nonzero ... 
        {
            // Compute the stepsize \alpha_k from Nocedal & Wright, Eq. 16.41,
            // keeping track of the blocking constraints 
            ProgramType alpha = std::numeric_limits<ProgramType>::infinity();
            std::vector<int> blocking; 
            for (int j = 0; j < N; ++j)
            {
                if (!working(j))
                {
                    ProgramType q = A.row(j).dot(solution);
                    if (q < 0)
                    { 
                        ProgramType bound = (b(j) - A.row(j).dot(xk)) / q;
                        if (alpha > bound)
                        {
                            alpha = bound;
                            blocking.clear();
                            blocking.push_back(j); 
                        }
                        else if (alpha == bound)
                        {
                            blocking.push_back(j); 
                        }
                    }
                } 
            }
            ProgramType stepsize = (alpha < 1 ? alpha : 1);

            // Update the current iterate 
            xk += stepsize * solution;

            // If there are any blocking constraints, choose one to add to the
            // working set 
            if (blocking.size() > 0)
                working(blocking[0]) = true;
        }
    }

    return xk; 
}

#endif
