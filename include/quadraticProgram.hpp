/**
 * A simple implementation of linear and quadratic programming with arbitrary
 * scalar types.
 *
 * **Authors:**
 *     Kee-Myoung Nam
 *
 * **Last updated:**
 *     11/4/2022
 */

#ifndef LINEAR_QUADRATIC_PROGRAMMING_SOLVER_HPP
#define LINEAR_QUADRATIC_PROGRAMMING_SOLVER_HPP

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen; 

/**
 * Solve the given convex quadratic program with equality constraints by 
 * directly solving the associated Karush-Kuhn-Tucker system (Nocedal & Wright,
 * Eq. 16.5). 
 *
 * This function assumes that `G` is positive semidefinite, and that the
 * dimensions of `G`, `c`, `A`, and `b` match.
 *
 * @param G Symmetric matrix in quadratic objective.
 * @param c Vector in linear part of objective.
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
 * subject to the constraint `A * x >= b`. 
 *
 * @param G        Symmetric matrix in quadratic objective.
 * @param c        Vector in linear part of objective. 
 * @param A        Left-hand constraint matrix.
 * @param b        Right-hand constraint vector.
 * @param x_init   Initial iterate (must satisfy constraints).
 * @param tol      Tolerance for assessing whether a stepsize is zero.
 * @param max_iter Maximum number of iterations.
 * @returns        Solution vector (including the associated Lagrange multipliers),
 *                 together with indicator of whether algorithm terminated
 *                 within given maximum number of iterations.  
 * @throws std::runtime_error If `G` is not positive semidefinite.
 */
template <typename T>
std::pair<Matrix<T, Dynamic, 1>, bool> solveConvexQuadraticProgram(const Ref<const Matrix<T, Dynamic, Dynamic> >& G,
                                                                   const Ref<const Matrix<T, Dynamic, 1> >& c,
                                                                   const Ref<const Matrix<T, Dynamic, Dynamic> >& A,
                                                                   const Ref<const Matrix<T, Dynamic, 1> >& b,
                                                                   const Ref<const Matrix<T, Dynamic, 1> >& x_init,
                                                                   const T tol,
                                                                   const int max_iter)
{
    const int D = A.cols();    // Number of variables 
    const int N = A.rows();    // Number of constraints

    // Check that G is positive semidefinite by attempting a Cholesky decomposition
    LDLT<Matrix<T, Dynamic, Dynamic> > cholesky(G); 
    if (cholesky.info() == Eigen::NumericalIssue || cholesky.isNegative())
        throw std::runtime_error("G is not positive semidefinite");
   
    // Identify working set of active constraints at the initial iterate
    Matrix<T, Dynamic, 1> xk = x_init;
    Matrix<bool, Dynamic, 1> working = ((A * xk).array() == b.array()).matrix();
    int nw = working.count();

    // Run main loop ... 
    Matrix<T, Dynamic, 1> solution(D + N), bk, gk, lk, sk;
    Matrix<T, Dynamic, Dynamic> Ak; 
    for (int k = 0; k < max_iter; ++k)
    {
        // Define and solve the k-th equality-constrained convex QP subproblem
        // (Nocedal & Wright, Eq. 16.39)
        Ak = Matrix<T, Dynamic, Dynamic>::Zero(nw, D);
        int i = 0;  
        for (int j = 0; j < N; ++j)
        {
            if (working(j))
            {
                Ak.row(i) = A.row(j);
                i++; 
            }
        }
        bk = Matrix<T, Dynamic, 1>::Zero(nw);
        gk = G * xk + c;
        sk = solveEqualityConstrainedConvexQuadraticProgram<T>(G, gk, Ak, bk);

        // If the solution to the k-th subproblem is (close to) zero ...
        if (sk.squaredNorm() <= tol * tol)    // To allow for rational types
        {
            // Compute corresponding Lagrange multipliers (Nocedal & Wright,
            // Eq. 16.42)
            lk = Ak.transpose().fullPivLu().solve(gk);

            // If all Lagrange multipliers are non-negative among all working
            // active constraints, then the current iterate is in fact a
            // solution to the main QP 
            if ((lk.array() >= 0).all())
            {
                solution.head(D) = xk;
                solution.tail(N) = Matrix<T, Dynamic, 1>::Zero(N);
                i = 0; 
                for (int j = 0; j < N; ++j)
                {
                    if (working(j))
                    {
                        solution(D + j) = lk(i); 
                        i++;
                    }
                }
                return std::make_pair(solution, true);
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
                nw--;
            }
        }
        else    // If the solution to the k-th subproblem is nonzero ... 
        {
            // Compute the stepsize \alpha_k from Nocedal & Wright, Eq. 16.41,
            // keeping track of the blocking constraints 
            T alpha = 1;
            std::vector<int> blocking; 
            for (int j = 0; j < N; ++j)
            {
                // A blocking constraint is a non-working constraint with
                // A.row(j).dot(sk) < 0
                if (!working(j))
                {
                    T q = A.row(j).dot(sk);
                    if (q < 0)
                    { 
                        T bound = (b(j) - A.row(j).dot(xk)) / q;
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
            T stepsize;
            if (alpha < 1 || (alpha == 1 && blocking.size() > 0))
            {
                stepsize = alpha;
            }
            else    // alpha == 1 and blocking is empty 
            {
                stepsize = 1;
            }

            // Update the current iterate 
            xk += stepsize * sk;

            // If there are any blocking constraints, choose one to add to the
            // working active set 
            if (blocking.size() > 0)
                working(blocking[0]) = true;
            nw++;
        }
    }

    solution.head(D) = xk;
    solution.tail(N) = Matrix<T, Dynamic, 1>::Zero(N);
    i = 0; 
    for (int j = 0; j < N; ++j)
    {
        if (working(j))
        {
            solution(D + j) = lk(i); 
            i++;
        }
    }
    return std::make_pair(solution, false);
}

#endif
