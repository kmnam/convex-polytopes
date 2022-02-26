#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "../../include/polytopes.hpp"
#include "../../include/vertexEnum.hpp"

/**
 * A brief example that samples points from the unit cube in three dimensions.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/26/2022
 */
// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Parse the .poly file containing the unit cube constraints, check its contents,
 * and return the parsed system.  
 */
Polytopes::PolyhedralDictionarySystem* TEST_MODULE_PARSE()
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(Polytopes::InequalityType::LessThanOrEqualTo);
    poly->parse("cube.poly");

    // Check the linear constraints themselves 
    MatrixXr A = poly->getA();
    assert(A.rows() == 6); 
    assert(A.cols() == 3); 
    assert(A(Eigen::seqN(0, 3), Eigen::all) == MatrixXr::Identity(3, 3)); 
    assert(A(3, 0) == -1); 
    assert(A(3, 1) == 0); 
    assert(A(3, 2) == 0); 
    assert(A(4, 0) == 0); 
    assert(A(4, 1) == 0); 
    assert(A(4, 2) == -1); 
    assert(A(5, 0) == 0); 
    assert(A(5, 1) == -1); 
    assert(A(5, 2) == 0);
    VectorXr b = poly->getb();
    assert(b.size() == 6);  
    assert(b(0) == 1); 
    assert(b(1) == 1); 
    assert(b(2) == 1); 
    assert(b(3) == 0); 
    assert(b(4) == 0); 
    assert(b(5) == 0);

    // Check the initial basis and cobasis 
    VectorXi basis = poly->getBasis();
    assert(basis.size() == 7);  
    for (int i = 0; i < 7; ++i)
        assert(basis(i) == i); 
    VectorXi cobasis = poly->getCobasis();
    assert(basis.size() == 4); 
    for (int i = 0; i < 4; ++i)
        assert(cobasis(i) == 7 + i);  

    // Check the entries in the core matrix
    MatrixXr core_A = poly->getCoreMatrix();
    assert(core_A.rows() == 7); 
    assert(core_A.cols() == 11); 
    assert(core_A(Eigen::all, Eigen::seqN(0, 7)) == MatrixXr::Identity(7, 7));
    assert(core_A(0, 7) == 1); 
    assert(core_A(0, 8) == 1); 
    assert(core_A(0, 9) == 1); 
    assert(core_A(0, 10) == 0); 
    assert(core_A(1, 7) == 1); 
    assert(core_A(1, 8) == 0); 
    assert(core_A(1, 9) == 0); 
    assert(core_A(1, 10) == -1); 
    assert(core_A(2, 7) == 0); 
    assert(core_A(2, 8) == 1); 
    assert(core_A(2, 9) == 0); 
    assert(core_A(2, 10) == -1); 
    assert(core_A(3, 7) == 0); 
    assert(core_A(3, 8) == 0); 
    assert(core_A(3, 9) == 1); 
    assert(core_A(3, 10) == -1); 
    assert(core_A(4, 7) == -1); 
    assert(core_A(4, 8) == 0); 
    assert(core_A(4, 9) == 0); 
    assert(core_A(4, 10) == 0); 
    assert(core_A(5, 7) == 0); 
    assert(core_A(5, 8) == 0); 
    assert(core_A(5, 9) == -1); 
    assert(core_A(5, 10) == 0); 
    assert(core_A(6, 7) == 0); 
    assert(core_A(6, 8) == -1); 
    assert(core_A(6, 9) == 0); 
    assert(core_A(6, 10) == 0);

    // Check the entries in the initial dictionary matrix 
    MatrixXr dict_coefs = poly->getDictCoefs(); 
    assert(dict_coefs.rows() == 7); 
    assert(dict_coefs.rows() == 4); 
    assert(dict_coefs == -core_A(Eigen::all, Eigen::seqN(7, 4)));  

    return poly;  
}

/**
 * Test that three different pivots for the unit cube system (two valid and 
 * one invalid), starting from the initial optimal dictionary, are performed
 * correctly. 
 */
void TEST_MODULE_THREE_PIVOT_ATTEMPTS(Polytopes::PolyhedralDictionarySystem* poly)
{
    // Check that 1 (basis(1)) is primal feasible prior to pivoting
    assert(poly->isPrimalFeasible(1));

    // Check that 7 (cobasis(0)) is dual feasible prior to pivoting
    assert(poly->isDualFeasible(0));  

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 1 (basis(1)) and 7 (cobasis(0))
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);
    assert(new_i == 6); 
    assert(new_j == 0); 

    // Check that the new basis and cobasis are correct
    VectorXi basis = poly->getBasis();       // Should be 0, 2, 3, 4, 5, 6, 7
    assert(basis(0) == 0); 
    for (int i = 1; i < 7; ++i)
        assert(basis(i) == i + 1); 
    VectorXi cobasis = poly->getCobasis();   // Should be 1, 8, 9, 10
    assert(cobasis(0) == 1); 
    for (int i = 1; i < 4; ++i)
        assert(cobasis(i) == 7 + i);

    // Check that the new dictionary coefficient matrix is correct
    MatrixXr dict_coefs = poly->getDictCoefs();
    assert(dict_coefs(0, 0) == 1); 
    assert(dict_coefs(0, 1) == -1); 
    assert(dict_coefs(0, 2) == -1); 
    assert(dict_coefs(0, 3) == -1);
    assert(dict_coefs(1, 0) == 0); 
    assert(dict_coefs(1, 1) == -1); 
    assert(dict_coefs(1, 2) == 0); 
    assert(dict_coefs(1, 3) == 1); 
    assert(dict_coefs(2, 0) == 0); 
    assert(dict_coefs(2, 1) == 0); 
    assert(dict_coefs(2, 2) == -1); 
    assert(dict_coefs(2, 3) == 1); 
    assert(dict_coefs(3, 0) == -1); 
    assert(dict_coefs(3, 1) == 0); 
    assert(dict_coefs(3, 2) == 0); 
    assert(dict_coefs(3, 3) == 1); 
    assert(dict_coefs(4, 0) == 0); 
    assert(dict_coefs(4, 1) == 0); 
    assert(dict_coefs(4, 2) == 1); 
    assert(dict_coefs(4, 3) == 0); 
    assert(dict_coefs(5, 0) == 0); 
    assert(dict_coefs(5, 1) == 1); 
    assert(dict_coefs(5, 2) == 0); 
    assert(dict_coefs(5, 3) == 0); 
    assert(dict_coefs(6, 0) == -1); 
    assert(dict_coefs(6, 1) == 0); 
    assert(dict_coefs(6, 2) == 0); 
    assert(dict_coefs(6, 3) == 1);

    // Check that 7 (basis(6)) is primal feasible
    assert(poly->isPrimalFeasible(6));

    // Check that 1 (cobasis(0)) is dual infeasible 
    assert(!poly->isDualFeasible(0));

    // Check that the indicator passed by pivot() indicates that the pivot was 
    // primal feasible but dual infeasible
    assert(ind == 1);

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 1); 
    assert(new_j == 0); 
    assert(ind == 1);

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now attempt pivoting 1 (basis(1)) and 8 (cobasis(1)) -- this is an invalid
    // pivot and should raise an InvalidPivotException
    bool exception_caught = false; 
    try
    {
        poly->pivot(1, 1); 
    }
    catch (const Polytopes::InvalidPivotException& e)
    {
        exception_caught = true; 
    }
    assert(exception_caught); 

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now attempt pivoting 3 (basis(3)) and 9 (cobasis(2))
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);
    assert(new_i == 6); 
    assert(new_j == 0); 

    // Check that the new basis and cobasis are correct
    basis = poly->getBasis();       // Should be 0, 1, 2, 4, 5, 6, 9
    assert(basis(0) == 0); 
    assert(basis(1) == 1); 
    assert(basis(2) == 2); 
    assert(basis(3) == 4); 
    assert(basis(4) == 5); 
    assert(basis(5) == 6);
    assert(basis(6) == 9);  
    cobasis = poly->getCobasis();   // Should be 3, 8, 9, 10
    assert(cobasis(0) == 3);
    assert(cobasis(1) == 8); 
    assert(cobasis(2) == 9); 
    assert(cobasis(3) == 10);

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = poly->getDictCoefs();
    assert(dict_coefs(0, 0) == 1); 
    assert(dict_coefs(0, 1) == -1); 
    assert(dict_coefs(0, 2) == -1); 
    assert(dict_coefs(0, 3) == -1);
    assert(dict_coefs(1, 0) == 0); 
    assert(dict_coefs(1, 1) == -1); 
    assert(dict_coefs(1, 2) == 0); 
    assert(dict_coefs(1, 3) == 1); 
    assert(dict_coefs(2, 0) == 0); 
    assert(dict_coefs(2, 1) == 0); 
    assert(dict_coefs(2, 2) == -1); 
    assert(dict_coefs(2, 3) == 1); 
    assert(dict_coefs(3, 0) == 0); 
    assert(dict_coefs(3, 1) == 1); 
    assert(dict_coefs(3, 2) == 0); 
    assert(dict_coefs(3, 3) == 0); 
    assert(dict_coefs(4, 0) == -1); 
    assert(dict_coefs(4, 1) == 0); 
    assert(dict_coefs(4, 2) == 0); 
    assert(dict_coefs(4, 3) == 1); 
    assert(dict_coefs(5, 0) == 0); 
    assert(dict_coefs(5, 1) == 0); 
    assert(dict_coefs(5, 2) == 1); 
    assert(dict_coefs(5, 3) == 0); 
    assert(dict_coefs(6, 0) == -1); 
    assert(dict_coefs(6, 1) == 0); 
    assert(dict_coefs(6, 2) == 0); 
    assert(dict_coefs(6, 3) == 1);

    // Check that 9 (basis(6)) is primal feasible
    assert(poly->isPrimalFeasible(6));

    // Check that 3 (cobasis(0)) is dual infeasible 
    assert(!poly->isDualFeasible(0));

    // Check that the indicator passed by pivot() indicates that the pivot was 
    // primal feasible but dual infeasible
    assert(ind == 1);

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 1); 
    assert(new_j == 0); 
    assert(ind == 1);
}

/**
 * Test that six different candidate reverse Bland pivots for the unit cube
 * system (**all** incorrect), starting from the initial optimal dictionary,
 * are performed correctly.
 */
void TEST_MODULE_SIX_REVERSE_BLAND_PIVOT_CANDIDATES(Polytopes::PolyhedralDictionarySystem* poly)
{
    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Pivot 1 (basis(1)) and 7 (cobasis(0)): *not* a reverse Bland pivot
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);

    // Then determine the Bland pivot of the new dictionary 
    int bland_i, bland_j; 
    std::tie(bland_i, bland_j) = poly->findBland();
    assert(bland_i == 3); 
    assert(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 1); 
    assert(new_j == 0); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(1, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 3 (basis(3)) and 9 (cobasis(2)): *not* a reverse Bland pivot
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);

    // Then determine the Bland pivot of the new dictionary 
    std::tie(bland_i, bland_j) = poly->findBland();
    assert(bland_i == 4); 
    assert(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 3); 
    assert(new_j == 2); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(3, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 2 (basis(2)) and 8 (cobasis(1)): *not* a reverse Bland pivot
    std::tie(new_i, new_j, ind) = poly->pivot(2, 1);

    // Then determine the Bland pivot of the new dictionary 
    std::tie(bland_i, bland_j) = poly->findBland();
    assert(bland_i == 5); 
    assert(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 2); 
    assert(new_j == 1); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(2, 1));

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(4)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Now pivot 4 (basis(4)) and 7 (cobasis(0)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(4, 0);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the Bland pivot of the new dictionary -- this should 
    // throw a DualFeasibleException
    bool exception_caught = false; 
    try
    {
        std::tie(bland_i, bland_j) = poly->findBland(); 
    }
    catch (const Polytopes::DualFeasibleException& e)
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 4); 
    assert(new_j == 0); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(4, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 5 (basis(5)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 5 (basis(5)) and 9 (cobasis(2)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(5, 2);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the Bland pivot of the new dictionary -- this should 
    // throw a DualFeasibleException
    exception_caught = false; 
    try
    {
        std::tie(bland_i, bland_j) = poly->findBland(); 
    }
    catch (const Polytopes::DualFeasibleException& e)
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 5); 
    assert(new_j == 2); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(5, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 6 (basis(6)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 6 (basis(6)) and 8 (cobasis(1)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(6, 1);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the Bland pivot of the new dictionary -- this should 
    // throw a DualFeasibleException
    exception_caught = false; 
    try
    {
        std::tie(bland_i, bland_j) = poly->findBland(); 
    }
    catch (const Polytopes::DualFeasibleException& e)
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 6); 
    assert(new_j == 1); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    assert(!poly->isReverseBlandPivot(6, 1));
}

/**
 * Test that six different candidate reverse criss-cross pivots for the unit
 * cube system (**all** incorrect), starting from the initial optimal
 * dictionary, are performed correctly.
 */
void TEST_MODULE_SIX_REVERSE_CRISSCROSS_PIVOT_CANDIDATES(Polytopes::PolyhedralDictionarySystem* poly)
{
    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Pivot 1 (basis(1)) and 7 (cobasis(0)): *not* a reverse criss-cross pivot
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);

    // Then determine the criss-cross pivot of the new dictionary 
    int cc_i, cc_j; 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    assert(cc_i == 2); 
    assert(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 1); 
    assert(new_j == 0); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(1, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 3 (basis(3)) and 9 (cobasis(2)): *not* a reverse criss-cross pivot
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);

    // Then determine the criss-cross pivot of the new dictionary 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    assert(cc_i == 3); 
    assert(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 3); 
    assert(new_j == 2); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(3, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 2 (basis(2)) and 8 (cobasis(1)): *not* a reverse criss-cross pivot
    std::tie(new_i, new_j, ind) = poly->pivot(2, 1);

    // Then determine the criss-cross pivot of the new dictionary 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    assert(cc_i == 5); 
    assert(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 2); 
    assert(new_j == 1); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(2, 1));

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(4)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Now pivot 4 (basis(4)) and 7 (cobasis(0)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(4, 0);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the criss-cross pivot of the new dictionary -- this
    // should throw a OptimalDictionaryException
    bool exception_caught = false; 
    try
    {
        std::tie(cc_i, cc_j) = poly->findCrissCross(); 
    }
    catch (const Polytopes::OptimalDictionaryException& e)
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 4); 
    assert(new_j == 0); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(4, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 5 (basis(5)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 5 (basis(5)) and 9 (cobasis(2)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(5, 2);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the criss-cross pivot of the new dictionary -- this
    // should throw a OptimalDictionaryException 
    exception_caught = false; 
    try
    {
        std::tie(cc_i, cc_j) = poly->findCrissCross(); 
    }
    catch (const Polytopes::OptimalDictionaryException& e) 
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 5); 
    assert(new_j == 2); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(5, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 6 (basis(6)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 6 (basis(6)) and 8 (cobasis(1)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(6, 1);
    for (int i = 0; i < 3; ++i)
        assert(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        assert(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    assert(ind == 3);

    // Try to determine the criss-cross pivot of the new dictionary -- this
    // should throw a OptimalDictionaryException 
    exception_caught = false; 
    try
    {
        std::tie(cc_i, cc_j) = poly->findCrissCross(); 
    }
    catch (const Polytopes::OptimalDictionaryException& e) 
    {
        exception_caught = true; 
    } 
    assert(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    assert(new_i == 6); 
    assert(new_j == 1); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    assert(!poly->isReverseCrissCrossPivot(6, 1));
}

int main(int argc, char** argv)
{
    // Instantiate a polyhedral dictionary system from the unit cube constraints
    Polytopes::PolyhedralDictionarySystem* poly = TEST_MODULE_PARSE(); 
    std::cout << "TEST_MODULE_PARSE: all tests passed" << std::endl;

    // Test that three different pivots (two valid and one invalid) are performed
    // correctly 
    TEST_MODULE_THREE_PIVOT_ATTEMPTS(poly);
    std::cout << "TEST_MODULE_THREE_PIVOTS: all tests passed" << std::endl;

    // Test that six reverse Bland pivot candidates are categorized correctly 
    TEST_MODULE_SIX_REVERSE_BLAND_PIVOT_CANDIDATES(poly); 
    std::cout << "TEST_MODULE_SIX_REVERSE_BLAND_PIVOT_CANDIDATES: all tests passed" << std::endl; 

    // Test that six reverse criss-cross pivot candidates are categorized correctly 
    TEST_MODULE_SIX_REVERSE_CRISSCROSS_PIVOT_CANDIDATES(poly); 
    std::cout << "TEST_MODULE_SIX_REVERSE_CRISSCROSS_PIVOT_CANDIDATES: all tests passed" << std::endl; 
    
    delete poly; 
    return 0;
}
