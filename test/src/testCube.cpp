#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../../include/vertexEnum.hpp"

/**
 * Test module for `PolyhedralDictionarySystem` based on the unit cube in
 * three dimensions. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     11/13/2022
 */
// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Parse the .poly file containing the unit cube constraints, check its contents,
 * and return the parsed system.  
 */
TEST_CASE("parse .poly file")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    // Check the linear constraints themselves 
    MatrixXr A = poly->getA();
    REQUIRE(A.rows() == 6); 
    REQUIRE(A.cols() == 3); 
    REQUIRE(A(Eigen::seqN(0, 3), Eigen::all) == MatrixXr::Identity(3, 3)); 
    REQUIRE(A(3, 0) == -1); 
    REQUIRE(A(3, 1) == 0); 
    REQUIRE(A(3, 2) == 0); 
    REQUIRE(A(4, 0) == 0); 
    REQUIRE(A(4, 1) == 0); 
    REQUIRE(A(4, 2) == -1); 
    REQUIRE(A(5, 0) == 0); 
    REQUIRE(A(5, 1) == -1); 
    REQUIRE(A(5, 2) == 0);
    VectorXr b = poly->getb();
    REQUIRE(b.size() == 6);  
    REQUIRE(b(0) == 1); 
    REQUIRE(b(1) == 1); 
    REQUIRE(b(2) == 1); 
    REQUIRE(b(3) == 0); 
    REQUIRE(b(4) == 0); 
    REQUIRE(b(5) == 0);

    // Check the initial basis and cobasis 
    VectorXi basis = poly->getBasis();
    REQUIRE(basis.size() == 7);  
    for (int i = 0; i < 7; ++i)
        REQUIRE(basis(i) == i); 
    VectorXi cobasis = poly->getCobasis();
    REQUIRE(cobasis.size() == 4); 
    for (int i = 0; i < 4; ++i)
        REQUIRE(cobasis(i) == 7 + i);  

    // Check the entries in the core matrix
    MatrixXr core_A = poly->getCoreMatrix();
    REQUIRE(core_A.rows() == 7); 
    REQUIRE(core_A.cols() == 11); 
    REQUIRE(core_A(Eigen::all, Eigen::seqN(0, 7)) == MatrixXr::Identity(7, 7));
    REQUIRE(core_A(0, 7) == 1); 
    REQUIRE(core_A(0, 8) == 1); 
    REQUIRE(core_A(0, 9) == 1); 
    REQUIRE(core_A(0, 10) == 0); 
    REQUIRE(core_A(1, 7) == 1); 
    REQUIRE(core_A(1, 8) == 0); 
    REQUIRE(core_A(1, 9) == 0); 
    REQUIRE(core_A(1, 10) == -1); 
    REQUIRE(core_A(2, 7) == 0); 
    REQUIRE(core_A(2, 8) == 1); 
    REQUIRE(core_A(2, 9) == 0); 
    REQUIRE(core_A(2, 10) == -1); 
    REQUIRE(core_A(3, 7) == 0); 
    REQUIRE(core_A(3, 8) == 0); 
    REQUIRE(core_A(3, 9) == 1); 
    REQUIRE(core_A(3, 10) == -1); 
    REQUIRE(core_A(4, 7) == -1); 
    REQUIRE(core_A(4, 8) == 0); 
    REQUIRE(core_A(4, 9) == 0); 
    REQUIRE(core_A(4, 10) == 0); 
    REQUIRE(core_A(5, 7) == 0); 
    REQUIRE(core_A(5, 8) == 0); 
    REQUIRE(core_A(5, 9) == -1); 
    REQUIRE(core_A(5, 10) == 0); 
    REQUIRE(core_A(6, 7) == 0); 
    REQUIRE(core_A(6, 8) == -1); 
    REQUIRE(core_A(6, 9) == 0); 
    REQUIRE(core_A(6, 10) == 0);

    // Check the entries in the initial dictionary matrix 
    MatrixXr dict_coefs = poly->getDictCoefs();
    REQUIRE(dict_coefs.rows() == 7); 
    REQUIRE(dict_coefs.cols() == 4); 
    REQUIRE(dict_coefs == -core_A(Eigen::all, Eigen::seqN(7, 4)));

    delete poly;  
}

/**
 * Test that three different pivots for the unit cube system (two valid and 
 * one invalid), starting from the initial optimal dictionary, are performed
 * correctly. 
 */
TEST_CASE("attempt three pivots, two valid and one invalid")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    // Check that 1 (basis(1)) is primal feasible prior to pivoting
    REQUIRE(poly->isPrimalFeasible(1));

    // Check that 7 (cobasis(0)) is dual feasible prior to pivoting
    REQUIRE(poly->isDualFeasible(0));  

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 1 (basis(1)) and 7 (cobasis(0))
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);
    REQUIRE(new_i == 6); 
    REQUIRE(new_j == 0); 

    // Check that the new basis and cobasis are correct
    VectorXi basis = poly->getBasis();       // Should be 0, 2, 3, 4, 5, 6, 7
    REQUIRE(basis(0) == 0); 
    for (int i = 1; i < 7; ++i)
        REQUIRE(basis(i) == i + 1); 
    VectorXi cobasis = poly->getCobasis();   // Should be 1, 8, 9, 10
    REQUIRE(cobasis(0) == 1); 
    for (int i = 1; i < 4; ++i)
        REQUIRE(cobasis(i) == 7 + i);

    // Check that the new dictionary coefficient matrix is correct
    MatrixXr dict_coefs = poly->getDictCoefs();
    REQUIRE(dict_coefs(0, 0) == 1); 
    REQUIRE(dict_coefs(0, 1) == -1); 
    REQUIRE(dict_coefs(0, 2) == -1); 
    REQUIRE(dict_coefs(0, 3) == -1);
    REQUIRE(dict_coefs(1, 0) == 0); 
    REQUIRE(dict_coefs(1, 1) == -1); 
    REQUIRE(dict_coefs(1, 2) == 0); 
    REQUIRE(dict_coefs(1, 3) == 1); 
    REQUIRE(dict_coefs(2, 0) == 0); 
    REQUIRE(dict_coefs(2, 1) == 0); 
    REQUIRE(dict_coefs(2, 2) == -1); 
    REQUIRE(dict_coefs(2, 3) == 1); 
    REQUIRE(dict_coefs(3, 0) == -1); 
    REQUIRE(dict_coefs(3, 1) == 0); 
    REQUIRE(dict_coefs(3, 2) == 0); 
    REQUIRE(dict_coefs(3, 3) == 1); 
    REQUIRE(dict_coefs(4, 0) == 0); 
    REQUIRE(dict_coefs(4, 1) == 0); 
    REQUIRE(dict_coefs(4, 2) == 1); 
    REQUIRE(dict_coefs(4, 3) == 0); 
    REQUIRE(dict_coefs(5, 0) == 0); 
    REQUIRE(dict_coefs(5, 1) == 1); 
    REQUIRE(dict_coefs(5, 2) == 0); 
    REQUIRE(dict_coefs(5, 3) == 0); 
    REQUIRE(dict_coefs(6, 0) == -1); 
    REQUIRE(dict_coefs(6, 1) == 0); 
    REQUIRE(dict_coefs(6, 2) == 0); 
    REQUIRE(dict_coefs(6, 3) == 1);

    // Check that 7 (basis(6)) is primal feasible
    REQUIRE(poly->isPrimalFeasible(6));

    // Check that 1 (cobasis(0)) is dual infeasible 
    REQUIRE(!poly->isDualFeasible(0));

    // Check that the indicator passed by pivot() indicates that the pivot was 
    // primal feasible but dual infeasible
    REQUIRE(ind == 1);

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 1); 
    REQUIRE(new_j == 0); 
    REQUIRE(ind == 1);

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
    REQUIRE(exception_caught); 

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now attempt pivoting 3 (basis(3)) and 9 (cobasis(2))
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);
    REQUIRE(new_i == 6); 
    REQUIRE(new_j == 0); 

    // Check that the new basis and cobasis are correct
    basis = poly->getBasis();       // Should be 0, 1, 2, 4, 5, 6, 9
    REQUIRE(basis(0) == 0); 
    REQUIRE(basis(1) == 1); 
    REQUIRE(basis(2) == 2); 
    REQUIRE(basis(3) == 4); 
    REQUIRE(basis(4) == 5); 
    REQUIRE(basis(5) == 6);
    REQUIRE(basis(6) == 9);  
    cobasis = poly->getCobasis();   // Should be 3, 7, 8, 10
    REQUIRE(cobasis(0) == 3);
    REQUIRE(cobasis(1) == 7); 
    REQUIRE(cobasis(2) == 8); 
    REQUIRE(cobasis(3) == 10);

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = poly->getDictCoefs();
    REQUIRE(dict_coefs(0, 0) == 1); 
    REQUIRE(dict_coefs(0, 1) == -1); 
    REQUIRE(dict_coefs(0, 2) == -1); 
    REQUIRE(dict_coefs(0, 3) == -1);
    REQUIRE(dict_coefs(1, 0) == 0); 
    REQUIRE(dict_coefs(1, 1) == -1); 
    REQUIRE(dict_coefs(1, 2) == 0); 
    REQUIRE(dict_coefs(1, 3) == 1); 
    REQUIRE(dict_coefs(2, 0) == 0); 
    REQUIRE(dict_coefs(2, 1) == 0); 
    REQUIRE(dict_coefs(2, 2) == -1); 
    REQUIRE(dict_coefs(2, 3) == 1); 
    REQUIRE(dict_coefs(3, 0) == 0); 
    REQUIRE(dict_coefs(3, 1) == 1); 
    REQUIRE(dict_coefs(3, 2) == 0); 
    REQUIRE(dict_coefs(3, 3) == 0); 
    REQUIRE(dict_coefs(4, 0) == -1); 
    REQUIRE(dict_coefs(4, 1) == 0); 
    REQUIRE(dict_coefs(4, 2) == 0); 
    REQUIRE(dict_coefs(4, 3) == 1); 
    REQUIRE(dict_coefs(5, 0) == 0); 
    REQUIRE(dict_coefs(5, 1) == 0); 
    REQUIRE(dict_coefs(5, 2) == 1); 
    REQUIRE(dict_coefs(5, 3) == 0); 
    REQUIRE(dict_coefs(6, 0) == -1); 
    REQUIRE(dict_coefs(6, 1) == 0); 
    REQUIRE(dict_coefs(6, 2) == 0); 
    REQUIRE(dict_coefs(6, 3) == 1);

    // Check that 9 (basis(6)) is primal feasible
    REQUIRE(poly->isPrimalFeasible(6));

    // Check that 3 (cobasis(0)) is dual infeasible 
    REQUIRE(!poly->isDualFeasible(0));

    // Check that the indicator passed by pivot() indicates that the pivot was 
    // primal feasible but dual infeasible
    REQUIRE(ind == 1);

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 3); 
    REQUIRE(new_j == 2); 
    REQUIRE(ind == 1);

    delete poly;
}

/**
 * Test that six different candidate reverse Bland pivots for the unit cube
 * system (**all** incorrect), starting from the initial optimal dictionary,
 * are performed correctly.
 */
TEST_CASE("attempt six candidate reverse Bland pivots, all invalid")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Pivot 1 (basis(1)) and 7 (cobasis(0)): *not* a reverse Bland pivot
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);

    // Then determine the Bland pivot of the new dictionary 
    int bland_i, bland_j; 
    std::tie(bland_i, bland_j) = poly->findBland();
    REQUIRE(bland_i == 3); 
    REQUIRE(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 1); 
    REQUIRE(new_j == 0); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(1, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 3 (basis(3)) and 9 (cobasis(2)): *not* a reverse Bland pivot
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);

    // Then determine the Bland pivot of the new dictionary 
    std::tie(bland_i, bland_j) = poly->findBland();
    REQUIRE(bland_i == 4); 
    REQUIRE(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 3); 
    REQUIRE(new_j == 2); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(3, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 2 (basis(2)) and 8 (cobasis(1)): *not* a reverse Bland pivot
    std::tie(new_i, new_j, ind) = poly->pivot(2, 1);

    // Then determine the Bland pivot of the new dictionary 
    std::tie(bland_i, bland_j) = poly->findBland();
    REQUIRE(bland_i == 5); 
    REQUIRE(bland_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 2); 
    REQUIRE(new_j == 1); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(2, 1));

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(4)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Now pivot 4 (basis(4)) and 7 (cobasis(0)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(4, 0);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 4); 
    REQUIRE(new_j == 0); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(4, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 5 (basis(5)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 5 (basis(5)) and 9 (cobasis(2)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(5, 2);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught);
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 5); 
    REQUIRE(new_j == 2); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(5, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 6 (basis(6)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 6 (basis(6)) and 8 (cobasis(1)): *not* a reverse Bland pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(6, 1);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 6); 
    REQUIRE(new_j == 1);
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse Bland pivot
    REQUIRE(!poly->isReverseBlandPivot(6, 1));

    delete poly;
}

/**
 * Test that six different candidate reverse criss-cross pivots for the unit
 * cube system (**all** incorrect), starting from the initial optimal
 * dictionary, are performed correctly.
 */
TEST_CASE("attempt six candidate reverse criss-cross pivots, all invalid")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Pivot 1 (basis(1)) and 7 (cobasis(0)): *not* a reverse criss-cross pivot
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);

    // Then determine the criss-cross pivot of the new dictionary 
    int cc_i, cc_j; 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    REQUIRE(cc_i == 3); 
    REQUIRE(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 1); 
    REQUIRE(new_j == 0); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(1, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 3 (basis(3)) and 9 (cobasis(2)): *not* a reverse criss-cross pivot
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);

    // Then determine the criss-cross pivot of the new dictionary 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    REQUIRE(cc_i == 4); 
    REQUIRE(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 3); 
    REQUIRE(new_j == 2); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(3, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 2 (basis(2)) and 8 (cobasis(1)): *not* a reverse criss-cross pivot
    std::tie(new_i, new_j, ind) = poly->pivot(2, 1);

    // Then determine the criss-cross pivot of the new dictionary 
    std::tie(cc_i, cc_j) = poly->findCrissCross();
    REQUIRE(cc_i == 5); 
    REQUIRE(cc_j == 0);

    // Reverse the pivot ... 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 2); 
    REQUIRE(new_j == 1); 

    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(2, 1));

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(4)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Now pivot 4 (basis(4)) and 7 (cobasis(0)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(4, 0);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 4); 
    REQUIRE(new_j == 0); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(4, 0));

    // --------------------------------------------------------------- //
    //             PIVOTING 5 (basis(5)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    // Now pivot 5 (basis(5)) and 9 (cobasis(2)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(5, 2);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 5); 
    REQUIRE(new_j == 2); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(5, 2));

    // --------------------------------------------------------------- //
    //             PIVOTING 6 (basis(6)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Now pivot 6 (basis(6)) and 8 (cobasis(1)): *not* a reverse criss-cross pivot
    // but rather yields an optimal dictionary
    std::tie(new_i, new_j, ind) = poly->pivot(6, 1);
    for (int i = 0; i < 3; ++i)
        REQUIRE(poly->isDualFeasible(i));     // Check that cobasis(0), ..., cobasis(2) are dual feasible
    for (int i = 1; i <= 6; ++i)
        REQUIRE(poly->isPrimalFeasible(i));   // Check that basis(1), ..., basis(6) are primal feasible  
    REQUIRE(ind == 3);

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
    REQUIRE(exception_caught); 
    
    // Reverse the pivot ...
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    REQUIRE(new_i == 6); 
    REQUIRE(new_j == 1); 
    
    // ... and check that the original pivot (from the original optimal dictionary)
    // is *not* a reverse criss-cross pivot
    REQUIRE(!poly->isReverseCrissCrossPivot(6, 1));

    delete poly;
}

/**
 * Test that the initial optimal dictionary, along with six other primal
 * feasible dictionaries obtained through pivoting, all yield vertices of
 * the unit cube. 
 */
TEST_CASE("obtain seven vertices through pivots from initial optimal dictionary")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    VectorXr vertex = poly->getVertex();
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = poly->pivot(1, 0);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // Reverse the pivot 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(3)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    std::tie(new_i, new_j, ind) = poly->pivot(3, 2);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 
    
    // Reverse the pivot 
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    std::tie(new_i, new_j, ind) = poly->pivot(2, 1);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // Reverse the pivot  
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(4)) AND 7 (cobasis(0))            //
    // --------------------------------------------------------------- //
    std::tie(new_i, new_j, ind) = poly->pivot(4, 0);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);
    
    // --------------------------------------------------------------- //
    //             PIVOTING 5 (basis(5)) AND 9 (cobasis(2))            //
    // --------------------------------------------------------------- //
    std::tie(new_i, new_j, ind) = poly->pivot(5, 2);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);

    // --------------------------------------------------------------- //
    //             PIVOTING 6 (basis(6)) AND 8 (cobasis(1))            //
    // --------------------------------------------------------------- //
    std::tie(new_i, new_j, ind) = poly->pivot(6, 1);
    vertex = poly->getVertex(); 
    for (int k = 0; k < 3; ++k)
        REQUIRE((vertex(k) == 0 || vertex(k) == 1)); 

    // Reverse the pivot
    std::tie(new_i, new_j, ind) = poly->pivot(new_i, new_j);

    delete poly;
}

/**
 * Test that `enumVertices()` correctly enumerates the eight vertices of 
 * the unit cube. 
 */
TEST_CASE("obtain all eight vertices of the cube")
{
    Polytopes::PolyhedralDictionarySystem* poly = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::LessThanOrEqualTo
    );
    poly->parse("cube.poly");

    MatrixXr vertices = poly->enumVertices();
    MatrixXr correct_vertices(8, 3);
    correct_vertices << 0, 0, 0,
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                        1, 1, 0,
                        1, 0, 1,
                        0, 1, 1,
                        1, 1, 1;

    // Search for each of the eight correct vertices ... 
    std::vector<int> found;
    for (int i = 0; i < 8; ++i)
    {
        int found_index = -1; 
        for (int j = 0; j < 8; ++j)
        {
            if (correct_vertices.row(i) == vertices.row(j))
            {
                found_index = j;
                break;
            }
        }
        REQUIRE(found_index != -1);   // Was the i-th vertex found?
        found.push_back(found_index);
    }
    REQUIRE(found.size() == 8);       // There should have been eight vertices found
    for (int i = 0; i < 8; ++i)
        REQUIRE(std::find(found.begin(), found.end(), i) != found.end());

    delete poly; 
}

