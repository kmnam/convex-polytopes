#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../../include/vertexEnum.hpp"

/**
 * Test module for `HyperplaneArrangement` based on the 2-D arrangement
 * of five lines from Avis & Fukuda (1992). 
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
 * Parse the .hpa file containing the hyperplanes, check its contents, and 
 * return the parsed system. 
 */
TEST_CASE("parse .hpa file")
{
    Polytopes::HyperplaneArrangement* arr = new Polytopes::HyperplaneArrangement(); 
    arr->parse("example1.hpa"); 

    // Check the linear constraints themselves 
    MatrixXr A = arr->getA();
    REQUIRE(A.rows() == 5); 
    REQUIRE(A.cols() == 2);
    REQUIRE(A(0, 0) == 1); 
    REQUIRE(A(0, 1) == 3); 
    REQUIRE(A(1, 0) == 5); 
    REQUIRE(A(1, 1) == 1); 
    REQUIRE(A(2, 0) == 3); 
    REQUIRE(A(2, 1) == 2); 
    REQUIRE(A(3, 0) == -1); 
    REQUIRE(A(3, 1) == -3); 
    REQUIRE(A(4, 0) == -2); 
    REQUIRE(A(4, 1) == 1);  
    VectorXr b = arr->getb();
    REQUIRE(b.size() == 5); 
    REQUIRE(b(0) == 4); 
    REQUIRE(b(1) == 5); 
    REQUIRE(b(2) == 2); 
    REQUIRE(b(3) == 1); 
    REQUIRE(b(4) == 2);

    // Check the initial basis and cobasis 
    VectorXi basis = arr->getBasis();
    REQUIRE(basis.size() == 4);  
    REQUIRE(basis(0) == 0); 
    REQUIRE(basis(1) == 1); 
    REQUIRE(basis(2) == 2); 
    REQUIRE(basis(3) == 5); 
    VectorXi cobasis = arr->getCobasis();
    REQUIRE(cobasis.size() == 3); 
    REQUIRE(cobasis(0) == 3); 
    REQUIRE(cobasis(1) == 4); 
    REQUIRE(cobasis(2) == 6);

    // Check the entries in the core matrix
    MatrixXr core_A = arr->getCoreMatrix();
    REQUIRE(core_A.rows() == 4); 
    REQUIRE(core_A.cols() == 7); 
    REQUIRE(core_A(Eigen::all, basis) == MatrixXr::Identity(4, 4)); 
    REQUIRE(core_A(0, 3) == 1); 
    REQUIRE(core_A(0, 4) == 0); 
    REQUIRE(core_A(0, 6) == -5); 
    REQUIRE(core_A(1, 3) == 1); 
    REQUIRE(core_A(1, 4) == 2); 
    REQUIRE(core_A(1, 6) == -10); 
    REQUIRE(core_A(2, 3) == 1); 
    REQUIRE(core_A(2, 4) == 1); 
    REQUIRE(core_A(2, 6) == -5); 
    REQUIRE(core_A(3, 3) == 1); 
    REQUIRE(core_A(3, 4) == 1); 
    REQUIRE(core_A(3, 6) == 0);

    // Check the entries in the initial dictionary matrix 
    MatrixXr dict_coefs = arr->getDictCoefs(); 
    REQUIRE(dict_coefs.rows() == 4); 
    REQUIRE(dict_coefs.cols() == 3); 
    REQUIRE(dict_coefs == -core_A(Eigen::all, cobasis)); 

    delete arr; 
}

/**
 * Test that the first few pivots in the sequence in Avis & Fukuda (1992) are
 * performed correctly. 
 */
TEST_CASE("test first few pivots in the sequence in Avis & Fukuda (1992)")
{
    Polytopes::HyperplaneArrangement* arr = new Polytopes::HyperplaneArrangement(); 
    arr->parse("example1.hpa"); 

    // Check that 0 (basis(0)) is primal feasible prior to pivoting
    REQUIRE(arr->isPrimalFeasible(0));

    // Check that 3 (cobasis(0)) is dual feasible prior to pivoting
    REQUIRE(arr->isDualFeasible(0));  

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 0 (basis(0)) and 3 (cobasis(0))
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);
    REQUIRE(new_i == 2); 
    REQUIRE(new_j == 0);

    // Check that the new basis and cobasis are correct
    VectorXi basis = arr->getBasis();       // Should be 1, 2, 3, 5
    REQUIRE(basis(0) == 1);
    REQUIRE(basis(1) == 2); 
    REQUIRE(basis(2) == 3); 
    REQUIRE(basis(3) == 5);  
    VectorXi cobasis = arr->getCobasis();   // Should be 0, 4, 6
    REQUIRE(cobasis(0) == 0);
    REQUIRE(cobasis(1) == 4); 
    REQUIRE(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    MatrixXr dict_coefs = arr->getDictCoefs(); 
    REQUIRE(dict_coefs(0, 0) == 1); 
    REQUIRE(dict_coefs(0, 1) == -2); 
    REQUIRE(dict_coefs(0, 2) == 5); 
    REQUIRE(dict_coefs(1, 0) == 1); 
    REQUIRE(dict_coefs(1, 1) == -1); 
    REQUIRE(dict_coefs(1, 2) == 0); 
    REQUIRE(dict_coefs(2, 0) == -1); 
    REQUIRE(dict_coefs(2, 1) == 0); 
    REQUIRE(dict_coefs(2, 2) == 5); 
    REQUIRE(dict_coefs(3, 0) == 1); 
    REQUIRE(dict_coefs(3, 1) == -1); 
    REQUIRE(dict_coefs(3, 2) == -5);

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(0)) AND 0 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 1 (basis(0)) and 0 (cobasis(0))
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);
    REQUIRE(new_i == 0); 
    REQUIRE(new_j == 0);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 0, 2, 3, 5
    REQUIRE(basis(0) == 0);
    REQUIRE(basis(1) == 2); 
    REQUIRE(basis(2) == 3); 
    REQUIRE(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 1, 4, 6
    REQUIRE(cobasis(0) == 1);
    REQUIRE(cobasis(1) == 4); 
    REQUIRE(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs(); 
    REQUIRE(dict_coefs(0, 0) == 1); 
    REQUIRE(dict_coefs(0, 1) == 2); 
    REQUIRE(dict_coefs(0, 2) == -5); 
    REQUIRE(dict_coefs(1, 0) == 1); 
    REQUIRE(dict_coefs(1, 1) == 1); 
    REQUIRE(dict_coefs(1, 2) == -5); 
    REQUIRE(dict_coefs(2, 0) == -1); 
    REQUIRE(dict_coefs(2, 1) == -2); 
    REQUIRE(dict_coefs(2, 2) == 10); 
    REQUIRE(dict_coefs(3, 0) == 1); 
    REQUIRE(dict_coefs(3, 1) == 1); 
    REQUIRE(dict_coefs(3, 2) == -10);

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 1 (cobasis(0))            //
    //              (TO RE-OBTAIN THE PREVIOUS DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(0, 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(1)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Try pivoting 2 (basis(1)) and 4 (cobasis(1))
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);
    REQUIRE(new_i == 2); 
    REQUIRE(new_j == 1);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 1, 3, 4, 5
    REQUIRE(basis(0) == 1);
    REQUIRE(basis(1) == 3); 
    REQUIRE(basis(2) == 4); 
    REQUIRE(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 0, 2, 6
    REQUIRE(cobasis(0) == 0);
    REQUIRE(cobasis(1) == 2); 
    REQUIRE(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs(); 
    REQUIRE(dict_coefs(0, 0) == -1); 
    REQUIRE(dict_coefs(0, 1) == 2); 
    REQUIRE(dict_coefs(0, 2) == 5); 
    REQUIRE(dict_coefs(1, 0) == -1); 
    REQUIRE(dict_coefs(1, 1) == 0); 
    REQUIRE(dict_coefs(1, 2) == 5); 
    REQUIRE(dict_coefs(2, 0) == 1); 
    REQUIRE(dict_coefs(2, 1) == -1); 
    REQUIRE(dict_coefs(2, 2) == 0); 
    REQUIRE(dict_coefs(3, 0) == 0); 
    REQUIRE(dict_coefs(3, 1) == 1); 
    REQUIRE(dict_coefs(3, 2) == -5);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 2 (cobasis(1))            //
    //               THEN 3 (basis(2)) AND 0 (cobasis(0))              //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 1);
    arr->pivot(2, 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    // Try pivoting 1 (basis(1)) and 4 (cobasis(1))
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);
    REQUIRE(new_i == 2); 
    REQUIRE(new_j == 0);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 0, 2, 4, 5
    REQUIRE(basis(0) == 0);
    REQUIRE(basis(1) == 2); 
    REQUIRE(basis(2) == 4); 
    REQUIRE(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 1, 3, 6
    REQUIRE(cobasis(0) == 1);
    REQUIRE(cobasis(1) == 3); 
    REQUIRE(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs();
    mpq_rational half = static_cast<mpq_rational>(1) / static_cast<mpq_rational>(2); 
    REQUIRE(dict_coefs(0, 0) == 0); 
    REQUIRE(dict_coefs(0, 1) == -1); 
    REQUIRE(dict_coefs(0, 2) == 5); 
    REQUIRE(dict_coefs(1, 0) == half); 
    REQUIRE(dict_coefs(1, 1) == -half); 
    REQUIRE(dict_coefs(1, 2) == 0); 
    REQUIRE(dict_coefs(2, 0) == -half); 
    REQUIRE(dict_coefs(2, 1) == -half); 
    REQUIRE(dict_coefs(2, 2) == 5); 
    REQUIRE(dict_coefs(3, 0) == half); 
    REQUIRE(dict_coefs(3, 1) == -half); 
    REQUIRE(dict_coefs(3, 2) == -5);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 1 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);

    delete arr;
}

/**
 * Test that the full sequence of pivots in Avis & Fukuda (1992) are performed
 * correctly:
 *
 * 1)  0 <-> 3 *
 * 2)  1 <-> 0 *
 * 3)  0 <-> 1
 * 4)  2 <-> 4 *
 * 5)  4 <-> 2
 * 6)  3 <-> 0
 * 7)  1 <-> 4 *
 * 8)  0 <-> 3 *
 * 9)  3 <-> 0
 * 10) 2 <-> 3 *
 * 11) 3 <-> 2
 * 12) 4 <-> 1
 * 13) 2 <-> 3 *
 * 14) 3 <-> 2
 * 15) 2 <-> 4 *
 * 16) 4 <-> 2 
 *
 * and that the reverse criss-cross pivots (starred) among them are identified
 * correctly.
 */
TEST_CASE("test full pivot sequence in Avis & Fukuda (1992)")
{
    Polytopes::HyperplaneArrangement* arr = new Polytopes::HyperplaneArrangement(); 
    arr->parse("example1.hpa"); 

    int new_i, new_j, ind, cc_i, cc_j;

    // Obtain all possible reverse criss-cross pivots
    MatrixXi reverse_cc = arr->getReverseCrissCrossPivots(); 

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(0))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(0, 0));      // Check that (0, 3) is a r.c.c. pivot
    REQUIRE(reverse_cc(0, 0) == 0);                    // Check that (0, 3) is the first possible r.c.c. pivot
    REQUIRE(reverse_cc(0, 1) == 0); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (3, 0) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i);
    REQUIRE(cc_j == new_j);

    // Obtain all possible reverse criss-cross pivots from the new dictionary
    MatrixXi reverse_cc2 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc2.rows() == 2);  

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(0)) AND 0 (cobasis(0))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(0, 0));      // Check that (1, 0) is a r.c.c. pivot
    REQUIRE(reverse_cc2(0, 0) == 0);                   // Check that (1, 0) is the first possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc2(0, 1) == 0); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (0, 1) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc3 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc3.rows() == 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 1 (cobasis(0))            //
    //              (TO RE-OBTAIN THE PREVIOUS DICTIONARY)             //
    // --------------------------------------------------------------- //
    REQUIRE(!arr->isReverseCrissCrossPivot(0, 0));     // Check that (0, 1) is *not* a r.c.c. pivot
    arr->pivot(0, 0);                                 // Perform the pivot

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(1)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(1, 1));      // Check that (2, 4) is a r.c.c. pivot
    REQUIRE(reverse_cc2(1, 0) == 1);                   // Check that (2, 4) is the next possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc2(1, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 4) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc4 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc4.rows() == 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 2 (cobasis(1))            //
    //               THEN 3 (basis(2)) AND 0 (cobasis(0))              //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 1);
    arr->pivot(2, 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(1)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(1, 1));      // Check that (1, 4) is a r.c.c. pivot
    REQUIRE(reverse_cc(1, 0) == 1);                    // Check that (1, 4) is the next possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc(1, 1) == 1);
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (1, 4) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Obtain all possible reverse criss-cross pivots from the new dictionary
    MatrixXi reverse_cc5 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc5.rows() == 2);  

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(1))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(0, 1));      // Check that (0, 3) is a r.c.c. pivot
    REQUIRE(reverse_cc5(0, 0) == 0);                   // Check that (0, 3) is the first possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc5(0, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (0, 3) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc6 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc6.rows() == 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(1)) AND 0 (cobasis(0))            //
    //              (TO RE-OBTAIN THE PREVIOUS DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(1, 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(1)) AND 3 (cobasis(1))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(1, 1));      // Check that (2, 3) is a r.c.c. pivot
    REQUIRE(reverse_cc5(1, 0) == 1);                   // Check that (2, 3) is the first possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc5(1, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 3) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc7 = arr->getReverseCrissCrossPivots();
    REQUIRE(reverse_cc7.rows() == 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(1)) AND 2 (cobasis(1))            //
    //               THEN 4 (basis(2)) AND 1 (cobasis(0))              //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(1, 1);
    arr->pivot(2, 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 3 (cobasis(0))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(2, 0));      // Check that (2, 3) is a r.c.c. pivot
    REQUIRE(reverse_cc(2, 0) == 2);                    // Check that (2, 3) is the third possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc(2, 1) == 0);
    std::tie(new_i, new_j, ind) = arr->pivot(2, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 3) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc8 = arr->getReverseCrissCrossPivots(); 
    REQUIRE(reverse_cc8.rows() == 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(2)) AND 2 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    REQUIRE(arr->isReverseCrissCrossPivot(2, 1));      // Check that (2, 4) is a r.c.c. pivot
    REQUIRE(reverse_cc(3, 0) == 2);                    // Check that (2, 4) is the fourth possible r.c.c. pivot from this dictionary
    REQUIRE(reverse_cc(3, 1) == 1);
    std::tie(new_i, new_j, ind) = arr->pivot(2, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 4) is the c.c. pivot from new dictionary
    REQUIRE(cc_i == new_i); 
    REQUIRE(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc9 = arr->getReverseCrissCrossPivots(); 
    REQUIRE(reverse_cc9.rows() == 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 2 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);

    delete arr;
}

/**
 * Test that `searchFromOptimalDictCrissCross()` visits each optimal dictionary
 * visited by Avis & Fukuda (1992) via the following sequence of pivots (i.e.,
 * the dictionary formed after each starred pivot):
 *
 *               (Initial basis: 0, 1, 2, 5)
 * 1)  0 <-> 3 * (1, 2, 3, 5)
 * 2)  1 <-> 0 * (0, 2, 3, 5)
 * 3)  0 <-> 1
 * 4)  2 <-> 4 * (1, 3, 4, 5)
 * 5)  4 <-> 2
 * 6)  3 <-> 0
 * 7)  1 <-> 4 * (0, 2, 4, 5)
 * 8)  0 <-> 3 * (2, 3, 4, 5)
 * 9)  3 <-> 0
 * 10) 2 <-> 3 * (0, 3, 4, 5)
 * 11) 3 <-> 2
 * 12) 4 <-> 1
 * 13) 2 <-> 3 * (0, 1, 3, 5)
 * 14) 3 <-> 2
 * 15) 2 <-> 4 * (0, 1, 4, 5)
 * 16) 4 <-> 2 
 *
 * in the correct order.
 */
TEST_CASE("test method for optimal dictionary enumeration via criss-cross pivots")
{
    Polytopes::HyperplaneArrangement* arr = new Polytopes::HyperplaneArrangement(); 
    arr->parse("example1.hpa"); 

    auto result = arr->searchFromOptimalDictCrissCross();
    MatrixXi bases = result.first;
    MatrixXi bases_correct(9, 4); 
    bases_correct << 0, 1, 2, 5,
                     1, 2, 3, 5,
                     0, 2, 3, 5,
                     1, 3, 4, 5,
                     0, 2, 4, 5,
                     2, 3, 4, 5,
                     0, 3, 4, 5, 
                     0, 1, 3, 5,
                     0, 1, 4, 5;
    REQUIRE(bases == bases_correct);

    delete arr; 
}

/**
 * Test that `enumVertices()` correctly enumerates the five vertices of the 
 * arrangement, in the same fashion as in `searchFromOptimalDictCrissCross()`.
 */
/*
void TEST_MODULE_ENUM_VERTICES(Polytopes::HyperplaneArrangement* arr)
{
    MatrixXr vertices = arr->enumVertices();
    std::cout << vertices << std::endl;  
}
*/
