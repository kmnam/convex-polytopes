#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "../../include/vertexEnum.hpp"

/**
 * A brief test suite for `HyperplaneArrangement` based on the 2-D arrangement
 * of five lines from Avis & Fukuda (1992). 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     3/3/2022
 */
// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Parse the .hpa file containing the hyperplanes, check its contents, and 
 * return the parsed system. 
 */
Polytopes::HyperplaneArrangement* TEST_MODULE_PARSE()
{
    Polytopes::HyperplaneArrangement* arr = new Polytopes::HyperplaneArrangement(); 
    arr->parse("example1.hpa"); 

    // Check the linear constraints themselves 
    MatrixXr A = arr->getA();
    assert(A.rows() == 5); 
    assert(A.cols() == 2);
    assert(A(0, 0) == 1); 
    assert(A(0, 1) == 3); 
    assert(A(1, 0) == 5); 
    assert(A(1, 1) == 1); 
    assert(A(2, 0) == 3); 
    assert(A(2, 1) == 2); 
    assert(A(3, 0) == -1); 
    assert(A(3, 1) == -3); 
    assert(A(4, 0) == -2); 
    assert(A(4, 1) == 1);  
    VectorXr b = arr->getb();
    assert(b.size() == 5); 
    assert(b(0) == 4); 
    assert(b(1) == 5); 
    assert(b(2) == 2); 
    assert(b(3) == 1); 
    assert(b(4) == 2); 

    // Check the initial basis and cobasis 
    VectorXi basis = arr->getBasis();
    assert(basis.size() == 4);  
    assert(basis(0) == 0); 
    assert(basis(1) == 1); 
    assert(basis(2) == 2); 
    assert(basis(3) == 5); 
    VectorXi cobasis = arr->getCobasis();
    assert(cobasis.size() == 3); 
    assert(cobasis(0) == 3); 
    assert(cobasis(1) == 4); 
    assert(cobasis(2) == 6);

    // Check the entries in the core matrix
    MatrixXr core_A = arr->getCoreMatrix();
    assert(core_A.rows() == 4); 
    assert(core_A.cols() == 7); 
    assert(core_A(Eigen::all, basis) == MatrixXr::Identity(4, 4)); 
    assert(core_A(0, 3) == 1); 
    assert(core_A(0, 4) == 0); 
    assert(core_A(0, 6) == -5); 
    assert(core_A(1, 3) == 1); 
    assert(core_A(1, 4) == 2); 
    assert(core_A(1, 6) == -10); 
    assert(core_A(2, 3) == 1); 
    assert(core_A(2, 4) == 1); 
    assert(core_A(2, 6) == -5); 
    assert(core_A(3, 3) == 1); 
    assert(core_A(3, 4) == 1); 
    assert(core_A(3, 6) == 0); 

    // Check the entries in the initial dictionary matrix 
    MatrixXr dict_coefs = arr->getDictCoefs(); 
    assert(dict_coefs.rows() == 4); 
    assert(dict_coefs.cols() == 3); 
    assert(dict_coefs == -core_A(Eigen::all, cobasis)); 

    return arr;  
}

/**
 * Test that the first few pivots in the sequence in Avis & Fukuda (1992) are
 * performed correctly. 
 */
void TEST_MODULE_PARTIAL_PIVOT_SEQUENCE(Polytopes::HyperplaneArrangement* arr)
{
    // Check that 0 (basis(0)) is primal feasible prior to pivoting
    assert(arr->isPrimalFeasible(0));

    // Check that 3 (cobasis(0)) is dual feasible prior to pivoting
    assert(arr->isDualFeasible(0));  

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 0 (basis(0)) and 3 (cobasis(0))
    int new_i, new_j, ind; 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);
    assert(new_i == 2); 
    assert(new_j == 0);

    // Check that the new basis and cobasis are correct
    VectorXi basis = arr->getBasis();       // Should be 1, 2, 3, 5
    assert(basis(0) == 1);
    assert(basis(1) == 2); 
    assert(basis(2) == 3); 
    assert(basis(3) == 5);  
    VectorXi cobasis = arr->getCobasis();   // Should be 0, 4, 6
    assert(cobasis(0) == 0);
    assert(cobasis(1) == 4); 
    assert(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    MatrixXr dict_coefs = arr->getDictCoefs(); 
    assert(dict_coefs(0, 0) == 1); 
    assert(dict_coefs(0, 1) == -2); 
    assert(dict_coefs(0, 2) == 5); 
    assert(dict_coefs(1, 0) == 1); 
    assert(dict_coefs(1, 1) == -1); 
    assert(dict_coefs(1, 2) == 0); 
    assert(dict_coefs(2, 0) == -1); 
    assert(dict_coefs(2, 1) == 0); 
    assert(dict_coefs(2, 2) == 5); 
    assert(dict_coefs(3, 0) == 1); 
    assert(dict_coefs(3, 1) == -1); 
    assert(dict_coefs(3, 2) == -5);

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(0)) AND 0 (cobasis(0))            //
    // --------------------------------------------------------------- //
    // Try pivoting 1 (basis(0)) and 0 (cobasis(0))
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);
    assert(new_i == 0); 
    assert(new_j == 0);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 0, 2, 3, 5
    assert(basis(0) == 0);
    assert(basis(1) == 2); 
    assert(basis(2) == 3); 
    assert(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 1, 4, 6
    assert(cobasis(0) == 1);
    assert(cobasis(1) == 4); 
    assert(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs(); 
    assert(dict_coefs(0, 0) == 1); 
    assert(dict_coefs(0, 1) == 2); 
    assert(dict_coefs(0, 2) == -5); 
    assert(dict_coefs(1, 0) == 1); 
    assert(dict_coefs(1, 1) == 1); 
    assert(dict_coefs(1, 2) == -5); 
    assert(dict_coefs(2, 0) == -1); 
    assert(dict_coefs(2, 1) == -2); 
    assert(dict_coefs(2, 2) == 10); 
    assert(dict_coefs(3, 0) == 1); 
    assert(dict_coefs(3, 1) == 1); 
    assert(dict_coefs(3, 2) == -10);

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
    assert(new_i == 2); 
    assert(new_j == 1);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 1, 3, 4, 5
    assert(basis(0) == 1);
    assert(basis(1) == 3); 
    assert(basis(2) == 4); 
    assert(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 0, 2, 6
    assert(cobasis(0) == 0);
    assert(cobasis(1) == 2); 
    assert(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs(); 
    assert(dict_coefs(0, 0) == -1); 
    assert(dict_coefs(0, 1) == 2); 
    assert(dict_coefs(0, 2) == 5); 
    assert(dict_coefs(1, 0) == -1); 
    assert(dict_coefs(1, 1) == 0); 
    assert(dict_coefs(1, 2) == 5); 
    assert(dict_coefs(2, 0) == 1); 
    assert(dict_coefs(2, 1) == -1); 
    assert(dict_coefs(2, 2) == 0); 
    assert(dict_coefs(3, 0) == 0); 
    assert(dict_coefs(3, 1) == 1); 
    assert(dict_coefs(3, 2) == -5);

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
    assert(new_i == 2); 
    assert(new_j == 0);

    // Check that the new basis and cobasis are correct
    basis = arr->getBasis();       // Should be 0, 2, 4, 5
    assert(basis(0) == 0);
    assert(basis(1) == 2); 
    assert(basis(2) == 4); 
    assert(basis(3) == 5);  
    cobasis = arr->getCobasis();   // Should be 1, 3, 6
    assert(cobasis(0) == 1);
    assert(cobasis(1) == 3); 
    assert(cobasis(2) == 6);  

    // Check that the new dictionary coefficient matrix is correct
    dict_coefs = arr->getDictCoefs();
    mpq_rational half = static_cast<mpq_rational>(1) / static_cast<mpq_rational>(2); 
    assert(dict_coefs(0, 0) == 0); 
    assert(dict_coefs(0, 1) == -1); 
    assert(dict_coefs(0, 2) == 5); 
    assert(dict_coefs(1, 0) == half); 
    assert(dict_coefs(1, 1) == -half); 
    assert(dict_coefs(1, 2) == 0); 
    assert(dict_coefs(2, 0) == -half); 
    assert(dict_coefs(2, 1) == -half); 
    assert(dict_coefs(2, 2) == 5); 
    assert(dict_coefs(3, 0) == half); 
    assert(dict_coefs(3, 1) == -half); 
    assert(dict_coefs(3, 2) == -5);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 1 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);
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
void TEST_MODULE_FULL_PIVOT_SEQUENCE(Polytopes::HyperplaneArrangement* arr)
{
    int new_i, new_j, ind, cc_i, cc_j;

    // Obtain all possible reverse criss-cross pivots
    MatrixXi reverse_cc = arr->getReverseCrissCrossPivots(); 

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(0))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(0, 0));      // Check that (0, 3) is a r.c.c. pivot
    assert(reverse_cc(0, 0) == 0);                    // Check that (0, 3) is the first possible r.c.c. pivot
    assert(reverse_cc(0, 1) == 0); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (3, 0) is the c.c. pivot from new dictionary
    assert(cc_i == new_i);
    assert(cc_j == new_j);

    // Obtain all possible reverse criss-cross pivots from the new dictionary
    MatrixXi reverse_cc2 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc2.rows() == 2);  

    // --------------------------------------------------------------- //
    //             PIVOTING 1 (basis(0)) AND 0 (cobasis(0))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(0, 0));      // Check that (1, 0) is a r.c.c. pivot
    assert(reverse_cc2(0, 0) == 0);                   // Check that (1, 0) is the first possible r.c.c. pivot from this dictionary
    assert(reverse_cc2(0, 1) == 0); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (0, 1) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc3 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc3.rows() == 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 1 (cobasis(0))            //
    //              (TO RE-OBTAIN THE PREVIOUS DICTIONARY)             //
    // --------------------------------------------------------------- //
    assert(!arr->isReverseCrissCrossPivot(0, 0));     // Check that (0, 1) is *not* a r.c.c. pivot
    arr->pivot(0, 0);                                 // Perform the pivot

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(1)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(1, 1));      // Check that (2, 4) is a r.c.c. pivot
    assert(reverse_cc2(1, 0) == 1);                   // Check that (2, 4) is the next possible r.c.c. pivot from this dictionary
    assert(reverse_cc2(1, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 4) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc4 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc4.rows() == 0); 

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
    assert(arr->isReverseCrissCrossPivot(1, 1));      // Check that (1, 4) is a r.c.c. pivot
    assert(reverse_cc(1, 0) == 1);                    // Check that (1, 4) is the next possible r.c.c. pivot from this dictionary
    assert(reverse_cc(1, 1) == 1);
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (1, 4) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Obtain all possible reverse criss-cross pivots from the new dictionary
    MatrixXi reverse_cc5 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc5.rows() == 2);  

    // --------------------------------------------------------------- //
    //             PIVOTING 0 (basis(0)) AND 3 (cobasis(1))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(0, 1));      // Check that (0, 3) is a r.c.c. pivot
    assert(reverse_cc5(0, 0) == 0);                   // Check that (0, 3) is the first possible r.c.c. pivot from this dictionary
    assert(reverse_cc5(0, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(0, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (0, 3) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc6 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc6.rows() == 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(1)) AND 0 (cobasis(0))            //
    //              (TO RE-OBTAIN THE PREVIOUS DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(1, 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(1)) AND 3 (cobasis(1))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(1, 1));      // Check that (2, 3) is a r.c.c. pivot
    assert(reverse_cc5(1, 0) == 1);                   // Check that (2, 3) is the first possible r.c.c. pivot from this dictionary
    assert(reverse_cc5(1, 1) == 1); 
    std::tie(new_i, new_j, ind) = arr->pivot(1, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 3) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc7 = arr->getReverseCrissCrossPivots();
    assert(reverse_cc7.rows() == 0);

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
    assert(arr->isReverseCrissCrossPivot(2, 0));      // Check that (2, 3) is a r.c.c. pivot
    assert(reverse_cc(2, 0) == 2);                    // Check that (2, 3) is the third possible r.c.c. pivot from this dictionary
    assert(reverse_cc(2, 1) == 0);
    std::tie(new_i, new_j, ind) = arr->pivot(2, 0);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 3) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc8 = arr->getReverseCrissCrossPivots(); 
    assert(reverse_cc8.rows() == 0); 

    // --------------------------------------------------------------- //
    //             PIVOTING 3 (basis(2)) AND 2 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 2 (basis(2)) AND 4 (cobasis(1))            //
    // --------------------------------------------------------------- //
    assert(arr->isReverseCrissCrossPivot(2, 1));      // Check that (2, 4) is a r.c.c. pivot
    assert(reverse_cc(3, 0) == 2);                    // Check that (2, 4) is the fourth possible r.c.c. pivot from this dictionary
    assert(reverse_cc(3, 1) == 1);
    std::tie(new_i, new_j, ind) = arr->pivot(2, 1);   // Perform the pivot
    std::tie(cc_i, cc_j) = arr->findCrissCross();     // Check that (2, 4) is the c.c. pivot from new dictionary
    assert(cc_i == new_i); 
    assert(cc_j == new_j); 

    // Check that there are *no* possible reverse criss-cross pivots from the
    // new dictionary
    MatrixXi reverse_cc9 = arr->getReverseCrissCrossPivots(); 
    assert(reverse_cc9.rows() == 0);

    // --------------------------------------------------------------- //
    //             PIVOTING 4 (basis(2)) AND 2 (cobasis(0))            //
    //              (TO RE-OBTAIN THE ORIGINAL DICTIONARY)             //
    // --------------------------------------------------------------- //
    arr->pivot(2, 0);
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
void TEST_MODULE_SEARCH_FROM_OPTIMAL_DICT_CRISSCROSS(Polytopes::HyperplaneArrangement* arr)
{
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
    assert(bases == bases_correct); 
}

/**
 * Test that `enumVertices()` correctly enumerates the five vertices of the 
 * arrangement, in the same fashion as in `searchFromOptimalDictCrissCross()`.
 */
void TEST_MODULE_ENUM_VERTICES(Polytopes::HyperplaneArrangement* arr)
{
    MatrixXr vertices = arr->enumVertices();
    std::cout << vertices << std::endl;  
}

int main(int argc, char** argv)
{
    // Instantiate a hyperplane arrangement from the given file 
    Polytopes::HyperplaneArrangement* arr = TEST_MODULE_PARSE(); 
    std::cout << "TEST_MODULE_PARSE: all tests passed" << std::endl;

    // Test that the first few pivots in the sequence in Avis & Fukuda (1992)
    // are performed correctly 
    TEST_MODULE_PARTIAL_PIVOT_SEQUENCE(arr);
    std::cout << "TEST_MODULE_PARTIAL_PIVOT_SEQUENCE: all tests passed" << std::endl;

    // Test that the full sequence of pivots in Avis & Fukuda (1992) are
    // performed correctly, and the reverse criss-cross pivots among them
    // are identified correctly
    TEST_MODULE_FULL_PIVOT_SEQUENCE(arr); 
    std::cout << "TEST_MODULE_FULL_PIVOT_SEQUENCE: all tests passed" << std::endl;  
    
    // Test that searchFromOptimalDictCrissCross() visits the full sequence
    // of pivots in Avis & Fukuda (1992) in the correct order 
    TEST_MODULE_SEARCH_FROM_OPTIMAL_DICT_CRISSCROSS(arr);
    std::cout << "TEST_MODULE_SEARCH_FROM_OPTIMAL_DICT_CRISSCROSS: all tests passed" << std::endl;

    // Test that enumVertices() correctly enumerates the five vertices of the 
    // arrangement, in the same fashion as in searchFromOptimalDictCrissCross()
    TEST_MODULE_ENUM_VERTICES(arr);
    std::cout << "TEST_MODULE_ENUM_VERTICES: all tests passed" << std::endl;

    delete arr; 
    return 0;
}
