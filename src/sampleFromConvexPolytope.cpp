/**
 * A simple script that samples from the interior of a convex polytope. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     12/7/2022
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include "../include/polytopes.hpp"

using namespace Eigen;

// Internal precision for computing simplex volumes
const int INTERNAL_PRECISION = 100; 

int main(int argc, char** argv)
{
    // Check input arguments 
    if (argc != 3)
        throw std::runtime_error("Invalid call signature");

    // Initialize the random number generator 
    boost::random::mt19937 rng(1234567890); 

    // Parse the input file containing the polytope's vertices ...
    std::string filename = argv[1];
     
    // ... and the number of points to sample from the interior 
    int n = std::stoi(argv[2]); 

    // Print the sampled points to stdout
    MatrixXd points = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(filename, n, 0, rng);
    std::cout << points << std::endl;  

    return 0; 
}
