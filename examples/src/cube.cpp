#include <iostream>
#include <boost/random.hpp>
#include "../../include/polytopes.hpp"

/**
 * A brief example that samples points from the unit cube in three dimensions.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/17/2022
 */
// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

// Internal floating-point precision
const int INTERNAL_PRECISION = 100; 

int main(int argc, char** argv)
{
    unsigned n;
    sscanf(argv[1], "%u", &n);    // Parse number of points to sample 
    MatrixXd params = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>("cube.vert", n, 0, rng);

    // Print the sampled points to stdout  
    std::cout << params << std::endl; 
    
    return 0;
}
