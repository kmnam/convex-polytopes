/**
 * A simple script that enumerates the vertices of a convex polytope from its 
 * halfspace representation. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     3/3/2022
 */

#include <iostream>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include "../include/vertexEnum.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
using Polytopes::PolyhedralDictionarySystem; 
using Polytopes::InequalityType; 

int main(int argc, char** argv)
{
    // Parse the input file containing the polytope's linear constraints ... 
    std::string filename = argv[1];

    // ... and instantiate the corresponding polyhedral dictionary system 
    PolyhedralDictionarySystem* dict = new PolyhedralDictionarySystem(InequalityType::LessThanOrEqualTo);
    dict->parse(filename);  
     
    // Print the enumerated vertices to stdout 
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices(); 
    std::cout << vertices << std::endl;  

    return 0; 
}
