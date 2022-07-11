/**
 * A simple script that enumerates the vertices of a convex polytope from its 
 * halfspace representation. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     7/11/2022
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
    if (argc == 3 && !strcmp("--geq"))
        dict->parse(filename, InequalityType::GreaterThanOrEqualTo); 
    else if (argc == 2) 
        dict->parse(filename);
    else
        throw std::runtime_error("Invalid call signature"); 
     
    // Print the enumerated vertices to stdout 
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices(); 
    std::cout << vertices << std::endl;  

    delete dict; 
    return 0; 
}
