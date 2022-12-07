/**
 * A simple script that enumerates the vertices of a convex polytope from its 
 * halfspace representation. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     12/7/2022
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
    // Check that the input file name was specified 
    if (argc != 2)
        throw std::runtime_error("Invalid call signature"); 

    // Parse the input file containing the polytope's linear constraints ... 
    std::string filename = argv[1];

    // ... and instantiate the corresponding polyhedral dictionary system 
    PolyhedralDictionarySystem* dict = new PolyhedralDictionarySystem(InequalityType::LessThanOrEqualTo);
    dict->parse(filename);

    // If the input inequalities were specified as >=, switch to <=
    if (dict->getInequalityType() == InequalityType::GreaterThanOrEqualTo)
        dict->switchInequalityType(); 
     
    // Print the enumerated vertices to stdout
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices();
    for (int i = 0; i < vertices.rows(); ++i)
    {
        for (int j = 0; j < vertices.cols() - 1; ++j)
            std::cout << vertices(i, j) << " ";
        std::cout << vertices(i, vertices.cols() - 1) << std::endl; 
    } 

    delete dict; 
    return 0; 
}
