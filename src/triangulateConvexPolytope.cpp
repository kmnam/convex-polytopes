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
#include "../include/linearConstraints.hpp"
#include "../include/vertexEnum.hpp"

using namespace Eigen;
using Polytopes::InequalityType; 
using Polytopes::PolyhedralDictionarySystem;
typedef Delaunay_triangulation::Finite_full_cell_iterator  Finite_full_cell_iterator; 
typedef Delaunay_triangulation::Vertex_handle              Vertex_handle; 

// Internal precision for computing simplex volumes
const int INTERNAL_PRECISION = 100; 

int main(int argc, char** argv)
{
    // Check input arguments 
    if (argc != 2)
        throw std::runtime_error("Invalid call signature");

    // Parse the input file containing the polytope's linear constraints ... 
    std::string filename = argv[1];

    // ... and instantiate the corresponding polyhedral dictionary system ...  
    PolyhedralDictionarySystem* dict = new PolyhedralDictionarySystem(InequalityType::LessThanOrEqualTo);
    dict->parse(filename);  
     
    // ... then enumerate the polytope's vertices ... 
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices();
    int dim = vertices.cols(); 

    // ... which we then use to triangulate the polytope 
    Delaunay_triangulation tri = Polytopes::triangulate(vertices);

    // Print each simplex in the triangulation to stdout
    for (auto it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); ++it)
    {
        for (int i = 0; i <= dim; ++i)
        { 
            Vertex_handle v = it->vertex(i);
            for (int j = 0; j < dim - 1; ++j)
                std::cout << v->point()[j] << ", ";
            std::cout << v->point()[dim - 1] << std::endl; 
        }
        std::cout << "-----" << std::endl;  
    } 

    delete dict; 
    return 0; 
}
