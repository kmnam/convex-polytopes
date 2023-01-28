/**
 * Functions for manipulating convex polytopes, simplices, and triangulations. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/28/2023
 */

#ifndef POLYTOPES_HPP 
#define POLYTOPES_HPP 

#include <assert.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <CGAL/Epick_d.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Triangulation.h>
#include <CGAL/Delaunay_triangulation.h>

using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::gmp_int; 
using boost::multiprecision::mpq_rational;
using boost::multiprecision::mpfr_float_backend;

typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag>         Kd;
typedef CGAL::Delaunay_triangulation<Kd>                   Delaunay_triangulation; 
typedef Delaunay_triangulation::Point                      Point;  
typedef Delaunay_triangulation::Full_cell_handle           Full_cell_handle;
typedef Delaunay_triangulation::Full_cell_iterator         Full_cell_iterator; 
typedef Delaunay_triangulation::Finite_full_cell_iterator  Finite_full_cell_iterator; 
typedef Delaunay_triangulation::Facet                      Facet;
typedef Delaunay_triangulation::Vertex                     Vertex;

namespace Polytopes {

/**
 * A simple wrapper class that contains a matrix of rational vertex
 * coordinates for a simplex. 
 */
class Simplex
{
    private:
        // Matrix of vertex coordinates (each row is a vertex)
        Matrix<mpq_rational, Dynamic, Dynamic> vertices;  

    public:
        /**
         * Trivial constructor that takes as input the matrix of *rational* 
         * vector coordinates. 
         */
        Simplex(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices) 
        {
            this->vertices = vertices; 
        }

        /**
         * A more flexible constructor that takes as input any matrix-based 
         * expression storing the vector coordinates. 
         */
        template <typename Derived>
        Simplex(const MatrixBase<Derived>& vertices) 
        {
            this->vertices = vertices.template cast<mpq_rational>(); 
        }

        /**
         * Trivial destructor. 
         */
        ~Simplex()
        {
        }

        /**
         * Return the i-th vertex in the simplex as a (column) vector. 
         */
        Matrix<mpq_rational, Dynamic, 1> getVertex(const int i) 
        {
            return this->vertices.row(i); 
        }

        /**
         * Compute the (square root of the absolute value of the) Cayley-Menger
         * determinant associated with the simplex.
         *
         * This quantity, when comparing amongst simplices of the same
         * dimension in the same ambient space, is proportional to the
         * simplex's volume.
         *
         * The determinant is computed using rational arithmetic; the square 
         * root is computed using floating-point arithmetic with the given 
         * precision. 
         *
         * @returns Square root of the absolute value of the Cayley-Menger
         *          determinant.   
         */
        template <int FloatPrecision>
        number<mpfr_float_backend<FloatPrecision> > sqrtAbsCayleyMenger()
        {
            // Compute the symmetric distance matrix ...
            int nv = this->vertices.rows(); 
            Matrix<mpq_rational, Dynamic, Dynamic> A = Matrix<mpq_rational, Dynamic, Dynamic>::Zero(nv + 1, nv + 1); 
            for (int j = 0; j < nv; ++j)
            {
                for (int k = j + 1; k < nv; ++k) 
                {
                    mpq_rational sqdist = (this->vertices.row(j) - this->vertices.row(k)).squaredNorm(); 
                    A(j, k) = sqdist; 
                    A(k, j) = sqdist; 
                }
                A(j, nv) = 1;
                A(nv, j) = 1;  
            }
            
            // ... and its determinant  
            mpq_rational absdet = boost::multiprecision::abs(A.determinant());
            number<gmp_int> numer = boost::multiprecision::numerator(absdet); 
            number<gmp_int> denom = boost::multiprecision::denominator(absdet);
            number<mpfr_float_backend<FloatPrecision> > ratio = numer / denom;   
            return boost::multiprecision::sqrt(ratio); 
        }

        /**
         * Given a desired number of points, randomly sample the given number
         * of points from the uniform density (i.e., flat Dirichlet) on the
         * simplex.
         *
         * @param npoints  Number of points to sample from the simplex.
         * @param rng      Reference to random number generator instance.
         * @returns        Matrix of sampled point coordinates. 
         */
        template <int FloatPrecision>
        Matrix<number<mpfr_float_backend<FloatPrecision> >, Dynamic, Dynamic> sample(int npoints, boost::random::mt19937& rng)
        {
            int dim = this->vertices.cols();     // Dimension of the ambient space
            int nv = this->vertices.rows();      // Number of vertices

            // Sample the desired number of points from the flat Dirichlet 
            // distribution on the standard simplex of appropriate dimension
            //
            // This sampling is performed with double scalars 
            MatrixXd barycentric(npoints, nv); 
            boost::random::gamma_distribution<double> gamma_dist(1.0);
            for (int i = 0; i < npoints; ++i)
            {
                // Sample (nv) independent Gamma-distributed variables with alpha = 1,
                // and normalize by their sum
                for (int j = 0; j < nv; ++j)
                    barycentric(i, j) = gamma_dist(rng);
                barycentric.row(i) = barycentric.row(i) / barycentric.row(i).sum();
            }
           
            // Convert from barycentric coordinates to Cartesian coordinates
            Matrix<number<mpfr_float_backend<FloatPrecision> >, Dynamic, Dynamic> points(npoints, dim); 
            for (int i = 0; i < npoints; ++i)
            {
                points.row(i) = (
                    barycentric.row(i).template cast<mpq_rational>() * this->vertices
                ).template cast<number<mpfr_float_backend<FloatPrecision> > >(); 
            }

            return points;
        }
}; 

/**
 * Write a comma-delimited string from the given `std::vector` of integers.
 *
 * @param v Given vector of integers. 
 * @returns Comma-delimited string containing the integers. 
 */
std::string intVectorToString(const std::vector<int>& v)
{
    std::stringstream ss;
    for (auto it = v.begin(); it != v.end(); ++it)
        ss << *it << ","; 
    int size = ss.str().length(); 
    return ss.str().substr(0, size - 1);
}

/**
 * Split the given comma-delimited string and write the integers to a `std::vector`.
 *
 * @param s Given string of integers. 
 * @returns Vector containing the integers. 
 */
std::vector<int> stringToIntVector(const std::string s) 
{
    std::stringstream ss(s);
    std::string token;  
    std::vector<int> v; 
    while (std::getline(ss, token, ','))
        v.push_back(std::stoi(token));
    return v;  
}

/**
 * Recursively generate all combinations of `k` items from the given `std::vector`
 * of integers.  
 */
std::vector<std::vector<int> > combinations(const std::vector<int>& v, const int k)
{
    std::vector<std::vector<int> > combos;

    // If we are to return all 1-combinations ...
    if (k == 1) 
    {
        for (auto it = v.begin(); it != v.end(); ++it)
        {
            std::vector<int> c = {*it}; 
            combos.push_back(c);
        } 
        return combos; 
    }
    // Otherwise, if we are to return the entire list of items as a single 
    // combination ... 
    else if (v.size() == k) 
    {
        std::vector<int> c(v); 
        combos.push_back(c); 
        return combos; 
    }
    else 
    {
        for (int i = 0; i < v.size(); ++i)
        {
            // Here, we are gathering all combinations containing the i-th
            // item and no previous items 
            int curr = i;
            int next = i + 1; 

            // First check that there are enough remaining items left in v
            // to skip over the previous i items 
            if (k <= v.size() - i)
            {
                std::vector<int> remaining(v.begin() + next, v.end()); 
                std::vector<std::vector<int> > subcombos = combinations(remaining, k - 1);
                for (auto it2 = subcombos.begin(); it2 != subcombos.end(); ++it2)
                {
                    std::vector<int> combo = {v[curr]}; 
                    combo.insert(combo.end(), it2->begin(), it2->end());
                    combos.push_back(combo);  
                }
            }
        }
        return combos; 
    }
}

/**
 * Return a vector of `Simplex` objects containing the vertex coordinates of 
 * the *full-dimensional faces* in the interior of the given triangulation. 
 *
 * @param tri Input triangulation.
 * @returns   Vector of `Simplex` objects, each containing the vertex coordinates
 *            of a full-dimensional face. 
 */
std::vector<Simplex> getFullDimFaces(Delaunay_triangulation& tri) 
{
    int dim = tri.current_dimension();  
    std::vector<Simplex> faces;

    // Run through the finite full-dimensional simplices in the triangulation ...  
    for (Finite_full_cell_iterator it = tri.finite_full_cells_begin(); it != tri.finite_full_cells_end(); ++it)
    {
        Matrix<mpq_rational, Dynamic, Dynamic> face_vertex_coords(dim + 1, dim);
        
        // For each vertex in the face ...
        for (int i = 0; i < dim + 1; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = it->vertex(i)->point();
            for (int j = 0; j < dim; ++j)
                face_vertex_coords(i, j) = static_cast<mpq_rational>(CGAL::to_double(p[j]));  
        }
        faces.emplace_back(Simplex(face_vertex_coords));
    }
    
    return faces;  
}

/**
 * Return a vector of `Simplex` objects containing the vertex coordinates of 
 * the *full-dimensional faces* in the interior of the given triangulation. 
 *
 * @param tri Pointer to dynamically allocated input triangulation.
 * @returns   Vector of `Simplex` objects, each containing the vertex coordinates
 *            of a full-dimensional face. 
 */
std::vector<Simplex> getFullDimFaces(Delaunay_triangulation* tri) 
{
    int dim = tri->current_dimension();  
    std::vector<Simplex> faces;

    // Run through the finite full-dimensional simplices in the triangulation ...  
    for (Finite_full_cell_iterator it = tri->finite_full_cells_begin(); it != tri->finite_full_cells_end(); ++it)
    {
        Matrix<mpq_rational, Dynamic, Dynamic> face_vertex_coords(dim + 1, dim);
        
        // For each vertex in the face ...
        for (int i = 0; i < dim + 1; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = it->vertex(i)->point();
            for (int j = 0; j < dim; ++j)
            {
                std::cout << p[j] << " " << CGAL::to_double(p[j]) << " " << static_cast<mpq_rational>(CGAL::to_double(p[j])) << std::endl;
                face_vertex_coords(i, j) = static_cast<mpq_rational>(CGAL::to_double(p[j]));
            } 
        }
        faces.emplace_back(Simplex(face_vertex_coords));
    }
    
    return faces;  
}

/**
 * Return the vertex handles of the *facets* on the boundary of the given
 * triangulation.
 *
 * @param tri Input triangulation.
 * @returns   Vector of vectors of vertex handles, each inner vector containing 
 *            handles to the vertices of a boundary facet. 
 */
std::vector<Simplex> getBoundaryFacets(Delaunay_triangulation& tri)
{
    int dim = tri.current_dimension();
    std::vector<Simplex> facets;  

    // Run through the full-dimensional simplices (both finite and infinite) 
    // in the triangulation ...  
    for (Full_cell_iterator it = tri.full_cells_begin(); it != tri.full_cells_end(); ++it)
    {
        Matrix<mpq_rational, Dynamic, Dynamic> facet_vertex_coords(dim, dim); 

        // The facets are in bijective correspondence with the *infinite* 
        // full-dimensional simplices in the triangulation 
        if (!tri.is_infinite(it))
            continue;
        Facet facet(it, it->index(tri.infinite_vertex()));

        // Get the simplex containing the facet and its co-vertex 
        Full_cell_handle c = tri.full_cell(facet); 
        int j = tri.index_of_covertex(facet);

        // The vertices of the facet are the vertex of the simplex minus the 
        // co-vertex 
        for (int i = 0; i < j; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = c->vertex(i)->point();
            for (int k = 0; k < dim; ++k)
                facet_vertex_coords(i, k) = static_cast<mpq_rational>(CGAL::to_double(p[k]));  
        } 
        for (int i = j + 1; i < tri.current_dimension() + 1; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = c->vertex(i)->point();
            for (int k = 0; k < dim; ++k)
                facet_vertex_coords(i - 1, k) = static_cast<mpq_rational>(CGAL::to_double(p[k]));  
        } 

        facets.emplace_back(Simplex(facet_vertex_coords)); 
    }

    return facets;  
}

/**
 * Return the vertex handles of the *facets* on the boundary of the given
 * triangulation.
 *
 * @param tri Pointer to dynamically allocated input triangulation.
 * @returns   Vector of vectors of vertex handles, each inner vector containing 
 *            handles to the vertices of a boundary facet. 
 */
std::vector<Simplex> getBoundaryFacets(Delaunay_triangulation* tri)
{
    int dim = tri->current_dimension();
    std::vector<Simplex> facets;  

    // Run through the full-dimensional simplices (both finite and infinite) 
    // in the triangulation ...  
    for (Full_cell_iterator it = tri->full_cells_begin(); it != tri->full_cells_end(); ++it)
    {
        Matrix<mpq_rational, Dynamic, Dynamic> facet_vertex_coords(dim, dim); 

        // The facets are in bijective correspondence with the *infinite* 
        // full-dimensional simplices in the triangulation 
        if (!tri->is_infinite(it))
            continue;
        Facet facet(it, it->index(tri->infinite_vertex()));

        // Get the simplex containing the facet and its co-vertex 
        Full_cell_handle c = tri->full_cell(facet); 
        int j = tri->index_of_covertex(facet);

        // The vertices of the facet are the vertex of the simplex minus the 
        // co-vertex 
        for (int i = 0; i < j; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = c->vertex(i)->point();
            for (int k = 0; k < dim; ++k)
                facet_vertex_coords(i, k) = static_cast<mpq_rational>(CGAL::to_double(p[k]));  
        } 
        for (int i = j + 1; i < tri->current_dimension() + 1; ++i)
        {
            // Get the coordinates of the i-th vertex 
            Point p = c->vertex(i)->point();
            for (int k = 0; k < dim; ++k)
                facet_vertex_coords(i - 1, k) = static_cast<mpq_rational>(CGAL::to_double(p[k]));  
        } 

        facets.emplace_back(Simplex(facet_vertex_coords)); 
    }

    return facets;  
}

/**
 * Given a triangulation, return the faces of the given codimension on the 
 * boundary of the triangulation. 
 *
 * @param tri   Input triangulation. 
 * @param codim Desired codimension.
 * @returns     Vector of vectors of vertex handles, each inner vector containing
 *              handles to the vertices of a boundary face of the given codimension. 
 */
std::vector<Simplex> getBoundaryFaces(Delaunay_triangulation& tri, const int codim) 
{
    // Only faces of codimension 1, ..., dimension - 1 are valid 
    int tri_dim = tri.current_dimension(); 
    if (codim < 1 || codim > tri_dim - 1) 
        throw std::invalid_argument("Invalid codimension specified");
    int face_dim = tri_dim - codim;

    // If the desired faces are *facets* (codimension is 1), simply return them 
    std::vector<Simplex> facets = getBoundaryFacets(tri); 
    if (codim == 1) 
        return facets;

    // Each simplex of dimension D is spanned by D + 1 vertices 
    //
    // Therefore, to generate all possible sub-faces of the boundary facets
    // (dimension = tri_dim - 1), we generate all (face_dim + 1)-combinations
    // of the range 0, ..., tri_dim - 1 of the length corresponding to the
    // desired codimension
    std::vector<int> idx; 
    for (int i = 0; i < tri_dim; ++i)
        idx.push_back(i);
    std::vector<std::vector<int> > combos = combinations(idx, face_dim + 1);

    // Get the coordinates of all vertices in the triangulation in a single 
    // matrix
    Matrix<mpq_rational, Dynamic, Dynamic> all_vertex_coords(tri.number_of_vertices(), tri_dim);
    int i = 0;  
    for (auto it = tri.vertices_begin(); it != tri.vertices_end(); ++it) 
    {
        if (!tri.is_infinite(*it))
        {
            // Get the coordinates of each vertex
            Point p = it->point(); 
            for (int j = 0; j < tri_dim; ++j) 
                all_vertex_coords(i, j) = static_cast<mpq_rational>(CGAL::to_double(p[j]));
            i++;  
        }
    }

    // Run through the boundary facets ... 
    std::vector<Simplex> faces; 
    std::unordered_set<std::string> face_strings; 
    for (auto&& f : facets) 
    {
        // Get the indices of the vertices of the facet, as dictated by the 
        // matrix of vertex coordinates for the full triangulation 
        std::vector<int> vertex_indices_in_facet;
        for (int i = 0; i < tri_dim; ++i)
        {
            Matrix<mpq_rational, Dynamic, 1> v = f.getVertex(i);
            Matrix<mpq_rational, Dynamic, 1>::Index nearest;
            (all_vertex_coords.rowwise() - v.transpose()).cwiseAbs().rowwise().sum().minCoeff(&nearest);
            vertex_indices_in_facet.push_back(static_cast<int>(nearest));
        }

        // Sort the vertex indices
        std::sort(vertex_indices_in_facet.begin(), vertex_indices_in_facet.end());

        // For each combination of vertices (which corresponds to a sub-face
        // of the boundary facet of the desired codimension), generate the
        // corresponding integer-string and check that it wasn't previously
        // encountered
        for (auto&& c : combos)
        {
            std::vector<int> sub; 
            for (auto it = c.begin(); it != c.end(); ++it)
                sub.push_back(vertex_indices_in_facet[*it]);
            std::string s = intVectorToString(sub);

            // If the combination (sub-face) was *not* encountered, then 
            // add it to the vector to be returned
            if (face_strings.find(s) == face_strings.end())
            {
                faces.emplace_back(Simplex(all_vertex_coords(sub, all)));
                face_strings.insert(s); 
            }
        }
    }

    return faces; 
}

/**
 * Given a triangulation, return the faces of the given codimension on the 
 * boundary of the triangulation. 
 *
 * @param tri   Pointer to dynamically allocated input triangulation. 
 * @param codim Desired codimension.
 * @returns     Vector of vectors of vertex handles, each inner vector containing
 *              handles to the vertices of a boundary face of the given codimension. 
 */
std::vector<Simplex> getBoundaryFaces(Delaunay_triangulation* tri, const int codim) 
{
    // Only faces of codimension 1, ..., dimension - 1 are valid 
    int tri_dim = tri->current_dimension(); 
    if (codim < 1 || codim > tri_dim - 1) 
        throw std::invalid_argument("Invalid codimension specified");
    int face_dim = tri_dim - codim;

    // If the desired faces are *facets* (codimension is 1), simply return them 
    std::vector<Simplex> facets = getBoundaryFacets(tri); 
    if (codim == 1) 
        return facets;

    // Each simplex of dimension D is spanned by D + 1 vertices 
    //
    // Therefore, to generate all possible sub-faces of the boundary facets
    // (dimension = tri_dim - 1), we generate all (face_dim + 1)-combinations
    // of the range 0, ..., tri_dim - 1 of the length corresponding to the
    // desired codimension
    std::vector<int> idx; 
    for (int i = 0; i < tri_dim; ++i)
        idx.push_back(i);
    std::vector<std::vector<int> > combos = combinations(idx, face_dim + 1);

    // Get the coordinates of all vertices in the triangulation in a single 
    // matrix
    Matrix<mpq_rational, Dynamic, Dynamic> all_vertex_coords(tri->number_of_vertices(), tri_dim);
    int i = 0;  
    for (auto it = tri->vertices_begin(); it != tri->vertices_end(); ++it) 
    {
        if (!tri->is_infinite(*it))
        {
            // Get the coordinates of each vertex
            Point p = it->point(); 
            for (int j = 0; j < tri_dim; ++j) 
                all_vertex_coords(i, j) = static_cast<mpq_rational>(CGAL::to_double(p[j]));
            i++;  
        }
    }

    // Run through the boundary facets ... 
    std::vector<Simplex> faces; 
    std::unordered_set<std::string> face_strings; 
    for (auto&& f : facets) 
    {
        // Get the indices of the vertices of the facet, as dictated by the 
        // matrix of vertex coordinates for the full triangulation 
        std::vector<int> vertex_indices_in_facet;
        for (int i = 0; i < tri_dim; ++i)
        {
            Matrix<mpq_rational, Dynamic, 1> v = f.getVertex(i);
            Matrix<mpq_rational, Dynamic, 1>::Index nearest;
            (all_vertex_coords.rowwise() - v.transpose()).cwiseAbs().rowwise().sum().minCoeff(&nearest);
            vertex_indices_in_facet.push_back(static_cast<int>(nearest));
        }

        // Sort the vertex indices
        std::sort(vertex_indices_in_facet.begin(), vertex_indices_in_facet.end());

        // For each combination of vertices (which corresponds to a sub-face
        // of the boundary facet of the desired codimension), generate the
        // corresponding integer-string and check that it wasn't previously
        // encountered
        for (auto&& c : combos)
        {
            std::vector<int> sub; 
            for (auto it = c.begin(); it != c.end(); ++it)
                sub.push_back(vertex_indices_in_facet[*it]);
            std::string s = intVectorToString(sub);

            // If the combination (sub-face) was *not* encountered, then 
            // add it to the vector to be returned
            if (face_strings.find(s) == face_strings.end())
            {
                faces.emplace_back(Simplex(all_vertex_coords(sub, all)));
                face_strings.insert(s); 
            }
        }
    }

    return faces; 
}

/**
 * Given a matrix of coordinates for the *vertices* of a convex polytope, 
 * return the Delaunay triangulation of the convex polytope.
 * 
 * @param vertices Matrix of vertex coordinates. 
 * @returns        `Delaunay_triangulation` instance storing the triangulation. 
 */
Delaunay_triangulation triangulate(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices)
{
    // The number of columns in the matrix gives the dimension of the polytope's 
    // ambient space 
    int dim = vertices.cols();

    // Convert coordinates to doubles 
    MatrixXd _vertices = vertices.cast<double>(); 

    // Instantiate the Delaunay triangulation and insert each vertex 
    Delaunay_triangulation tri(dim);
    for (int i = 0; i < _vertices.rows(); ++i) 
        tri.insert(Point(dim, _vertices.row(i).begin(), _vertices.row(i).end())); 

    return tri; 
}

/**
 * Given an existing `Delaunay_triangulation` instance and a matrix of coordinates
 * for the *vertices* of a convex polytope, *update* the given triangulation.
 *
 * @param vertices Matrix of vertex coordinates.
 * @param tri      Reference to existing `Delaunay_triangulation` instance. 
 * @returns        Reference to updated `Delaunay_triangulation` instance. 
 */
Delaunay_triangulation& triangulate(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices,
                                    Delaunay_triangulation& tri)
{
    // The number of columns in the matrix gives the dimension of the polytope's 
    // ambient space 
    int dim = vertices.cols();

    // Convert coordinates to doubles 
    MatrixXd _vertices = vertices.cast<double>(); 

    // Clear the existing triangulation
    tri.clear(); 

    // Add each new vertex into the triangulation 
    for (int i = 0; i < _vertices.rows(); ++i) 
        tri.insert(Point(dim, _vertices.row(i).begin(), _vertices.row(i).end())); 

    return tri; 
}

/**
 * Given a *pointer* to a dynamically allocated `Delaunay_triangulation` instance
 * and a matrix of coordinates for the *vertices* of a convex polytope, *update*
 * the given triangulation.
 *
 * @param vertices Matrix of vertex coordinates.
 * @param tri      Pointer to existing `Delaunay_triangulation` instance. 
 * @returns        Reference to updated `Delaunay_triangulation` instance. 
 */
void triangulate(const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices,
                 Delaunay_triangulation* tri)
{
    // The number of columns in the matrix gives the dimension of the polytope's 
    // ambient space 
    int dim = vertices.cols();

    // Convert coordinates to doubles 
    MatrixXd _vertices = vertices.cast<double>(); 

    // Clear the existing triangulation
    tri->clear(); 

    // Add each new vertex into the triangulation 
    for (int i = 0; i < _vertices.rows(); ++i) 
        tri->insert(Point(dim, _vertices.row(i).begin(), _vertices.row(i).end())); 
}

/**
 * Parse the given .vert file specifying a convex polytope in terms of its 
 * vertices, and return the matrix of vertex coordinates.
 *
 * The file is assumed to be non-empty.
 *
 * @param filename Path to input .vert polytope triangulation file. 
 * @returns        Matrix of vertex coordinates.
 */
Matrix<mpq_rational, Dynamic, Dynamic> parseVertexCoords(const std::string filename)
{
    // Parse the *first* line of the given input file to obtain the dimension
    // of the polytope's ambient space 
    std::string line, token; 
    std::ifstream infile(filename);
    std::getline(infile, line);

    // Each vertex is specified as a space-delimited string of N coefficients,
    // where N is the dimension of the ambient space 
    std::stringstream ss_first; 
    ss_first << line; 
    std::vector<mpq_rational> vertex_first; 
    while (std::getline(ss_first, token, ' '))
        vertex_first.push_back(mpq_rational(token)); 
    int dim = vertex_first.size();

    // Parse each subsequent line and store vertices in matrix
    int n = 1;  
    Matrix<mpq_rational, Dynamic, Dynamic> vertices(n, dim);
    for (int i = 0; i < dim; ++i)
        vertices(0, i) = vertex_first[i]; 
    while (std::getline(infile, line))
    {
        std::stringstream ss; 
        std::vector<mpq_rational> vertex; 
        ss << line;
        while (std::getline(ss, token, ' '))
            vertex.push_back(mpq_rational(token));
       
        // Does the current vertex match all previous vertices in length? 
        if (vertex.size() != dim)
            throw std::runtime_error("Vertices of multiple dimensions specified in input file");

        // Add vertex coordinates to matrix
        n++;
        vertices.conservativeResize(n, dim);
        for (int i = 0; i < dim; ++i)
            vertices(n - 1, i) = vertex[i]; 
    }

    return vertices; 
}

/**
 * Parse the given .vert file specifying a convex polytope in terms of its 
 * *vertices*, and return the Delaunay triangulation of the convex polytope.
 *
 * The file is assumed to be non-empty.  
 *
 * @param filename Path to input .vert polytope triangulation file.
 * @returns        `Delaunay_triangulation` instance storing the triangulation. 
 */
Delaunay_triangulation parseVerticesFile(const std::string filename) 
{
    // Parse the *first* line of the given input file to obtain the dimension
    // of the polytope's ambient space 
    std::string line, token; 
    std::ifstream infile(filename);
    std::getline(infile, line);

    // Each vertex is specified as a space-delimited string of N coefficients,
    // where N is the dimension of the ambient space 
    std::stringstream ss_first; 
    ss_first << line; 
    std::vector<mpq_rational> vertex_first; 
    while (std::getline(ss_first, token, ' '))
        vertex_first.push_back(mpq_rational(token));    // Parse each vertex as rational vector
    int dim = vertex_first.size();

    // Instantiate the Delaunay triangulation and insert the first point
    Delaunay_triangulation tri(dim); 
    tri.insert(Point(dim, vertex_first.begin(), vertex_first.end())); 

    // Parse each subsequent line in the input file 
    while (std::getline(infile, line))
    {
        std::stringstream ss; 
        std::vector<mpq_rational> vertex; 
        ss << line;
        while (std::getline(ss, token, ' '))
            vertex.push_back(mpq_rational(token));    // Parse each vertex as rational vector
       
        // Does the current vertex match all previous vertices in length? 
        if (vertex.size() != dim)
            throw std::runtime_error("Vertices of multiple dimensions specified in input file");

        // Insert the current point into the Delaunay triangulation 
        tri.insert(Point(dim, vertex.begin(), vertex.end()));
    }

    return tri; 
}

/**
 * Parse the given .vert file specifying a convex polytope in terms of its 
 * *vertices*, and *update* the given `Delaunay_triangulation` instance with 
 * a new Delaunay triangulation of the convex polytope. 
 *
 * The file is assumed to be non-empty.  
 *
 * @param filename Path to input .vert polytope triangulation file.
 * @param tri      Reference to existing `Delaunay_triangulation` instance. 
 * @returns        Reference to updated `Delaunay_triangulation` instance. 
 */
Delaunay_triangulation& parseVerticesFile(const std::string filename, Delaunay_triangulation& tri) 
{
    // Parse the *first* line of the given input file to obtain the dimension
    // of the polytope's ambient space 
    std::string line, token; 
    std::ifstream infile(filename);
    std::getline(infile, line);

    // Each vertex is specified as a space-delimited string of N coefficients,
    // where N is the dimension of the ambient space 
    std::stringstream ss_first; 
    ss_first << line; 
    std::vector<mpq_rational> vertex_first; 
    while (std::getline(ss_first, token, ' '))
        vertex_first.push_back(mpq_rational(token));    // Parse each vertex as rational vector
    int dim = vertex_first.size();

    // Clear the existing triangulation
    tri.clear(); 

    // Insert the first point into the given triangulation
    tri.insert(Point(dim, vertex_first.begin(), vertex_first.end())); 

    // Parse each subsequent line in the input file 
    while (std::getline(infile, line))
    {
        std::stringstream ss; 
        std::vector<mpq_rational> vertex; 
        ss << line;
        while (std::getline(ss, token, ' '))
            vertex.push_back(mpq_rational(token));    // Parse each vertex as rational vector
       
        // Does the current vertex match all previous vertices in length? 
        if (vertex.size() != dim)
            throw std::runtime_error("Vertices of multiple dimensions specified in input file");

        // Insert the current point into the Delaunay triangulation 
        tri.insert(Point(dim, vertex.begin(), vertex.end()));
    }

    return tri; 
}

/**
 * Parse the given .vert file specifying a convex polytope in terms of its 
 * *vertices*, and *update* the given dynamically allocated `Delaunay_triangulation`
 * instance with a new Delaunay triangulation of the convex polytope. 
 *
 * The file is assumed to be non-empty.  
 *
 * @param filename Path to input .vert polytope triangulation file.
 * @param tri      Pointer to existing `Delaunay_triangulation` instance. 
 * @returns        Reference to updated `Delaunay_triangulation` instance. 
 */
void parseVerticesFile(const std::string filename, Delaunay_triangulation* tri) 
{
    // Parse the *first* line of the given input file to obtain the dimension
    // of the polytope's ambient space 
    std::string line, token; 
    std::ifstream infile(filename);
    std::getline(infile, line);

    // Each vertex is specified as a space-delimited string of N coefficients,
    // where N is the dimension of the ambient space 
    std::stringstream ss_first; 
    ss_first << line; 
    std::vector<mpq_rational> vertex_first; 
    while (std::getline(ss_first, token, ' '))
        vertex_first.push_back(mpq_rational(token));    // Parse each vertex as rational vector
    int dim = vertex_first.size();

    // Clear the existing triangulation
    tri->clear(); 

    // Insert the first point into the given triangulation
    tri->insert(Point(dim, vertex_first.begin(), vertex_first.end())); 

    // Parse each subsequent line in the input file 
    while (std::getline(infile, line))
    {
        std::stringstream ss; 
        std::vector<mpq_rational> vertex; 
        ss << line;
        while (std::getline(ss, token, ' '))
            vertex.push_back(mpq_rational(token));    // Parse each vertex as rational vector
       
        // Does the current vertex match all previous vertices in length? 
        if (vertex.size() != dim)
            throw std::runtime_error("Vertices of multiple dimensions specified in input file");

        // Insert the current point into the Delaunay triangulation 
        tri->insert(Point(dim, vertex.begin(), vertex.end()));
    }
}

/**
 * Given a Delaunay triangulation of a convex polytope, sample uniformly from
 * different subsets of the polytope: 
 *
 * - If `codim == 0`, then sample from the *interior* of the polytope (the 
 *   union of its full-dimensional faces). 
 * - If `codim > 0`, then sample from the subset of the *boundary* of the 
 *   polytope formed by the union of the faces of the given codimension on 
 *   the polytope's boundary. 
 *
 * Note that the sampled points are returned as doubles.  
 *
 * @param tri      Input triangulation.
 * @param npoints  Number of points to sample from the polytope.
 * @param codim    Codimension of the simplices to sample from. 
 * @param rng      Reference to existing random number generator instance.
 * @returns        Delaunay triangulation parsed from the given file and the 
 *                 matrix of sampled points. 
 */
template <int CayleyMengerPrecision, int SamplePrecision = CayleyMengerPrecision> 
MatrixXd sampleFromConvexPolytope(Delaunay_triangulation& tri, const int npoints,
                                  const int codim, boost::random::mt19937& rng)
{
    // Obtain the desired subset of simplices in the triangulation  
    std::vector<Simplex> faces; 
    if (codim == 0) 
        faces = getFullDimFaces(tri); 
    else if (codim == 1) 
        faces = getBoundaryFacets(tri);
    else 
        faces = getBoundaryFaces(tri, codim);

    // Compute the (scaled) volume of each face 
    int dim = tri.current_dimension();
    Matrix<number<mpfr_float_backend<CayleyMengerPrecision> >, Dynamic, 1> volumes(faces.size()); 
    for (int i = 0; i < faces.size(); ++i) 
        volumes(i) = faces[i].sqrtAbsCayleyMenger<CayleyMengerPrecision>();

    // Instantiate a categorical distribution with probabilities 
    // proportional to the *boundary* simplex volumes
    //
    // The volumes are normalized into probabilities using double
    // arithmetic  
    VectorXd probs = (volumes / volumes.sum()).template cast<double>();
    std::vector<double> probs_vec; 
    for (int i = 0; i < probs.size(); ++i)
        probs_vec.push_back(probs(i));
    boost::random::discrete_distribution<> dist(probs_vec);

    // Sample from the desired subset of simplices
    MatrixXd sample(npoints, dim); 
    for (int i = 0; i < npoints; ++i)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Get the corresponding simplex
        sample.row(i) = faces[j].sample<SamplePrecision>(1, rng).template cast<double>();
    }
    
    return sample;
}

/**
 * Given a *pointer* to a dynamically allocated `Delaunay_triangulation` for a 
 * convex polytope, sample uniformly from different subsets of the polytope: 
 *
 * - If `codim == 0`, then sample from the *interior* of the polytope (the 
 *   union of its full-dimensional faces). 
 * - If `codim > 0`, then sample from the subset of the *boundary* of the 
 *   polytope formed by the union of the faces of the given codimension on 
 *   the polytope's boundary. 
 *
 * Note that the sampled points are returned as doubles.  
 *
 * @param tri      Pointer to dynamically allocated input triangulation.
 * @param npoints  Number of points to sample from the polytope.
 * @param codim    Codimension of the simplices to sample from. 
 * @param rng      Reference to existing random number generator instance.
 * @returns        Delaunay triangulation parsed from the given file and the 
 *                 matrix of sampled points. 
 */
template <int CayleyMengerPrecision, int SamplePrecision = CayleyMengerPrecision> 
MatrixXd sampleFromConvexPolytope(Delaunay_triangulation* tri, const int npoints,
                                  const int codim, boost::random::mt19937& rng)
{
    // Obtain the desired subset of simplices in the triangulation  
    std::vector<Simplex> faces; 
    if (codim == 0) 
        faces = getFullDimFaces(tri); 
    else if (codim == 1) 
        faces = getBoundaryFacets(tri);
    else 
        faces = getBoundaryFaces(tri, codim);

    // Compute the (scaled) volume of each face 
    int dim = tri->current_dimension();
    Matrix<number<mpfr_float_backend<CayleyMengerPrecision> >, Dynamic, 1> volumes(faces.size()); 
    for (int i = 0; i < faces.size(); ++i) 
        volumes(i) = faces[i].sqrtAbsCayleyMenger<CayleyMengerPrecision>();

    // Instantiate a categorical distribution with probabilities 
    // proportional to the *boundary* simplex volumes
    //
    // The volumes are normalized into probabilities using double
    // arithmetic  
    VectorXd probs = (volumes / volumes.sum()).template cast<double>();
    std::vector<double> probs_vec; 
    for (int i = 0; i < probs.size(); ++i)
        probs_vec.push_back(probs(i));
    boost::random::discrete_distribution<> dist(probs_vec);

    // Sample from the desired subset of simplices
    MatrixXd sample(npoints, dim); 
    for (int i = 0; i < npoints; ++i)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Get the corresponding simplex
        sample.row(i) = faces[j].sample<SamplePrecision>(1, rng).template cast<double>();
    }
    
    return sample;
}

/**
 * Given a .vert file specifying a convex polytope in terms of its vertices,
 * parse the vertices and sample uniformly from different subsets of the
 * polytope: 
 *
 * - If `codim == 0`, then sample from the *interior* of the polytope (the 
 *   union of its full-dimensional faces). 
 * - If `codim > 0`, then sample from the subset of the *boundary* of the 
 *   polytope formed by the union of the faces of the given codimension on 
 *   the polytope's boundary. 
 *
 * Note that the sampled points are returned as doubles.  
 *
 * @param filename Path to input .vert file of polytope vertices. 
 * @param npoints  Number of points to sample from the polytope.
 * @param codim    Codimension of the simplices to sample from. 
 * @param rng      Reference to existing random number generator instance.
 * @returns        Delaunay triangulation parsed from the given file and the 
 *                 matrix of sampled points. 
 */
template <int CayleyMengerPrecision, int SamplePrecision = CayleyMengerPrecision> 
MatrixXd sampleFromConvexPolytope(std::string filename, const int npoints,
                                  const int codim, boost::random::mt19937& rng)
{
    // Parse the given .vert file and obtain the Delaunay triangulation of 
    // the stored convex polytope 
    Delaunay_triangulation tri = parseVerticesFile(filename);

    return sampleFromConvexPolytope<CayleyMengerPrecision, SamplePrecision>(tri, npoints, codim, rng); 
}

}   // namespace Polytopes

#endif
