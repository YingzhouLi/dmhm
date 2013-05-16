/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_SPARSE_MATRIX_HPP
#define DMHM_SPARSE_MATRIX_HPP 1

namespace dmhm {

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
struct Sparse
{
    bool symmetric;
    int height, width;
    std::vector<Scalar> nonzeros;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;

    void Print( const std::string tag ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline void
Sparse<Scalar>::Print( const std::string tag ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Print");
#endif
    if( symmetric )
        std::cout << tag << "(symmetric)\n";
    else
        std::cout << tag << "\n";

    for( int i=0; i<height; ++i )
    {
        const int numCols = rowOffsets[i+1]-rowOffsets[i];
        const int rowOffset = rowOffsets[i];
        for( int k=0; k<numCols; ++k )
        {
            const int j = columnIndices[rowOffset+k];
            const Scalar alpha = nonzeros[rowOffset+k];
            std::cout << i << " " << j << " " << WrapScalar(alpha) << "\n";
        }
    }
    std::cout << std::endl;
}

} // namespace dmhm

#endif // ifndef DMHM_SPARSE_MATRIX_HPP
