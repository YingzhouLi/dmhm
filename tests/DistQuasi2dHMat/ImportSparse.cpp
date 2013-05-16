/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

void Usage()
{
    std::cout << "ImportSparse <xSize> <ySize> <zSize> <numLevels> "
              << "<strongly admissible?> <r> <print?> <print structure?>"
              << std::endl;
}

template<typename Real>
void
FormCol
( int x, int y, int z, int xSize, int ySize, int zSize, 
  std::vector< std::complex<Real> >& col, std::vector<int>& rowIndices )
{
    typedef std::complex<Real> Scalar;
    const int colIdx = x + xSize*y + xSize*ySize*z;

    col.resize( 0 );
    rowIndices.resize( 0 );

    // Set up the diagonal entry
    rowIndices.push_back( colIdx );
    col.push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        rowIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        col.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        rowIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        col.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        rowIndices.push_back( x + xSize*(y-1) + xSize*ySize*z );
        col.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        rowIndices.push_back( x + xSize*(y+1) + xSize*ySize*z );
        col.push_back( (Scalar)-1 );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        rowIndices.push_back( x + xSize*y + xSize*ySize*(z-1) );
        col.push_back( (Scalar)-1 );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        rowIndices.push_back( x + xSize*y + xSize*ySize*(z+1) );
        col.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int rank = dmhm::mpi::CommRank( MPI_COMM_WORLD );

    if( argc < 9 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int maxRank = atoi( argv[6] );
    //const bool print = atoi( argv[7] );
    //const bool printStructure = atoi( argv[8] );

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing import of distributed sparse matrix         \n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef dmhm::DistQuasi2dHMat<Scalar> DistQuasi2d;

        // Build a non-initialized H-matrix tree
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        DistQuasi2d H
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize, teams );

        // Grab out our local geometric target info
        const int firstLocalX = H.FirstLocalXTarget();
        const int firstLocalY = H.FirstLocalYTarget();
        const int firstLocalZ = H.FirstLocalZTarget();
        const int localXSize = H.LocalXTargetSize();
        const int localYSize = H.LocalYTargetSize();
        const int localZSize = H.LocalZTargetSize();

        if( rank == 0 )
        {
            std::cout << "firstLocalX = " << firstLocalX << "\n"
                      << "firstLocalY = " << firstLocalY << "\n"
                      << "firstLocalZ = " << firstLocalZ << "\n"
                      << "localXSize  = " << localXSize << "\n"
                      << "localYSize  = " << localYSize << "\n"
                      << "localZSize  = " << localZSize << std::endl;

            std::cout << "This routine is not yet finished." << std::endl;
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

