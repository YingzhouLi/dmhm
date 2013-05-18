/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
using namespace dmhm;

void Usage()
{
    std::cout << "ImportSparse <xSize> <ySize> <numLevels> "
              << "<strongly admissible?> <r> <print?> <print structure?>"
              << std::endl;
}

template<typename Real>
void
FormCol
( int x, int y, int xSize, int ySize, 
  std::vector<std::complex<Real> >& col, std::vector<int>& rowIndices )
{
    typedef std::complex<Real> Scalar;
    const int colIdx = x + xSize*y;

    col.resize( 0 );
    rowIndices.resize( 0 );

    // Set up the diagonal entry
    rowIndices.push_back( colIdx );
    col.push_back( (Scalar)8 );

    // Front connection to (x-1,y)
    if( x != 0 )
    {
        rowIndices.push_back( (x-1) + xSize*y );
        col.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y)
    if( x != xSize-1 )
    {
        rowIndices.push_back( (x+1) + xSize*y );
        col.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1)
    if( y != 0 )
    {
        rowIndices.push_back( x + xSize*(y-1) );
        col.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1)
    if( y != ySize-1 )
    {
        rowIndices.push_back( x + xSize*(y+1) );
        col.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int rank = mpi::CommRank( mpi::COMM_WORLD );

    // TODO: Use Choice for better command-line argument processing
    if( argc < 8 )
    {
        if( rank == 0 )
            Usage();
        Finalize();
        return 0;
    }
    int arg=1;
    const int xSize = atoi( argv[arg++] );
    const int ySize = atoi( argv[arg++] );
    const int numLevels = atoi( argv[arg++] );
    const bool stronglyAdmissible = atoi( argv[arg++] );
    const int maxRank = atoi( argv[arg++] );
    //const bool print = atoi( argv[arg++] );
    //const bool printStructure = atoi( argv[arg++] );

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
        typedef DistHMat2d<Scalar> DistHMat;

        // Build a non-initialized H-matrix tree
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat H
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, teams );

        // Grab out our local geometric target info
        const int firstLocalX = H.FirstLocalXTarget();
        const int firstLocalY = H.FirstLocalYTarget();
        const int localXSize = H.LocalXTargetSize();
        const int localYSize = H.LocalYTargetSize();

        if( rank == 0 )
        {
            std::cout << "firstLocalX = " << firstLocalX << "\n"
                      << "firstLocalY = " << firstLocalY << "\n"
                      << "localXSize  = " << localXSize << "\n"
                      << "localYSize  = " << localYSize << std::endl;

            std::cout << "This routine is not yet finished." << std::endl;
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }
    
    Finalize();
    return 0;
}

