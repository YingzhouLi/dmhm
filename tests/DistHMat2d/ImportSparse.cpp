/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
using namespace dmhm;

template<typename Real>
void
FormCol
( int x, int y, int xSize, int ySize,
  Vector<std::complex<Real> >& col, Vector<int>& rowIndices )
{
    typedef std::complex<Real> Scalar;
    const int colIdx = x + xSize*y;

    col.Resize( 0 );
    rowIndices.Resize( 0 );

    // Set up the diagonal entry
    rowIndices.PushBack( colIdx );
    col.PushBack( (Scalar)4 );

    // Front connection to (x-1,y)
    if( x != 0 )
    {
        rowIndices.PushBack( (x-1) + xSize*y );
        col.PushBack( (Scalar)-1 );
    }

    // Back connection to (x+1,y)
    if( x != xSize-1 )
    {
        rowIndices.PushBack( (x+1) + xSize*y );
        col.PushBack( (Scalar)-1 );
    }

    // Left connection to (x,y-1)
    if( y != 0 )
    {
        rowIndices.PushBack( x + xSize*(y-1) );
        col.PushBack( (Scalar)-1 );
    }

    // Right connection to (x,y+1)
    if( y != ySize-1 )
    {
        rowIndices.PushBack( x + xSize*(y+1) );
        col.PushBack( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int rank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef DistHMat2d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        ProcessInput();
        PrintInputReport();

        // Build a non-initialized H-matrix tree
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat H( numLevels, maxRank, strong, xSize, ySize, teams );

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
    catch( ArgException& e ) { }
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
