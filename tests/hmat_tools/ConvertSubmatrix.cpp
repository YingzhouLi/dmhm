/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
using namespace dmhm;

int
main( int argc, char* argv[] )
{
    const int xSize = 4;
    const int ySize = 4;
    const int zSize = 2;
    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Converting double-precision sparse to dense         \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        Sparse<double> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );

            if( i >= xSize )
            {
                S.nonzeros.Push_back( S.nonzeros.Size()+1 );
                S.columnIndices.Push_back( i-xSize );
            }

            if( i >= 1 )
            {
                S.nonzeros.Push_back( S.nonzeros.Size()+1 );
                S.columnIndices.Push_back( i-1 );
            }

            S.nonzeros.Push_back( S.nonzeros.Size()+1 );    
            S.columnIndices.Push_back( i );

            if( i+1 < n )
            {
                S.nonzeros.Push_back( S.nonzeros.Size()+1 );
                S.columnIndices.Push_back( i+1 );
            }

            if( i+xSize < n )
            {
                S.nonzeros.Push_back( S.nonzeros.Size()+1 );
                S.columnIndices.Push_back( i+xSize );
            }
        }
        S.rowOffsets.Push_back( S.nonzeros.Size() );
        S.Print( "S" );

        Dense<double> D;
        hmat_tools::ConvertSubmatrix( D, S, 0, m, 0, n );
        D.Print( "D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }
    
    std::cout << "----------------------------------------------------\n"
              << "Converting double-precision sparse to low-rank      \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        Sparse<double> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        for( int i=0; i<r; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );

            if( i+xSize < n )
            {
                S.nonzeros.Push_back( S.nonzeros.Size()+1 );    
                S.columnIndices.Push_back( i+xSize );
            }
        }
        for( int i=r; i<m; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );
        }
        S.rowOffsets.Push_back( S.nonzeros.Size() );
        S.Print( "S" );

        LowRank<double> F;
        hmat_tools::ConvertSubmatrix( F, S, 0, m, 0, n );
        F.Print( "F" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    return 0;
}
