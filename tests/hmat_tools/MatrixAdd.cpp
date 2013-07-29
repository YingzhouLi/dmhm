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
    const int m = 8;
    const int n = 8;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision dense Add                  \n"
              << "----------------------------------------------------"
              << std::endl;
    try
    {
        Dense<double> A( m, n );
        Dense<double> B( m, n );
        Dense<double> C;

        // Set A to all 1's
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                A.Set( i, j, 1.0 );
        A.Print( "A" );

        // Set B to all 2's
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                B.Set( i, j, 2.0 );
        B.Print( "B" );

        // C := 3 A + 5 B
        hmat_tools::Add( 3.0, A, 5.0, B, C );
        C.Print( "C := 3A + 5B" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex single-precision dense Add          \n"
              << "----------------------------------------------------"
              << std::endl;
    try
    {
        Dense< std::complex<float> > A( m, n );
        Dense< std::complex<float> > B( m, n );
        Dense< std::complex<float> > C;

        // Set each entry of A to (1 + 2i)
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                A.Set( i, j, std::complex<float>(1,2) );
        A.Print( "A" );

        // Set each entry of B to (3 + 4i)
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                B.Set( i, j, std::complex<float>(3,4) );
        B.Print( "B" );

        // C := (5 + 6i)A + (7 + 8i)B
        hmat_tools::Add
        ( std::complex<float>(5,6), A, std::complex<float>(7,8), B, C );
        C.Print( "C := (5+6i)A + (7+8i)B" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision low-rank Add               \n"
              << "----------------------------------------------------"
              << std::endl;
    try
    {
        LowRank<double> A, B, C;

        A.U.Resize( m, r );
        A.V.Resize( n, r );
        B.U.Resize( m, r );
        B.V.Resize( n, r );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                A.U.Set( i, j, (double)j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                A.V.Set( i, j, (double)i+j );
        A.Print( "A" );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                B.U.Set( i, j, (double)-j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                B.V.Set( i, j, (double)-(i+j) );
        B.Print( "B" );

        // C := 3A + 5B
        hmat_tools::Add( 3.0, A, 5.0, B, C );
        C.Print( "C := 3A + 5B" );
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
