/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

int
main( int argc, char* argv[] )
{
    const int m = 8;
    const int n = 8;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision Compress                   \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        dmhm::Dense<double> D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, (double)i+j );
        D.Print( "D" );

        dmhm::LowRank<double> F;
        dmhm::hmat_tools::Compress( r, D, F );

        F.Print( "F.U F.V^T ~= D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex double-precision Compress           \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        dmhm::Dense< std::complex<double> > D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, std::complex<double>(i+j,i-j) );
        D.Print( "D" );

        // F = F.U F.V^T
        dmhm::LowRank<std::complex<double> > FFalse;
        dmhm::Dense< std::complex<double> > DCopy;
        dmhm::hmat_tools::Copy( D, DCopy );
        dmhm::hmat_tools::Compress( r, DCopy, FFalse );
        FFalse.Print( "F.U F.V^T ~= D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }

    return 0;
}
