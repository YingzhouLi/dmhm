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
    const int xSize = 4;
    const int ySize = 4;
    const int zSize = 2;
    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Converting double-precision low-rank to Quasi2dHMat \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        dmhm::LowRank<double> F;
        F.U.Resize( m, r );
        F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                F.U.Set( i, j, (double)i+j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                F.V.Set( i, j, (double)i-j );
        F.Print( "F" );

        dmhm::Quasi2dHMat<double> 
            H( F, 2, r, false, xSize, ySize, zSize );

        dmhm::Vector<double> x( n );
        double* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = 1.0;
        x.Print( "x" );

        dmhm::Vector<double> y;
        H.Multiply( 2.0, x, y );
        y.Print( "y := 2 H x ~= 2 F x" );
        H.TransposeMultiply( 2.0, x, y );
        y.Print( "y := 2 H^T x ~= 2 F^T x" );
        H.AdjointMultiply( 2.0, x, y );
        y.Print( "y := 2 H^H x ~= 2 F^H x" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }
    
    std::cout << "-----------------------------------------------\n"
              << "Converting double-complex sparse to Quasi2dHMat\n"
              << "------------------------------------------------" 
              << std::endl;
    try
    {
        dmhm::LowRank<std::complex<double> > F;
        F.U.Resize( m, r );
        F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                F.U.Set( i, j, std::complex<double>(i,j) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                F.V.Set( i, j, std::complex<double>(i+j,i-j) );
        F.Print( "F" );

        dmhm::Quasi2dHMat<std::complex<double> > 
            H( F, 2, r, false, xSize, ySize, zSize );

        dmhm::Vector< std::complex<double> > x( n );
        std::complex<double>* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = std::complex<double>(1.0,3.0);
        x.Print( "x" );

        dmhm::Vector< std::complex<double> > y;
        H.Multiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H x ~= (4+5i)F x" );
        H.TransposeMultiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^T x ~= (4+5i)F^T x" );
        H.AdjointMultiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^H x ~= (4+5i)F^H x" );
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
