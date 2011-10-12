/*
   Distributed-Memory Hierarchical Matrices (DMHM): a prototype implementation
   of distributed-memory H-matrix arithmetic. 

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

        dmhm::LowRank<double,false> F;
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
        dmhm::LowRank<std::complex<double>,false> FFalse;
        dmhm::Dense< std::complex<double> > DCopy;
        dmhm::hmat_tools::Copy( D, DCopy );
        dmhm::hmat_tools::Compress( r, DCopy, FFalse );
        FFalse.Print( "F.U F.V^T ~= D" );

        // F = F.U F.V^H
        dmhm::LowRank<std::complex<double>,true> FTrue;
        dmhm::hmat_tools::Copy( D, DCopy );
        dmhm::hmat_tools::Compress( r, DCopy, FTrue );
        FTrue.Print( "F.U F.V^H ~= D" );
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
