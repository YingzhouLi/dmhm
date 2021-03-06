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
    Initialize( argc, argv );
    try
    {
        const int r = Input("--r","width of stacked matrices",10);
        const int s = Input("--s","height of top matrix",100);
        const int t = Input("--t","height of bottom matrix",100);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        if( s > r || t > r )
            throw std::logic_error("s and t cannot be greater than r");

        // Fill a packed version of two concatenated upper triangular s x r
        // and t x r matrices with Gaussian random variables.
        const int minDimT = std::min(s,r);
        const int minDimB = std::min(t,r);
        const int packedSize =
            (minDimT*minDimT+minDimT)/2 + (r-minDimT)*minDimT +
            (minDimB*minDimB+minDimB)/2 + (r-minDimB)*minDimB;

        // Form the random packed A
        std::vector<std::complex<double> > packedA( packedSize );
        for( int k=0; k<packedSize; ++k )
            SerialGaussianRandomVariable( packedA[k] );

        // Expand the packed A
        Dense<std::complex<double> > A( s+t, r );
        hmat_tools::Scale( std::complex<double>(0), A );
        {
            int k=0;
            for( int j=0; j<r; ++j )
            {
                for( int i=0; i<std::min(j+1,s); ++i )
                    A.Set( i, j, packedA[k++] );
                for( int i=s; i<std::min(j+s+1,s+t); ++i )
                    A.Set( i, j, packedA[k++] );
            }
        }
        if( print )
        {
            hmat_tools::PrintPacked( "packedA:", r, s, t, &packedA[0] );
            std::cout << std::endl;
        }

        // Allocate a workspace and perform the packed QR
        std::vector<std::complex<double> >
            tau( std::min(s+t,r) ), work(std::max(s+t,r));
        hmat_tools::PackedQR( r, s, t, &packedA[0], &tau[0], &work[0] );
        if( print )
        {
            hmat_tools::PrintPacked( "packedQR:", r, s, t, &packedA[0] );
            std::cout << "\ntau:\n";
            for( unsigned j=0; j<tau.size(); ++j )
                std::cout << WrapScalar(tau[j]) << "\n";
            std::cout << std::endl;
        }

        // Copy the R into a zeroed (s+t) x r matrix
        Dense<std::complex<double> > B( s+t, r );
        hmat_tools::Scale( std::complex<double>(0), B );
        int k=0;
        for( int j=0; j<r; ++j )
        {
            const int S = std::min(j+1,s);
            const int T = std::min(j+1,t);
            const int U = std::min(j+1,S+T);

            for( int i=0; i<U; ++i )
                B.Set( i, j, packedA[k++] );

            k += (S+T) - U;
        }
        if( print )
        {
            B.Print( "R" );
            std::cout << std::endl;
        }

        hmat_tools::ApplyPackedQFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<r; ++j )
                for( int i=0; i<s+t; ++i )
                    maxError =
                        std::max(maxError,Abs(B.Get(i,j)-A.Get(i,j)));
            std::cout << "||QR-A||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "QR ~= A" );
            std::cout << std::endl;
        }

        // Create an (s+t) x (s+t) identity matrix and then apply Q' Q from
        // the left
        B.Resize( s+t, s+t );
        work.resize( s+t );
        hmat_tools::Scale( std::complex<double>(0), B );
        for( int j=0; j<s+t; ++j )
            B.Set( j, j, 1.0 );
        hmat_tools::ApplyPackedQFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        hmat_tools::ApplyPackedQAdjointFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<s+t; ++j )
            {
                for( int i=0; i<s+t; ++i )
                {
                    const std::complex<double> computed = B.Get(i,j);
                    if( i == j )
                        maxError = std::max(maxError,Abs(computed-1.0));
                    else
                        maxError = std::max(maxError,Abs(computed));
                }
            }
            std::cout << "||I - Q'QI||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "Q' Q I" );
            std::cout << std::endl;
        }

        // Create an (s+t) x (s+t) identity matrix and then apply Q' Q from
        // the right
        hmat_tools::Scale( std::complex<double>(0), B );
        for( int j=0; j<s+t; ++j )
            B.Set( j, j, 1.0 );
        hmat_tools::ApplyPackedQAdjointFromRight
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        hmat_tools::ApplyPackedQFromRight
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<s+t; ++j )
            {
                for( int i=0; i<s+t; ++i )
                {
                    const std::complex<double> computed = B.Get(i,j);
                    if( i == j )
                        maxError = std::max(maxError,Abs(computed-1.0));
                    else
                        maxError = std::max(maxError,Abs(computed));
                }
            }
            std::cout << "||I - IQ'Q||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "I Q' Q" );
            std::cout << std::endl;
        }
    }
    catch( ArgException& e ) { }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    Finalize();
    return 0;
}
