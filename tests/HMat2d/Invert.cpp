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
FormRow
( int x, int y, int xSize, int ySize, double h,
 Dense<std::complex<Real> >& DA, Dense<std::complex<Real> >& DV,
 Vector<std::complex<Real> >& row, Vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y;
    double hh = h*h;

    row.Resize( 0 );
    colIndices.Resize( 0 );

    Scalar cv = DV.Get(x,y);

    // Front connection to (x-1,y)
    if( x != 0 )
    {
        colIndices.PushBack( (x-1) + xSize*y );
        Scalar coef = (DA.Get(x,y+1) + DA.Get(x+1,y+1)) / hh / 2.0;
        row.PushBack( -coef );
    }
    cv += (DA.Get(x,y+1) + DA.Get(x+1,y+1)) / hh / 2.0;

    // Back connection to (x+1,y)
    if( x != xSize-1 )
    {
        colIndices.PushBack( (x+1) + xSize*y );
        Scalar coef = (DA.Get(x+2,y+1) + DA.Get(x+1,y+1)) / hh / 2.0;
        row.PushBack( -coef );
    }
    cv += (DA.Get(x+2,y+1) + DA.Get(x+1,y+1)) / hh / 2.0;

    // Left connection to (x,y-1)
    if( y != 0 )
    {
        colIndices.PushBack( x + xSize*(y-1) );
        Scalar coef = (DA.Get(x+1,y) + DA.Get(x+1,y+1)) / hh / 2.0;
        row.PushBack( -coef );
    }
    cv += (DA.Get(x+1,y) + DA.Get(x+1,y+1)) / hh / 2.0;

    // Right connection to (x,y+1)
    if( y != ySize-1 )
    {
        colIndices.PushBack( x + xSize*(y+1) );
        Scalar coef = (DA.Get(x+1,y+2) + DA.Get(x+1,y+1)) / hh / 2.0;
        row.PushBack( -coef );
    }
    cv += (DA.Get(x+1,y+2) + DA.Get(x+1,y+1)) / hh / 2.0;

    // Set up the diagonal entry
    colIndices.PushBack( rowIdx );
    row.PushBack( (Scalar)cv );
}

template<typename Real>
void
CheckDistanceFromOnes( const Vector<std::complex<Real> >& z )
{
    typedef std::complex<Real> Scalar;

    const int m = z.Height();

    const double xNormL1 = m;
    const double xNormL2 = sqrt( m );
    const double xNormLInf = 1.0;
    double errorNormL1, errorNormL2, errorNormLInf;
    {
        errorNormL1 = 0;
        errorNormLInf = 0;

        const Scalar* zBuffer = z.LockedBuffer();
        double errorNormL2Squared = 0;
        for( int i=0; i<m; ++i )
        {
            const double deviation = std::abs( zBuffer[i] - 1.0 );
            errorNormL1 += deviation;
            errorNormL2Squared += deviation*deviation;
            errorNormLInf = std::max( errorNormLInf, deviation );
        }
        errorNormL2 = std::sqrt( errorNormL2Squared );
    }
    std::cout << "||e||_1 =  " << errorNormL1 << "\n"
              << "||e||_2 =  " << errorNormL2 << "\n"
              << "||e||_oo = " << errorNormLInf << "\n"
              << "\n"
              << "||e||_1  / ||x||_1  = " << errorNormL1/xNormL1 << "\n"
              << "||e||_2  / ||x||_2  = " << errorNormL2/xNormL2 << "\n"
              << "||e||_oo / ||x||_oo = " << errorNormLInf/xNormLInf << "\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int rank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat2d<Scalar> HMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const bool print = Input("--print","print matrices?",false);
        const bool structure = Input("--structure","print structure?",false);
        const int oversample = Input("--oversample","number of extra basis vecs",4);
        ProcessInput();
        PrintInputReport();

        SetOversample( oversample );

        const int m = xSize*ySize;
        const int n = xSize*ySize;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;
        const int pmlSize = 5;
        const double imagShift = 1.0;

        Vector<int> map;
        HMat::BuildNaturalToHierarchicalMap( map, xSize, ySize, numLevels );

        Vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( rank == 0 )
        {
            std::cout << "Filling sparse matrix...";
            std::cout.flush();
        }
        double fillStartTime = mpi::Time();
        Vector<Scalar> row;
        Vector<int> colIndices;
        Dense<Scalar> DomainA(xSize+2,ySize+2);
        Dense<Scalar> DomainV(xSize,ySize);
        SerialGaussianRandomVectors( DomainA );
        double h = 1.0/xSize;
        for( int x=0; x<xSize+2; ++x )
            for( int y=0; y<ySize+2; ++y )
                DomainA.Set(x,y,Abs(DomainA.Get(x,y))+Scalar(0.001));

        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.PushBack( S.nonzeros.Size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;

            FormRow
            ( x, y, xSize, ySize, h, DomainA, DomainV, row, colIndices );

            for( int j=0; j<row.Size(); ++j )
            {
                S.nonzeros.PushBack( row[j] );
                S.columnIndices.PushBack( map[colIndices[j]] );
            }
        }
        S.rowOffsets.PushBack( S.nonzeros.Size() );
        double fillStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds."
                      << std::endl;
            if( print )
                S.Print( "S" );
        }

        // Convert to H-matrix form
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrix...";
            std::cout.flush();
        }
        double constructStartTime = mpi::Time();
        HMat H( S, numLevels, maxRank, strong, xSize, ySize );
        double constructStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime
                      << " seconds." << std::endl;
            if( print )
                H.Print( "H" );
            if( structure )
            {
#ifdef HAVE_QT5
                H.Display("structure");
#endif
                H.LatexStructure("structure");
                H.MScriptStructure("structure");
            }
        }

        // Test against a vector of all 1's
        Vector<Scalar> x;
        x.Resize( m );
        Scalar* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector of all ones...";
            std::cout.flush();
        }
        Vector<Scalar> y;
        double matVecStartTime = mpi::Time();
        H.Multiply( 1.0, x, y );
        double matVecStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime
                      << " seconds." << std::endl;
            if( print )
                y.Print( "y := H x ~= S x" );
        }

        // Direct inversion test
        {
            // Make a copy for inversion
            if( rank == 0 )
            {
                std::cout << "Making a copy of the H-matrix for inversion...";
                std::cout.flush();
            }
            double copyStartTime = mpi::Time();
            HMat invH;
            invH.CopyFrom( H );
            double copyStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << copyStopTime-copyStartTime
                          << " seconds." << std::endl;
            }

            // Perform a direct inversion
            if( rank == 0 )
            {
                std::cout << "Directly inverting the H-matrix...";
                std::cout.flush();
            }
            double invertStartTime = mpi::Time();
            invH.DirectInvert();
            double invertStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << invertStopTime-invertStartTime
                          << " seconds." << std::endl;
                if( print )
                    invH.Print( "inv(H)" );
            }

            // Test for discrepancies in x and inv(H) H x
            if( rank == 0 )
            {
                std::cout << "Multiplying the direct inverse by a vector...";
                std::cout.flush();
            }
            matVecStartTime = mpi::Time();
            Vector<Scalar> z;
            invH.Multiply( 1.0, y, z );
            matVecStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << matVecStopTime-matVecStartTime
                          << " seconds." << std::endl;
                if( print )
                {
                    y.Print( "y := H x ~= S x" );
                    z.Print( "z := inv(H) H x ~= x" );
                }
                CheckDistanceFromOnes( z );
            }
        }

        // Schulz iteration tests
        {
            int numIterations=-1;
            // Make a copy
            if( rank == 0 )
            {
                std::cout << "Making a copy for Schulz inversion...";
                std::cout.flush();
            }
            double copyStartTime = mpi::Time();
            HMat invH;
            invH.CopyFrom( H );
            double copyStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << copyStopTime-copyStartTime
                          << " seconds." << std::endl;
            }

            // Perform the iterative inversion
            if( rank == 0 )
            {
                std::cout << "Performing " << numIterations
                          << " Schulz iterations...";
                std::cout.flush();
            }
            double invertStartTime = mpi::Time();
            invH.SchulzInvert( numIterations );
            double invertStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << invertStopTime-invertStartTime
                          << " seconds." << std::endl;
                if( print )
                    invH.Print( "inv(H)" );
            }

            // Test for discrepancies in x and inv(H) H x
            if( rank == 0 )
            {
                std::cout << "Multiplying the Schulz inverse by a vector...";
                std::cout.flush();
            }
            matVecStartTime = mpi::Time();
            Vector<Scalar> z;
            invH.Multiply( 1.0, y, z );
            matVecStopTime = mpi::Time();
            if( rank == 0 )
            {
                std::cout << "done: " << matVecStopTime-matVecStartTime
                          << " seconds." << std::endl;
                if( print )
                {
                    y.Print( "y := H x ~= S x" );
                    z.Print( "z := inv(H) H x ~= x" );
                }
                CheckDistanceFromOnes( z );
            }
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
