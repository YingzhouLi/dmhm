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
    std::cout << "Invert <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <r> <print?> <print structure?>" 
              << std::endl;
}

const double omega = 10.0;
const double C = 1.5*(2*M_PI);

template<typename Real>
std::complex<Real>
sInv( int k, int b, int size )
{
    if( (k+1)<b )
    {
        const Real delta = b-(k+1);
        const Real h = 1. / (size+1);
        const Real realPart = 1;
        const Real imagPart = ::C*delta*delta/(b*b*b*h*::omega);
        return std::complex<Real>(realPart,imagPart);
    }
    else if( k>(size-b) )
    {
        const Real delta = k-(size-b);
        const Real h = 1. / (size+1);
        const Real realPart = 1;
        const Real imagPart = ::C*delta*delta/(b*b*b*h*::omega);
        return std::complex<Real>(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
void
FormRow
( Real imagShift, 
  int x, int y, int z, int xSize, int ySize, int zSize, int pmlSize,
  std::vector< std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;
    const Real hx = 1.0 / (xSize+1);
    const Real hy = 1.0 / (ySize+1);
    const Real hz = 1.0 / (zSize+1);

    const Scalar s1InvL = sInv<Real>( x-1, pmlSize, xSize );
    const Scalar s1InvM = sInv<Real>( x,   pmlSize, xSize );
    const Scalar s1InvR = sInv<Real>( x+1, pmlSize, xSize );
    const Scalar s2InvL = sInv<Real>( y-1, pmlSize, ySize );
    const Scalar s2InvM = sInv<Real>( y,   pmlSize, ySize );
    const Scalar s2InvR = sInv<Real>( y+1, pmlSize, ySize );
    const Scalar s3InvL = sInv<Real>( z-1, pmlSize, ySize );
    const Scalar s3InvM = sInv<Real>( z,   pmlSize, ySize );
    const Scalar s3InvR = sInv<Real>( z+1, pmlSize, ySize );

    // Compute all of the x-shifted terms
    const Scalar xTempL = s2InvM*s3InvM/s1InvL;
    const Scalar xTempM = s2InvM*s3InvM/s1InvM;
    const Scalar xTempR = s2InvM*s3InvM/s1InvR;
    const Scalar xTermL = (xTempL+xTempM)/(2*hx*hx);
    const Scalar xTermR = (xTempR+xTempM)/(2*hx*hx);

    // Compute all of the y-shifted terms
    const Scalar yTempL = s1InvM*s3InvM/s2InvL;
    const Scalar yTempM = s1InvM*s3InvM/s2InvM;
    const Scalar yTempR = s1InvM*s3InvM/s2InvR;
    const Scalar yTermL = (yTempL+yTempM)/(2*hy*hy);
    const Scalar yTermR = (yTempR+yTempM)/(2*hy*hy);

    // Compute all of the z-shifted terms
    const Scalar zTempL = s1InvM*s2InvM/s3InvL;
    const Scalar zTempM = s1InvM*s2InvM/s3InvM;
    const Scalar zTempR = s1InvM*s2InvM/s3InvR;
    const Scalar zTermL = (zTempL+zTempM)/(2*hz*hz);
    const Scalar zTermR = (zTempR+zTempM)/(2*hz*hz);

    // Compute the center term
    const Scalar alpha = 1;
    const Scalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR) +
        (::omega*alpha)*(::omega*alpha)*s1InvM*s2InvM*s3InvM + 
        std::complex<Real>(0,imagShift);

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( centerTerm );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.push_back( xTermL );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.push_back( xTermR );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) + xSize*ySize*z);
        row.push_back( yTermL );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) + xSize*ySize*z);
        row.push_back( yTermR );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z-1));
        row.push_back( zTermL );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z+1));
        row.push_back( zTermR );
    }
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

    // TODO: Use Choice for better command-line argument processing
    if( argc < 9 )
    {
        if( rank == 0 )
            Usage();
        Finalize();
        return 0;
    }
    int arg=1;
    const int xSize = atoi( argv[arg++] );
    const int ySize = atoi( argv[arg++] );
    const int zSize = atoi( argv[arg++] );
    const int numLevels = atoi( argv[arg++] );
    const bool stronglyAdmissible = atoi( argv[arg++] );
    const int r = atoi( argv[arg++] );
    const bool print = atoi( argv[arg++] );
    const bool printStructure = atoi( argv[arg++] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    if( rank == 0 )
    {
        std::cout << "------------------------------------------\n"
                  << "Testing complex double HMat3d inversion   \n"
                  << "--------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef HMat3d<Scalar> HMat;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;
        const int pmlSize = 5;
        const double imagShift = 1.0;

        std::vector<int> map;
        HMat::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( rank == 0 )
        {
            std::cout << "Filling sparse matrix...";
            std::cout.flush();
        }
        double fillStartTime = mpi::Time();
        std::vector<Scalar> row;
        std::vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;
            const int z = (iNatural/(xSize*ySize));

            FormRow
            ( imagShift, x, y, z, 
              xSize, ySize, zSize, pmlSize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
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
        HMat H( S, numLevels, r, stronglyAdmissible, xSize, ySize, zSize );
        double constructStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print( "H" );
            if( printStructure )
            {
                H.LatexWriteStructure("structure");
                H.MScriptWriteStructure("structure");
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
        for( int numIterations=10; numIterations<60; numIterations+=10 )
        {
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
