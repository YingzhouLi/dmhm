/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying, 
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
( int x, int y, int z, int xSize, int ySize, int zSize, 
  std::vector<std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z-1) );
        row.push_back( (Scalar)-1 );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z+1) );
        row.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    typedef std::complex<double> Scalar;
    typedef HMat3d<Scalar> HMat;

    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    if( commSize != 1 )
    {
        if( commRank == 0 )
            std::cerr << "This test must be run with a single MPI process" 
                      << std::endl;
        Finalize();
        return 0;
    }

    try
    {
        const int xSize = Input("--xSize","size of x dimension",15);
        const int ySize = Input("--ySize","size of y dimension",15);
        const int zSize = Input("--zSize","size of z dimension",15);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const bool print = Input("--print","print matrices?",false);
        const bool structure = Input("--structure","print structure?",true);
        const bool multI = Input("--multI","multiply by identity?",false);
        const int oversample = Input("--oversample","number of extra basis vecs",4);
        ProcessInput();
        PrintInputReport();

        SetOversample( oversample );

        const int m = xSize*ySize*zSize;
        const int n = xSize*ySize*zSize;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        std::vector<int> map;
        HMat::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        std::cout << "Filling sparse matrices...";
        std::cout.flush();
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

            FormRow( x, y, z, xSize, ySize, zSize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        double fillStopTime = mpi::Time();
        std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                  << std::endl;
        // Convert to H-matrix form
        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        double constructStartTime = mpi::Time();
        HMat A( S, numLevels, maxRank, strong, xSize, ySize, zSize );
        double constructStopTime = mpi::Time();
        std::cout << "done: " << constructStopTime-constructStartTime 
                  << " seconds." << std::endl;

        // Invert H-matrix and make a copy
        std::cout << "Inverting H-matrix and making copy...";
        std::cout.flush();
        double invertStartTime = mpi::Time();
        A.DirectInvert();
        HMat B;
        B.CopyFrom( A );
        double invertStopTime = mpi::Time();
        std::cout << "done: " << invertStopTime-invertStartTime 
                  << " seconds." << std::endl;
        //if( print )
        //{
        //    A.Print("A");
        //    B.Print("B");
        //}
        if( structure )
        {
#ifdef HAVE_QT5
            A.Display("A");
#endif
            A.LatexStructure("A_structure");
            A.MScriptStructure("A_structure");
        }

        // Attempt to multiply the two matrices
        std::cout << "Multiplying H-matrices...";
        std::cout.flush();
        double multStartTime = mpi::Time();
        HMat C;
        A.Multiply( (Scalar)1, B, C );
        double multStopTime = mpi::Time();
        std::cout << "done: " << multStopTime-multStartTime
                  << " seconds." << std::endl;
        //if( print )
        //    C.Print("C");
        if( structure )
        {
#ifdef HAVE_QT5
            C.Display("C");
#endif
            C.LatexStructure("C_structure");
            C.MScriptStructure("C_structure");
        }

        // Check that CX = ABX for an arbitrary X
        std::cout << "Checking consistency: " << std::endl;
        Dense<Scalar> X;
        if( multI )
        {
            X.Resize( m, n );
            hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<n; ++j )
                X.Set( j, j, (Scalar)1 );
        }
        else
        {
            const int numRhs = 30;
            X.Resize( m, numRhs );
            SerialGaussianRandomVectors( X );
        }
        //if( print )
        //    X.Print("X");
        
        Dense<Scalar> Y, Z;
        // Y := AZ := ABX
        B.Multiply( (Scalar)1, X, Z );
        A.Multiply( (Scalar)1, Z, Y );
        //if( print )
        //{
        //    Z.Print("Z := B X");
        //    Y.Print("Y := A Z = A B X");
        //}
        // Z := CX
        C.Multiply( (Scalar)1, X, Z );
        //if( print )
        //    Z.Print("Z := C X");

        if( print )
        {
            std::ofstream YFile( "Y.m" );
            std::ofstream ZFile( "Z.m" );

            YFile << "YMat=[\n";
            Y.Print( YFile, "" );
            YFile << "];\n";

            ZFile << "ZMat=[\n";
            Z.Print( ZFile, "" );
            ZFile << "];\n";
        }

        // Compute the error norms and put Z := Y - Z
        double infTruth=0, infError=0, 
               L1Truth=0, L1Error=0, 
               L2SquaredTruth=0, L2SquaredError=0;
        for( int j=0; j<X.Width(); ++j )
        {
            for( int i=0; i<m; ++i )
            {
                const std::complex<double> truth = Y.Get(i,j);
                const std::complex<double> error = truth - Z.Get(i,j);
                const double truthMag = Abs( truth );
                const double errorMag = Abs( error );
                Z.Set( i, j, error );

                // RHS norms
                infTruth = std::max(infTruth,truthMag);
                L1Truth += truthMag;
                L2SquaredTruth += truthMag*truthMag;
                // Error norms
                infError = std::max(infError,errorMag);
                L1Error += errorMag;
                L2SquaredError += errorMag*errorMag;
            }
        }
        std::cout << "||ABX||_oo    = " << infTruth << "\n"
                  << "||ABX||_1     = " << L1Truth << "\n"
                  << "||ABX||_2     = " << sqrt(L2SquaredTruth) << "\n"
                  << "||CX-ABX||_oo = " << infError << "\n"
                  << "||CX-ABX||_1  = " << L1Error << "\n"
                  << "||CX-ABX||_2  = " << sqrt(L2SquaredError) 
                  << std::endl;

        if( print )
        {
            std::ofstream EFile( "E.m" );
            EFile << "EMat=[\n"; 
            Z.Print( EFile, "" );
            EFile << "];\n";
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
