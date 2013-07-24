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
( int x, int y, int xSize, int ySize, 
  Vector<std::complex<Real> >& row, Vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y;

    row.Resize( 0 );
    colIndices.Resize( 0 );

    // Set up the diagonal entry
    colIndices.Push_back( rowIdx );
    row.Push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.Push_back( (x-1) + xSize*y );
        row.Push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.Push_back( (x+1) + xSize*y );
        row.Push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.Push_back( x + xSize*(y-1) );
        row.Push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.Push_back( x + xSize*(y+1) );
        row.Push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat2d<Scalar> HMat;
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
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const bool print = Input("--print","print matrices?",false);
        const bool structure = Input("--structure","print structure?",true);
        const bool multI = Input("--multI","multiply identity?",false);
        const int oversample = Input
            ("--oversample","number of extra basis vecs",4);
        ProcessInput();
        PrintInputReport();

        SetOversample( oversample );

        const int m = xSize*ySize;
        const int n = xSize*ySize;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        Vector<int> map;
        HMat::BuildNaturalToHierarchicalMap( map, xSize, ySize, numLevels );

        Vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        std::cout << "Filling sparse matrices...";
        std::cout.flush();
        double fillStartTime = mpi::Time();
        Vector<Scalar> row;
        Vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;

            FormRow( x, y, xSize, ySize, row, colIndices );

            for( unsigned j=0; j<row.Size(); ++j )
            {
                S.nonzeros.Push_back( row[j] );
                S.columnIndices.Push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.Push_back( S.nonzeros.Size() );
        double fillStopTime = mpi::Time();
        std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                  << std::endl;

        // Convert to H-matrix form
        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        double constructStartTime = mpi::Time();
        HMat A( S, numLevels, maxRank, strong, xSize, ySize );
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
        if( print )
        {
            A.Print("A");
            B.Print("B");
        }
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
        if( print )
            C.Print("C");
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
        if( print )
            X.Print("X");
        
        Dense<Scalar> Y, Z;
        // Y := AZ := ABX
        B.Multiply( (Scalar)1, X, Z );
        A.Multiply( (Scalar)1, Z, Y );
        if( print )
        {
            Z.Print("Z := B X");
            Y.Print("Y := A Z = A B X");
        }
        // Z := CX
        C.Multiply( (Scalar)1, X, Z );
        if( print )
            Z.Print("Z := C X");

        if( print )
        {
            std::ofstream YFile( "Y.m" );
            std::ofstream ZFile( "Z.m" );

            YFile << "YMat=[\n";
            Y.Print( "", YFile );
            YFile << "];\n";

            ZFile << "ZMat=[\n";
            Z.Print( "", ZFile );
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
            Z.Print( "", EFile );
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

