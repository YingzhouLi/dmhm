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
( int x, int y, int z, int xSize, int ySize, int zSize, 
  Vector<std::complex<Real> >& row, Vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;

    row.Resize( 0 );
    colIndices.Resize( 0 );

    // Set up the diagonal entry
    colIndices.Push_back( rowIdx );
    row.Push_back( Scalar(8) );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.Push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.Push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.Push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.Push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.Push_back( x + xSize*(y-1) + xSize*ySize*z );
        row.Push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.Push_back( x + xSize*(y+1) + xSize*ySize*z );
        row.Push_back( (Scalar)-1 );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.Push_back( x + xSize*y + xSize*ySize*(z-1) );
        row.Push_back( (Scalar)-1 );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.Push_back( x + xSize*y + xSize*ySize*(z+1) );
        row.Push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat3d<Scalar> HMat;
    typedef DistHMat3d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",15);
        const int ySize = Input("--ySize","size of y dimension",15);
        const int zSize = Input("--zSize","size of z dimension",15);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const int multType = Input("--multType","multiply type",2);
        const bool print = Input("--print","print matrices?",false);
        const bool structure = Input("--structure","print structure?",false);
        const bool multI = Input("--multI","multiply by identity?",false);
        const int schuN = Input("--schuN","Iteration number of Schulz invert",15);
        const int oversample = Input("--oversample","num extra basis vecs",4);
        const double midcomputeTol = 
            Input("--midcomputeTol","tolerance for midcompute stage",1e-16);
        const double compressionTol =
            Input("--compressionTol","tolerance for compression",1e-16);
        ProcessInput();
        PrintInputReport();

        SetOversample( oversample );
        SetMidcomputeTolerance<double>( midcomputeTol );
        SetCompressionTolerance<double>( compressionTol );

        const int m = xSize*ySize*zSize;
        const int n = xSize*ySize*zSize;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        Vector<int> map;
        HMat::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        Vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( commRank == 0 )
        {
            std::cout << "Filling sparse matrices...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double fillStartTime = mpi::Time();
        Vector<Scalar> row;
        Vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;
            const int z = iNatural/(xSize*ySize);

            FormRow( x, y, z, xSize, ySize, zSize, row, colIndices );

            for( unsigned j=0; j<row.Size(); ++j )
            {
                S.nonzeros.Push_back( row[j] );
                S.columnIndices.Push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.Push_back( S.nonzeros.Size() );
        mpi::Barrier( mpi::COMM_WORLD );
        double fillStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
        }

        // Convert to H-matrix form
        DistHMat::Teams teams( mpi::COMM_WORLD );
        if( commRank == 0 )
        {
            std::cout << "Constructing H-matrices in dist...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double constructStartTime = mpi::Time();
        DistHMat A( S, numLevels, maxRank, strong, xSize, ySize, zSize, teams );
        DistHMat B( S, numLevels, maxRank, strong, xSize, ySize, zSize, teams );
        mpi::Barrier( mpi::COMM_WORLD );
        double constructStopTime = mpi::Time();
        if( commRank == 0 )
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
        
        //Clear all sparse part
        S.Clear();
        map.Clear();
        inverseMap.Clear();
        row.Clear();
        colIndices.Clear();

        if( structure )
        {
#ifdef HAVE_QT5
            std::ostringstream os;
            os << "A on " << commRank;
            A.DisplayLocal( os.str() );
#endif
            A.LatexLocalStructure("A_structure");
            A.MScriptLocalStructure("A_structure");
        }

        const int localHeight = A.LocalHeight();
        const int localWidth = A.LocalWidth();
        if( localHeight != localWidth )
            throw std::logic_error("A was not locally square");


        // SchulzInvert
        if( commRank == 0 )
        {
            std::cout << "SchulzInvert...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double SchulzInvertStartTime = mpi::Time();
        A.SchulzInvert(schuN);
        mpi::Barrier( mpi::COMM_WORLD );
        double SchulzInvertStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << SchulzInvertStopTime-SchulzInvertStartTime
                      << " seconds." << std::endl;
        }

        Dense<Scalar> XLocal;
        if( multI )
        {
            const int firstLocalRow = A.FirstLocalRow();
            XLocal.Resize( localHeight, n );
            hmat_tools::Scale( Scalar(0), XLocal );
            for( int j=firstLocalRow; j<firstLocalRow+localHeight; ++j )
                XLocal.Set( j-firstLocalRow, j, Scalar(1) );
        }
        else
        {
            const int numRhs = 30;
            XLocal.Resize( localHeight, numRhs );
            ParallelGaussianRandomVectors( XLocal );
        }
        
        Dense<Scalar> YLocal, ZLocal;
        // Y := AZ := ABX
        B.Multiply( Scalar(1), XLocal, ZLocal );
        if( print )
        {
            std::ostringstream sE;
            sE << "BLocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "BLocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", EFile );
            EFile << "];\n";
        }
        A.Multiply( Scalar(1), XLocal, ZLocal );
        if( print )
        {
            std::ostringstream sE;
            sE << "ALocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "ALocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", EFile );
            EFile << "];\n";
        }
        // Attempt to multiply the two matrices
        if( commRank == 0 )
        {
            std::cout << "Multiplying distributed H-matrices...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double multStartTime = mpi::Time();
        DistHMat C( teams );
        A.Multiply( Scalar(1), B, C, multType );
        mpi::Barrier( mpi::COMM_WORLD );
        double multStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << multStopTime-multStartTime
                      << " seconds." << std::endl;
        }
#ifdef MEMORY_INFO
        C.PrintMemoryInfo("Memory info of C");
        std::cout << "Memory of block now: " 
                  << MemoryUsage()/1024./1024. << "MB" << std::endl;
        std::cout << "Peak memory of block: "
                  << PeakMemoryUsage()/1024./1024. << "MB" << std::endl;
#endif
        if( structure )
        {
#ifdef HAVE_QT5
            std::ostringstream os;
            os << "C on " << commRank;
            C.DisplayLocal( os.str() );
#endif
            C.LatexLocalStructure("C_ghosted_structure");
            C.MScriptLocalStructure("C_ghosted_structure");
        }

        // Check that CX = ABX for an arbitrary X
        if( commRank == 0 )
            std::cout << "Checking consistency: " << std::endl;
        mpi::Barrier( mpi::COMM_WORLD );
        B.Multiply( Scalar(1), XLocal, ZLocal );
        A.Multiply( Scalar(1), ZLocal, YLocal );
        // Z := CX
        C.Multiply( Scalar(1), XLocal, ZLocal );

        if( print )
        {
            std::ostringstream sY, sZ;
            sY << "YLocal_" << commRank << ".m";
            sZ << "ZLocal_" << commRank << ".m";
            std::ofstream YFile( sY.str().c_str() );
            std::ofstream ZFile( sZ.str().c_str() );

            YFile << "YLocal{" << commRank+1 << "}=[\n";
            YLocal.Print( "", YFile );
            YFile << "];\n";

            ZFile << "ZLocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", ZFile );
            ZFile << "];\n";
        }

        // Compute the error norms and put ZLocal = YLocal-ZLocal
        double myInfTruth=0, myInfError=0, 
               myL1Truth=0, myL1Error=0, 
               myL2SquaredTruth=0, myL2SquaredError=0;
        for( int j=0; j<XLocal.Width(); ++j )
        {
            for( int i=0; i<localHeight; ++i )
            {
                const std::complex<double> truth = YLocal.Get(i,j);
                const std::complex<double> error = truth - ZLocal.Get(i,j);
                const double truthMag = Abs( truth );
                const double errorMag = Abs( error );
                ZLocal.Set( i, j, error );

                // RHS norms
                myInfTruth = std::max(myInfTruth,truthMag);
                myL1Truth += truthMag;
                myL2SquaredTruth += truthMag*truthMag;
                // Error norms
                myInfError = std::max(myInfError,errorMag);
                myL1Error += errorMag;
                myL2SquaredError += errorMag*errorMag;
            }
        }
        double infTruth, infError, 
               L1Truth, L1Error, 
               L2SquaredTruth, L2SquaredError;
        mpi::Reduce( &myInfTruth, &infTruth, 1, mpi::MAX, 0, mpi::COMM_WORLD );
        mpi::Reduce( &myInfError, &infError, 1, mpi::MAX, 0, mpi::COMM_WORLD );
        mpi::Reduce( &myL1Truth, &L1Truth, 1, mpi::SUM, 0, mpi::COMM_WORLD );
        mpi::Reduce( &myL1Error, &L1Error, 1, mpi::SUM, 0, mpi::COMM_WORLD );
        mpi::Reduce
        ( &myL2SquaredTruth, &L2SquaredTruth, 1, mpi::SUM, 0, mpi::COMM_WORLD );
        mpi::Reduce
        ( &myL2SquaredError, &L2SquaredError, 1, mpi::SUM, 0, mpi::COMM_WORLD );
        if( commRank == 0 )
        {
            std::cout << "||ABX||_oo    = " << infTruth << "\n"
                      << "||ABX||_1     = " << L1Truth << "\n"
                      << "||ABX||_2     = " << sqrt(L2SquaredTruth) << "\n"
                      << "||CX-ABX||_oo = " << infError << "\n"
                      << "||CX-ABX||_1  = " << L1Error << "\n"
                      << "||CX-ABX||_2  = " << sqrt(L2SquaredError) 
                      << std::endl;
        }

        if( print )
        {
            std::ostringstream sE;
            sE << "ELocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "ELocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", EFile );
            EFile << "];\n";
        }
    }
    catch( ArgException& e ) { }
    catch( std::exception& e )
    {
        std::cerr << "Process " << commRank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }
    
    Finalize();
    return 0;
}

