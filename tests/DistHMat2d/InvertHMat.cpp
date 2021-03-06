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

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat2d<Scalar> HMat;
    typedef DistHMat2d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const int multType = Input("--multType","multiply type",0);
        const bool print = Input("--print","print matrices?",false);
        const bool printRank =
            Input("--printRank","print rank of low rank blocks?",false);
        const bool structure = Input("--structure","print structure?",false);
        const bool multI = Input("--multI","multiply by identity?",false);
        const int schuN = Input("--schuN","Iteration number of Schulz invert",15);
        const int oversample = Input("--oversample","num extra basis vecs",4);
        const double pseudoinvTol =
            Input("--pseudoinvTol","tolerance for pseudo inverse",1e-16);
        const double compressionTol =
            Input("--compressionTol","tolerance for compression",1e-8);
        ProcessInput();
        PrintInputReport();

        SetOversample( oversample );
        SetPseudoinvTolerance<double>( pseudoinvTol );
        SetCompressionTolerance<double>( compressionTol );

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

        if( commRank == 0 )
        {
            std::cout << "Filling sparse matrices...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double fillStartTime = mpi::Time();
        Vector<Scalar> row;
        Vector<int> colIndices;
        Dense<Scalar> DomainA(xSize+2,ySize+2);
        Dense<Scalar> DomainV(xSize,ySize);
        ParallelGaussianRandomVectors( DomainA );
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

            FormRow( x, y, xSize, ySize, h, DomainA, DomainV, row, colIndices );

            for( int j=0; j<row.Size(); ++j )
            {
                S.nonzeros.PushBack( row[j] );
                S.columnIndices.PushBack( map[colIndices[j]] );
            }
        }
        S.rowOffsets.PushBack( S.nonzeros.Size() );
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
        DistHMat A( S, numLevels, maxRank, strong, xSize, ySize, teams );
        DistHMat B( S, numLevels, maxRank, strong, xSize, ySize, teams );
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
        A.SchulzInvert(schuN, multType, 1.5, 6, 1e-4, printRank);

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
        B.Multiply( Scalar(1), XLocal, YLocal );
        if( print )
        {
            std::ostringstream sE;
            sE << "BLocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "BLocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", EFile );
            EFile << "];\n";
        }
        A.Multiply( Scalar(1), YLocal, ZLocal );
        if( print )
        {
            std::ostringstream sE;
            sE << "ALocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "ALocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", EFile );
            EFile << "];\n";
        }

        if( commRank == 0 )
            std::cout << "Checking consistency: " << std::endl;
        mpi::Barrier( mpi::COMM_WORLD );

        if( print )
        {
            std::ostringstream sX, sZ;
            sX << "XLocal_" << commRank << ".m";
            sZ << "ZLocal_" << commRank << ".m";
            std::ofstream XFile( sX.str().c_str() );
            std::ofstream ZFile( sZ.str().c_str() );

            XFile << "XLocal{" << commRank+1 << "}=[\n";
            XLocal.Print( "", XFile );
            XFile << "];\n";

            ZFile << "ZLocal{" << commRank+1 << "}=[\n";
            ZLocal.Print( "", ZFile );
            ZFile << "];\n";
        }

        // Compute the error norms and put YLocal = XLocal-ZLocal
        double myInfTruth=0, myInfError=0,
               myL1Truth=0, myL1Error=0,
               myL2SquaredTruth=0, myL2SquaredError=0;
        for( int j=0; j<XLocal.Width(); ++j )
        {
            for( int i=0; i<localHeight; ++i )
            {
                const std::complex<double> truth = XLocal.Get(i,j);
                const std::complex<double> error = truth - ZLocal.Get(i,j);
                const double truthMag = Abs( truth );
                const double errorMag = Abs( error );
                YLocal.Set( i, j, error );

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
            std::cout << "||X||_oo    = " << infTruth << "\n"
                      << "||X||_1     = " << L1Truth << "\n"
                      << "||X||_2     = " << sqrt(L2SquaredTruth) << "\n"
                      << "||X-ABX||_oo = " << infError << "\n"
                      << "||X-ABX||_1  = " << L1Error << "\n"
                      << "||X-ABX||_2  = " << sqrt(L2SquaredError)
                      << std::endl;
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

