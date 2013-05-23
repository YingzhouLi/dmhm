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
    const int rank = mpi::CommRank( mpi::COMM_WORLD );
    const int p = mpi::CommSize( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat3d<Scalar> HMat;
    typedef DistHMat3d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",15);
        const int ySize = Input("--ySize","size of y dimension",15);
		const int zSize = Input("--zSize","size of z dimension",15);
        const int numLevels = Input("--numLevels","depth of H-matrix",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const bool print = Input("--print","print matrices?",false);
        const bool structure = Input("--structure","print structure?",true);
        ProcessInput();
        PrintInputReport();

    const int n = xSize*ySize*zSize;
    const bool symmetric = false;

        // Build a random H-matrix
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrices...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double constructStartTime = mpi::Time();
        HMat H
        ( numLevels, maxRank, symmetric, strong, xSize, ySize, zSize );
        throw std::logic_error("Constructor needs to be fixed...");
        H.SetToRandom();
        mpi::Barrier( mpi::COMM_WORLD );
        double constructStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
            if( structure )
            {
#ifdef HAVE_QT5
                H.Display("H");
#endif
                H.LatexStructure("H_serial_structure");
                H.MScriptStructure("H_serial_structure");
            }
        }

        // Store the result of a serial hmat-mat
        if( rank == 0 )
        {
            std::cout << "Y := H X...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double hmatMatStartTime = mpi::Time();
        Dense<Scalar> X( n, 30 );
        for( int j=0; j<X.Width(); ++j )
            for( int i=0; i<n; ++i )
                X.Set( i, j, i+j );
        Dense<Scalar> Y;
        H.Multiply( (Scalar)1, X, Y );
        mpi::Barrier( mpi::COMM_WORLD );
        double hmatMatStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << hmatMatStopTime-hmatMatStartTime 
                      << " seconds." << std::endl;
        }
        
        // Store the result of a serial hmat-trans-mat
        if( rank == 0 )
        {
            std::cout << "Z := H' X...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double hmatAdjointMatStartTime = mpi::Time();
        Dense<Scalar> Z;
        H.AdjointMultiply( (Scalar)1, X, Z );
        mpi::Barrier( mpi::COMM_WORLD );
        double hmatAdjointMatStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " 
                      << hmatAdjointMatStopTime-hmatAdjointMatStartTime 
                      << " seconds." << std::endl;
        }

        // Set up our subcommunicators and compute the packed sizes
        DistHMat::Teams teams( mpi::COMM_WORLD );
        std::vector<std::size_t> packedSizes;
        DistHMat::PackedSizes( packedSizes, H, teams ); 
        const std::size_t myMaxSize = 
            *(std::max_element( packedSizes.begin(), packedSizes.end() ));

        // Pack for a DistHMatHMat
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix for distribution...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double packStartTime = mpi::Time();
        std::vector<byte> sendBuffer( p*myMaxSize );
        std::vector<byte*> packedPieces( p );
        for( int i=0; i<p; ++i )
            packedPieces[i] = &sendBuffer[i*myMaxSize];
        DistHMat::Pack( packedPieces, H, teams );
        mpi::Barrier( mpi::COMM_WORLD );
        double packStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << packStopTime-packStartTime << " seconds."
                      << std::endl;
        }

        // Compute the maximum package size
        int myIntMaxSize, intMaxSize;
        {
            myIntMaxSize = myMaxSize;
            mpi::AllReduce
            ( &myIntMaxSize, &intMaxSize, 1, mpi::MAX, mpi::COMM_WORLD );
        }
        if( rank == 0 )
        {
            std::cout << "Maximum per-process message size: " 
                      << ((double)intMaxSize)/(1024.*1024.) << " MB." 
                      << std::endl;
        }
 
        // AllToAll
        if( rank == 0 )
        {
            std::cout << "AllToAll redistribution...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double allToAllStartTime = mpi::Time();
        std::vector<byte> recvBuffer( p*intMaxSize );
        mpi::AllToAll
        ( &sendBuffer[0], myIntMaxSize, &recvBuffer[0], intMaxSize,
          mpi::COMM_WORLD );
        mpi::Barrier( mpi::COMM_WORLD );
        double allToAllStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << allToAllStopTime-allToAllStartTime
                      << " seconds." << std::endl;
        }

        // Unpack our part of the matrix defined by process 0
        if( rank == 0 )
        {
            std::cout << "Unpacking...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double unpackStartTime = mpi::Time();
        DistHMat distH( &recvBuffer[0], teams );
        mpi::Barrier( mpi::COMM_WORLD );
        double unpackStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
        }
        if( structure )
        {
#ifdef HAVE_QT5
            std::ostringstream os;
            os << "distH on " << rank;
            distH.DisplayLocal( os.str() );
#endif
            distH.LatexLocalStructure("distH_structure");
            distH.MScriptLocalStructure("distH_structure");
        }

        // Apply the distributed H-matrix
        if( rank == 0 )
        {
            std::cout << "Distributed Y := H X...";
            std::cout.flush();
        }
        Dense<Scalar> XLocal;
        XLocal.LockedView
        ( X, distH.FirstLocalCol(), 0, distH.LocalWidth(), X.Width() );
        mpi::Barrier( mpi::COMM_WORLD );
        double distHmatMatStartTime = mpi::Time();
        Dense<Scalar> YLocal;
        distH.Multiply( (Scalar)1, XLocal, YLocal );
        mpi::Barrier( mpi::COMM_WORLD );
        double distHmatMatStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << distHmatMatStopTime-distHmatMatStartTime
                      << " seconds." << std::endl;
        }

        // Measure how close our result is to the serial results
        if( rank == 0 )
        {
            std::cout << "Comparing serial and distributed results...";
            std::cout.flush();
        }
        Dense<Scalar> YLocalTruth;
        YLocalTruth.View
        ( Y, distH.FirstLocalRow(), 0, distH.LocalHeight(), X.Width() );
        for( int j=0; j<YLocal.Width(); ++j )
        {
            for( int i=0; i<YLocal.Height(); ++i )
            {
                double error = std::abs(YLocalTruth.Get(i,j)-YLocal.Get(i,j));
                if( error > 1e-8 )
                {
                    std::ostringstream ss;
                    ss << "Answer differed at local index (" 
                        << i << "," << j << "), truth was "
                       << YLocalTruth.Get(i,j) << ", computed was "
                       << YLocal.Get(i,j) << std::endl;
                    throw std::logic_error( ss.str().c_str() );
                }
                YLocal.Set(i,j,error);
            }
        }
        mpi::Barrier( mpi::COMM_WORLD );
        if( rank == 0 )
            std::cout << "done" << std::endl;
 
        // Apply the adjoint of distributed H-matrix
        if( rank == 0 )
        {
            std::cout << "Distributed Z := H' X...";
            std::cout.flush();
        }
        XLocal.LockedView
        ( X, distH.FirstLocalRow(), 0, distH.LocalHeight(), X.Width() );
        mpi::Barrier( mpi::COMM_WORLD );
        double distHmatAdjointMatStartTime = mpi::Time();
        Dense<Scalar> ZLocal;
        distH.AdjointMultiply( (Scalar)1, XLocal, ZLocal );
        mpi::Barrier( mpi::COMM_WORLD );
        double distHmatAdjointMatStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " 
                      << distHmatAdjointMatStopTime-
                         distHmatAdjointMatStartTime
                      << " seconds." << std::endl;
        }

        // Measure how close our result is to the serial results
        if( rank == 0 )
        {
            std::cout << "Comparing serial and distributed results...";
            std::cout.flush();
        }
        Dense<Scalar> ZLocalTruth;
        ZLocalTruth.View
        ( Z, distH.FirstLocalCol(), 0, distH.LocalWidth(), X.Width() );
        for( int j=0; j<ZLocal.Width(); ++j )
        {
            for( int i=0; i<ZLocal.Height(); ++i )
            {
                double error = std::abs(ZLocalTruth.Get(i,j)-ZLocal.Get(i,j));
                if( error > 1e-8 )
                {
                    std::ostringstream ss;
                    ss << "Answer differed at local index (" 
                        << i << "," << j << "), truth was "
                       << ZLocalTruth.Get(i,j) << ", computed was "
                       << ZLocal.Get(i,j) << std::endl;
                    throw std::logic_error( ss.str().c_str() );
                }
                ZLocal.Set(i,j,error);
            }
        }
        mpi::Barrier( mpi::COMM_WORLD );
        if( rank == 0 )
            std::cout << "done" << std::endl;
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

