/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

void Usage()
{
    std::cout << "Pack <xSize> <ySize> <numLevels> "
                 "<strongly admissible?> <maxRank> <print?> <print structure?>" 
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int rank = dmhm::mpi::CommRank( MPI_COMM_WORLD );
    const int p = dmhm::mpi::CommSize( MPI_COMM_WORLD );

    if( argc < 8 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    int arg=1;
    const int xSize = atoi( argv[arg++] );
    const int ySize = atoi( argv[arg++] );
    const int numLevels = atoi( argv[arg++] );
    const bool stronglyAdmissible = atoi( argv[arg++] );
    const int maxRank = atoi( argv[arg++] );
    const bool print = atoi( argv[arg++] );
    const bool printStructure = atoi( argv[arg++] );

    const int n = xSize*ySize;
    const bool symmetric = false;

    if( rank == 0 )
    {
        std::cout << "-------------------------------------------------\n"
                  << "Testing complex double HMatHMat packing/unpacking\n"
                  << "into DistHMat2d                                  \n"
                  << "-------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef dmhm::HMat2d<Scalar> HMat;
        typedef dmhm::DistHMat2d<Scalar> DistHMat;

        // Build a random H-matrix
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrices...";
            std::cout.flush();
        }
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double constructStartTime = dmhm::mpi::WallTime();
        HMat H
        ( numLevels, maxRank, symmetric, stronglyAdmissible, xSize, ySize );
        throw std::logic_error("Constructor needs to be fixed...");
        H.SetToRandom();
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double constructStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
            if( printStructure )
            {
                H.LatexWriteStructure("H_serial_structure");
                H.MScriptWriteStructure("H_serial_structure");
            }
        }

        // Store the result of a serial hmat-mat
        if( rank == 0 )
        {
            std::cout << "Y := H X...";
            std::cout.flush();
        }
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double hmatMatStartTime = dmhm::mpi::WallTime();
        dmhm::Dense<Scalar> X( n, 30 );
        for( int j=0; j<X.Width(); ++j )
            for( int i=0; i<n; ++i )
                X.Set( i, j, i+j );
        dmhm::Dense<Scalar> Y;
        H.Multiply( (Scalar)1, X, Y );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double hmatMatStopTime = dmhm::mpi::WallTime();
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double hmatAdjointMatStartTime = dmhm::mpi::WallTime();
        dmhm::Dense<Scalar> Z;
        H.AdjointMultiply( (Scalar)1, X, Z );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double hmatAdjointMatStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " 
                      << hmatAdjointMatStopTime-hmatAdjointMatStartTime 
                      << " seconds." << std::endl;
        }

        // Set up our subcommunicators and compute the packed sizes
        DistHMat::Teams teams( MPI_COMM_WORLD );
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double packStartTime = dmhm::mpi::WallTime();
        std::vector<dmhm::byte> sendBuffer( p*myMaxSize );
        std::vector<dmhm::byte*> packedPieces( p );
        for( int i=0; i<p; ++i )
            packedPieces[i] = &sendBuffer[i*myMaxSize];
        DistHMat::Pack( packedPieces, H, teams );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double packStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << packStopTime-packStartTime << " seconds."
                      << std::endl;
        }

        // Compute the maximum package size
        int myIntMaxSize, intMaxSize;
        {
            myIntMaxSize = myMaxSize;
            dmhm::mpi::AllReduce
            ( &myIntMaxSize, &intMaxSize, 1, MPI_MAX, MPI_COMM_WORLD );
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStartTime = dmhm::mpi::WallTime();
        std::vector<dmhm::byte> recvBuffer( p*intMaxSize );
        dmhm::mpi::AllToAll
        ( &sendBuffer[0], myIntMaxSize, &recvBuffer[0], intMaxSize,
          MPI_COMM_WORLD );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStopTime = dmhm::mpi::WallTime();
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStartTime = dmhm::mpi::WallTime();
        DistHMat distH( &recvBuffer[0], teams );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            distH.LatexWriteLocalStructure("distH_structure");
            distH.MScriptWriteLocalStructure("distH_structure");
        }

        // Apply the distributed H-matrix
        if( rank == 0 )
        {
            std::cout << "Distributed Y := H X...";
            std::cout.flush();
        }
        dmhm::Dense<Scalar> XLocal;
        XLocal.LockedView
        ( X, distH.FirstLocalCol(), 0, distH.LocalWidth(), X.Width() );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatMatStartTime = dmhm::mpi::WallTime();
        dmhm::Dense<Scalar> YLocal;
        distH.Multiply( (Scalar)1, XLocal, YLocal );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatMatStopTime = dmhm::mpi::WallTime();
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
        dmhm::Dense<Scalar> YLocalTruth;
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatAdjointMatStartTime = dmhm::mpi::WallTime();
        dmhm::Dense<Scalar> ZLocal;
        distH.AdjointMultiply( (Scalar)1, XLocal, ZLocal );
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatAdjointMatStopTime = dmhm::mpi::WallTime();
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
        dmhm::Dense<Scalar> ZLocalTruth;
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        if( rank == 0 )
            std::cout << "done" << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

