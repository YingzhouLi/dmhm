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
    std::cout << "Ghost <xSize> <ySize> <numLevels> "
                 "<strongly admissible?> <maxRank> <print structure?>" 
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int rank = mpi::CommRank( mpi::COMM_WORLD );

    // TODO: Use Choice for command-line argument processing
    if( argc < 8 )
    {
        if( rank == 0 )
            Usage();
        Finalize();
        return 0;
    }
    int arg=1;
    const int xSize = atoi( argv[arg++] );
    const int ySize = atoi( argv[arg++] );
    const int numLevels = atoi( argv[arg++] );
    const bool stronglyAdmissible = atoi( argv[arg++] );
    const int maxRank = atoi( argv[arg++] );
    const bool printStructure = atoi( argv[arg++] );

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing formation of ghost nodes\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef DistHMat2d<Scalar> DistHMat;

        // Create a random distributed H-matrix
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat H
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, teams );
        H.SetToRandom();

        // Form the ghost nodes
        if( rank == 0 )
        {
            std::cout << "Forming ghost nodes...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double ghostStartTime = mpi::Time();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        mpi::Barrier( mpi::COMM_WORLD );
        double ghostStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << ghostStopTime-ghostStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            H.LatexWriteLocalStructure("H_ghosted_structure");
            H.MScriptWriteLocalStructure("H_ghosted_structure");
        }

        // Form the ghost nodes again
        if( rank == 0 )
        {
            std::cout << "Forming ghost nodes a second time...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double ghostStartTime2 = mpi::Time();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        mpi::Barrier( mpi::COMM_WORLD );
        double ghostStopTime2 = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << ghostStopTime2-ghostStartTime2
                      << " seconds." << std::endl;
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

