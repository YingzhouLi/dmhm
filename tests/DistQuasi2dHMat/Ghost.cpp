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
    std::cout << "Ghost <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <maxRank> <print structure?>" 
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int rank = dmhm::mpi::CommRank( MPI_COMM_WORLD );

    if( argc < 8 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int maxRank = atoi( argv[6] );
    const bool printStructure = atoi( argv[7] );

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
        typedef dmhm::DistQuasi2dHMat<Scalar> DistQuasi2d;

        // Create a random distributed H-matrix
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        DistQuasi2d H
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize, teams );
        H.SetToRandom();

        // Form the ghost nodes
        if( rank == 0 )
        {
            std::cout << "Forming ghost nodes...";
            std::cout.flush();
        }
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStartTime = dmhm::mpi::WallTime();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStopTime = dmhm::mpi::WallTime();
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
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStartTime2 = dmhm::mpi::WallTime();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        dmhm::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStopTime2 = dmhm::mpi::WallTime();
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
        dmhm::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

