/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    int rank = dmhm::mpi::CommRank( MPI_COMM_WORLD );
    int commSize = dmhm::mpi::CommSize( MPI_COMM_WORLD );

    // Print the first 3*commSize entries of the serial RNG and the first 
    // 3 entries from each process from the parallel RNG.
    try
    {
        dmhm::UInt64 seed = {{ 17U, 0U }};
        if( rank == 0 )
        {
            dmhm::SeedSerialLcg( seed );
            std::cout << "Serial values:" << std::endl;
            for( int i=0; i<3*commSize; ++i )
            {
                dmhm::UInt64 state = dmhm::SerialLcg();
                std::cout << state[0] << " " << state[1] << "\n";
            }
            std::cout << std::endl;
        }

        std::vector<dmhm::UInt32> myValues( 6 );
        std::vector<dmhm::UInt32> values( 6*commSize );
        dmhm::SeedParallelLcg( rank, commSize, seed );
        for( int i=0; i<3; ++i )
        {
            dmhm::UInt64 state = dmhm::ParallelLcg();
            myValues[2*i] = state[0];
            myValues[2*i+1] = state[1];
        }
        dmhm::mpi::AllGather( &myValues[0], 6, &values[0], 6, MPI_COMM_WORLD );
        if( rank == 0 )
        {
            std::cout << "Parallel values:" << std::endl;
            for( int i=0; i<3; ++i )
            {
                for( int j=0; j<commSize; ++j )
                {
                    const int k = i+3*j;
                    std::cout << values[2*k] << " " << values[2*k+1] << "\n";
                }
            }
            std::cout << std::endl;
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }

    MPI_Finalize();
    return 0;
}
