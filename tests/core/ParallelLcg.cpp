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
    int rank = mpi::CommRank( mpi::COMM_WORLD );
    int commSize = mpi::CommSize( mpi::COMM_WORLD );

    // Print the first 3*commSize entries of the serial RNG and the first
    // 3 entries from each process from the parallel RNG.
    try
    {
        if( rank == 0 )
        {
            std::cout << "Serial values:" << std::endl;
            for( int i=0; i<3*commSize; ++i )
            {
                UInt64 state = SerialLcg();
                std::cout << state[0] << " " << state[1] << "\n";
            }
            std::cout << std::endl;
        }

        std::vector<UInt32> myValues( 6 );
        std::vector<UInt32> values( 6*commSize );
        for( int i=0; i<3; ++i )
        {
            UInt64 state = ParallelLcg();
            myValues[2*i] = state[0];
            myValues[2*i+1] = state[1];
        }
        // We're treating the unsigned data as signed, but since we are only
        // gathering the bits, it doesn't matter
        mpi::AllGather
        ( (int*)&myValues[0], 6, (int*)&values[0], 6, mpi::COMM_WORLD );
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
    } // TODO: Command-line argument processing
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
