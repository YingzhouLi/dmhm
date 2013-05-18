/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
#include <unistd.h>
using namespace dmhm;

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv ); 
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    if( commSize != 1 )
    {
        if( commRank == 0 )
            std::cerr << "This test should be run with a single MPI process" 
                      << std::endl;
        MPI_Finalize();
        return 0;
    }

    try
    {
        std::cout << "Running timing experiment, please wait a few seconds..."
                  << std::endl;
        Timer timer;
        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;

        std::cout << "Repeating experiment without resetting." << std::endl;

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;
        
        std::cout << "Repeating experiment after clearing timer 0." 
                  << std::endl;
        timer.Clear( 0 );

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;

        std::cout << "Repeating experiment after clearing all timers." 
                  << std::endl;
        timer.Clear();

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;
    }
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
