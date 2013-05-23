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
    typedef std::complex<double> Scalar;
    typedef DistHMat3d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","x dimension size",15);
        const int ySize = Input("--ySize","y dimension size",15);
		const int zSize = Input("--zSize","z dimension size",15);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of blocks",5);
        const bool structure = Input("--structure","print structure?",false);
        ProcessInput();
        PrintInputReport();

        // Create a random distributed H-matrix
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat H
        ( numLevels, maxRank, strong, xSize, ySize, zSize, teams );
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
        if( structure )
        {
#ifdef HAVE_QT5
            std::ostringstream os;
            os << "Ghosted H on " << rank;
            H.DisplayLocal( os.str() );
#endif
            H.LatexLocalStructure("H_ghosted_structure");
            H.MScriptLocalStructure("H_ghosted_structure");
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
