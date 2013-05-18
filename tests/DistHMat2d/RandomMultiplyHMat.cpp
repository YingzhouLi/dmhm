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
    std::cout << "RandomMultiplyHMat <xSize> <ySize> <numLevels> "
                 "<strongly admissible?> <maxRank> <multType> <print structure?>"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );

    // TODO: Use Choice for better command-line argument processing
    if( argc < 8 )
    {
        if( commRank == 0 )
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
    const int multType = atoi( argv[arg++] );
    const bool printStructure = atoi( argv[arg++] );

    if( commRank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing performance of H-matrix/H-matrix mult on    \n"
                  << "random matrices\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef DistHMat2d<Scalar> DistHMat;

        // Set up two random distributed H-matrices
        if( commRank == 0 )
        {
            std::cout << "Creating random distributed H-matrices for "
                      <<  "performance testing...";
            std::cout.flush();
        }
        const double createStartTime = mpi::Time();
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat A
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, teams );
        DistHMat B
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, teams );
        A.SetToRandom();
        B.SetToRandom();
        const double createStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << createStopTime-createStartTime
                      << " seconds." << std::endl;
        }

        if( printStructure )
        {
            A.LatexWriteLocalStructure("A_structure");
            A.MScriptWriteLocalStructure("A_structure");
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
        A.Multiply( (Scalar)1, B, C, multType );
        mpi::Barrier( mpi::COMM_WORLD );
        double multStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << multStopTime-multStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            C.LatexWriteLocalStructure("C_ghosted_structure");
            C.MScriptWriteLocalStructure("C_ghosted_structure");
        }
    }
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

