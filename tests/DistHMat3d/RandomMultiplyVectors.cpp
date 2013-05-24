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
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef DistHMat3d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",15);
        const int ySize = Input("--ySize","size of y dimension",15);
        const int zSize = Input("--zSize","size of z dimension",15);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const int numVectors = Input("--numVectors","num vectors to mult",20);
        const bool structure = Input("--structure","print structure?",true);
        ProcessInput();
        PrintInputReport();

        // Set up a random H-matrix
        if( commRank == 0 )
        {
            std::cout << "Creating random distributed H-matrix for performance "
                      << "testing...";
            std::cout.flush();
        }
        const double createStartTime = mpi::Time();
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat A
        ( numLevels, maxRank, strong, xSize, ySize, zSize, teams );
        A.SetToRandom();
        const double createStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << createStopTime-createStartTime
                      << " seconds." << std::endl;
        }

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

        // Generate random vectors
        const int localWidth = A.LocalWidth();
        Dense<Scalar> X( localWidth, numVectors );
        ParallelGaussianRandomVectors( X );

        // Multiply against random vectors
        Dense<Scalar> Y;
        if( commRank == 0 )
        {
            std::cout << "Multiplying distributed H-matrix against vectors...";
            std::cout.flush();
        }
        mpi::Barrier( mpi::COMM_WORLD );
        double multStartTime = mpi::Time();
        A.Multiply( (Scalar)1, X, Y );
        mpi::Barrier( mpi::COMM_WORLD );
        double multStopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "done: " << multStopTime-multStartTime
                      << " seconds." << std::endl;
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

