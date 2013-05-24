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
    typedef DistHMat2d<Scalar> DistHMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const int multType = Input("--multType","multiply type",2);
        const bool structure = Input("--structure","print structure?",true);
        const int oversample = Input("--oversample","number of extra basis vecs",4);
        ProcessInput();
        PrintInputReport();

        // Allow for extra basis vectors to be used
        SetOversample( oversample );

        // Set up two random distributed H-matrices
        if( commRank == 0 )
        {
            std::cout << "Creating random distributed H-matrices for "
                      <<  "performance testing...";
            std::cout.flush();
        }
        const double createStartTime = mpi::Time();
        DistHMat::Teams teams( mpi::COMM_WORLD );
        DistHMat A( numLevels, maxRank, strong, xSize, ySize, teams );
        DistHMat B( numLevels, maxRank, strong, xSize, ySize, teams );
        A.SetToRandom();
        B.SetToRandom();
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
        if( structure )
        {
#ifdef HAVE_QT5
            std::ostringstream os;
            os << "C on " << commRank;
            C.DisplayLocal( os.str() );
#endif
            C.LatexLocalStructure("C_ghosted_structure");
            C.MScriptLocalStructure("C_ghosted_structure");
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

