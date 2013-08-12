/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

#include "./MultiplyHMatFormGhostRanks-incl.hpp"
#include "./MultiplyHMatParallelQR-incl.hpp"

#include "./MultiplyHMatMain-incl.hpp"
#include "./MultiplyHMatFHH-incl.hpp"
#include "./Truncation-incl.hpp"
#include "./MultiplyHMatCompress-incl.hpp"
//#include "./MultiplyHMatRandomCompress-incl.hpp"
#include "./MultiplyHMatFHHCompress-incl.hpp"

namespace dmhm {

// C := alpha A B
template<typename Scalar>
void
DistHMat2d<Scalar>::Multiply
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C,
  int multType )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Multiply");
    if( multType < 0 || multType > 2 )
        throw std::logic_error("Invalid multiplication type");
#endif
    DistHMat2d<Scalar>& A = *this;

#ifdef MEMORY_INFO
    ResetMemoryCount();
#endif
    if( multType == 0 )
        A.MultiplyHMatSingleUpdateAccumulate( alpha, B, C );
    else if( multType == 1 )
        A.MultiplyHMatSingleLevelAccumulate( alpha, B, C );
    else
        A.MultiplyHMatFullAccumulate( alpha, B, C );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFullAccumulate
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFullAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( numLevels_ != B.numLevels_ )
        throw std::logic_error("H-matrices must have same number of levels");
    if( level_ != B.level_ )
        throw std::logic_error("Mismatched levels");
#endif
    DistHMat2d<Scalar>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer;
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 1 );
#endif

#ifdef MEMORY_INFO
    A.PrintGlobalMemoryInfo("Matrix A before loop: ");
    B.PrintGlobalMemoryInfo("Matrix B before loop: ");
#endif

    const int startLevel = 0;
    const int endLevel = A.NumLevels();

    const int startUpdate = 0;
    const int endUpdate = 4;

#ifdef TIME_MULTIPLY
    timer.Start( 2 );
#endif
    A.MultiplyHMatMainPrecompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 2 );
    timer.Start( 3 );
#endif
    A.MultiplyHMatMainSums
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 3 );
    timer.Start( 4 );
#endif
    A.MultiplyHMatMainPassData
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 4 );
    timer.Start( 5 );
#endif
    A.MultiplyHMatMainBroadcasts
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 5 );
    timer.Start( 6 );
#endif
    A.MultiplyHMatMainPostcompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
    timer.Start( 7 );
#endif
    A.MultiplyHMatFHHPrecompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 7 );
    timer.Start( 8 );
#endif
    A.MultiplyHMatFHHSums
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 8 );
    timer.Start( 9 );
#endif
    A.MultiplyHMatFHHPassData
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 9 );
    timer.Start( 10 );
#endif
    A.MultiplyHMatFHHBroadcasts
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 10 );
    timer.Start( 11 );
#endif
    A.MultiplyHMatFHHPostcompute
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 11 );
    timer.Start( 12 );
#endif
//    A.MultiplyHMatFHHFinalize
//    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
    A.MultiplyHMatFHHCompress
    ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
    timer.Start( 13 );
#endif
    C.MultiplyHMatCompress();
    //C.MultiplyHMatRandomCompress();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 13 );
#endif

    C.PruneGhostNodes();

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-full-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:         " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatSingleLevelAccumulate
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatSingleLevelAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( numLevels_ != B.numLevels_ )
        throw std::logic_error("H-matrices must have same number of levels");
    if( level_ != B.level_ )
        throw std::logic_error("Mismatched levels");
#endif
    DistHMat2d<Scalar>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer;
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 1 );
#endif

#ifdef MEMORY_INFO
    A.PrintGlobalMemoryInfo("Matrix A before loop: ");
    B.PrintGlobalMemoryInfo("Matrix B before loop: ");
#endif

    const int startUpdate = 0;
    const int endUpdate = 4;

    const int numLevels = A.NumLevels();
    for( int level=0; level<numLevels; ++level )
    {
        const int startLevel = level;
        const int endLevel = level+1;

#ifdef TIME_MULTIPLY
        timer.Start( 2 );
#endif
        A.MultiplyHMatMainPrecompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 2 );
        timer.Start( 3 );
#endif
        A.MultiplyHMatMainSums
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 3 );
        timer.Start( 4 );
#endif
        A.MultiplyHMatMainPassData
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 4 );
        timer.Start( 5 );
#endif
        A.MultiplyHMatMainBroadcasts
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 5 );
        timer.Start( 6 );
#endif
        A.MultiplyHMatMainPostcompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
        timer.Start( 7 );
#endif
        A.MultiplyHMatFHHPrecompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 7 );
        timer.Start( 8 );
#endif
        A.MultiplyHMatFHHSums
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 8 );
        timer.Start( 9 );
#endif
        A.MultiplyHMatFHHPassData
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 9 );
        timer.Start( 10 );
#endif
        A.MultiplyHMatFHHBroadcasts
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 10 );
        timer.Start( 11 );
#endif
        A.MultiplyHMatFHHPostcompute
        ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 11 );
        timer.Start( 12 );
#endif
        //A.MultiplyHMatFHHFinalize
        //( B, C, startLevel, endLevel, startUpdate, endUpdate );
        A.MultiplyHMatFHHCompress
        ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
        timer.Start( 13 );
#endif
        C.MultiplyHMatCompress();
        //C.MultiplyHMatRandomCompress();
#ifdef TIME_MULTIPLY
        mpi::Barrier( mpi::COMM_WORLD );
        timer.Stop( 13 );
#endif
    }
    C.PruneGhostNodes();

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-singleLevel-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:         " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatSingleUpdateAccumulate
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatSingleUpdateAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( numLevels_ != B.numLevels_ )
        throw std::logic_error("H-matrices must have same number of levels");
    if( level_ != B.level_ )
        throw std::logic_error("Mismatched levels");
#endif
    DistHMat2d<Scalar>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

#ifdef TIME_MULTIPLY
    Timer timer;
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Start( 0 );
#endif
    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif
    A.MultiplyHMatFormGhostRanks( B );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 1 );
#endif

#ifdef MEMORY_INFO
    A.PrintGlobalMemoryInfo("Matrix A before loop: ");
    B.PrintGlobalMemoryInfo("Matrix B before loop: ");
#endif

    const int numLevels = A.NumLevels();
    for( int level=0; level<numLevels; ++level )
    {
        const int startLevel = level;
        const int endLevel = level+1;

        for( int update=0; update<4; ++update )
        {
            const int startUpdate = update;
            const int endUpdate = update+1;

#ifdef TIME_MULTIPLY
            timer.Start( 2 );
#endif
            A.MultiplyHMatMainPrecompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 2 );
            timer.Start( 3 );
#endif
            A.MultiplyHMatMainSums
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 3 );
            timer.Start( 4 );
#endif
            A.MultiplyHMatMainPassData
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 4 );
            timer.Start( 5 );
#endif
            A.MultiplyHMatMainBroadcasts
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 5 );
            timer.Start( 6 );
#endif
            A.MultiplyHMatMainPostcompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 6 );
#endif

#ifdef TIME_MULTIPLY
            timer.Start( 7 );
#endif
            A.MultiplyHMatFHHPrecompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 7 );
            timer.Start( 8 );
#endif
            A.MultiplyHMatFHHSums
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 8 );
            timer.Start( 9 );
#endif
            A.MultiplyHMatFHHPassData
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 9 );
            timer.Start( 10 );
#endif
            A.MultiplyHMatFHHBroadcasts
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 10 );
            timer.Start( 11 );
#endif
            A.MultiplyHMatFHHPostcompute
            ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 11 );
            timer.Start( 12 );
#endif
            //A.MultiplyHMatFHHFinalize
            //( B, C, startLevel, endLevel, startUpdate, endUpdate );
            A.MultiplyHMatFHHCompress
            ( B, C, startLevel, endLevel, startUpdate, endUpdate );
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 12 );
#endif

#ifdef TIME_MULTIPLY
            timer.Start( 13 );
#endif
            C.MultiplyHMatCompress();
            //C.MultiplyHMatRandomCompress();
#ifdef TIME_MULTIPLY
            mpi::Barrier( mpi::COMM_WORLD );
            timer.Stop( 13 );
#endif
        }
    }
    C.PruneGhostNodes();

#ifdef TIME_MULTIPLY
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-singleUpdate-" << commRank << ".log";
    std::ofstream file( os.str().c_str() );

    file << "Form ghost nodes: " << timer.GetTime( 0 ) << " seconds.\n"
         << "Form ghost ranks: " << timer.GetTime( 1 ) << " seconds.\n"
         << "Main precompute:  " << timer.GetTime( 2 ) << " seconds.\n"
         << "Main summations:  " << timer.GetTime( 3 ) << " seconds.\n"
         << "Main pass data:   " << timer.GetTime( 4 ) << " seconds.\n"
         << "Main broadcasts:  " << timer.GetTime( 5 ) << " seconds.\n"
         << "Main postcompute: " << timer.GetTime( 6 ) << " seconds.\n"
         << "FHH precompute:   " << timer.GetTime( 7 ) << " seconds.\n"
         << "FHH summations:   " << timer.GetTime( 8 ) << " seconds.\n"
         << "FHH pass data:    " << timer.GetTime( 9 ) << " seconds.\n"
         << "FHH broadcasts:   " << timer.GetTime( 10 ) << " seconds.\n"
         << "FHH postcompute:  " << timer.GetTime( 11 ) << " seconds.\n"
         << "FHH finalize:     " << timer.GetTime( 12 ) << " seconds.\n"
         << "Compress:         " << timer.GetTime( 13 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif
}

} // namespace dmhm
