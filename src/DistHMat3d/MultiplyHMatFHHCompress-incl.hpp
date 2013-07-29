/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompress
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompress");
#endif

#ifdef MEMORY_INFO
    //C.PrintMemoryInfo( "MemoryInfo before FHH Compression" );
#endif

    MultiplyHMatFHHCompressPrecompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

    MultiplyHMatFHHCompressReduces
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

    const Real midcomputeTol = MidcomputeTolerance<Real>();
    MultiplyHMatFHHCompressMidcompute
    ( B, C, midcomputeTol, startLevel, endLevel, startUpdate, endUpdate, 0 );

    MultiplyHMatFHHCompressBroadcasts
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0  );

    MultiplyHMatFHHCompressPostcompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

    C.MultiplyHMatFHHCompressCleanup
    ( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressPrecompute
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressPrecompute");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    if( C.inTargetTeam_ )
                    {
                        const int key = A.sourceOffset_;
                        const Dense<Scalar>& colU = C.colXMap_.Get( key );
                        const int Trank = colU.Width();
                        const int LH = colU.Height();
                        const int Omegarank = A.rowOmega_.Width();

                        C.colUSqrMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );
                        Dense<Scalar>& colUSqr = C.colUSqrMap_.Get( key );

                        C.colPinvMap_.Set
                        ( key, new Dense<Scalar>( Omegarank, Trank ) );
                        Dense<Scalar>& colPinv = C.colPinvMap_.Get( key );

                        C.BLMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );

                        //_colUSqr = ( A B Omega1)' ( A B Omega1 )
                        // TODO: Replace this with a Herk call...
                        colUSqr.Init();
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, LH,
                         Scalar(1), colU.LockedBuffer(), colU.LDim(),
                                    colU.LockedBuffer(), colU.LDim(),
                         Scalar(0), colUSqr.Buffer(), colUSqr.LDim() );

                        //_colPinv = Omega2' (A B Omega1)
                        colPinv.Init();
                        blas::Gemm
                        ( 'C', 'N', Omegarank, Trank, LH,
                          Scalar(1), A.rowOmega_.LockedBuffer(),
                                     A.rowOmega_.LDim(),
                                     colU.LockedBuffer(), colU.LDim(),
                          Scalar(0), colPinv.Buffer(),  colPinv.LDim() );
                    }
                    if( C.inSourceTeam_ )
                    {
                        const int key = A.sourceOffset_;
                        const Dense<Scalar>& rowU = C.rowXMap_.Get( key );
                        const int Trank = rowU.Width();
                        const int LH = rowU.Height();
                        const int Omegarank = B.colOmega_.Width();

                        C.rowUSqrMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );
                        Dense<Scalar>& rowUSqr = C.rowUSqrMap_.Get( key );

                        C.rowPinvMap_.Set
                        ( key, new Dense<Scalar>( Omegarank, Trank ) );
                        Dense<Scalar>& rowPinv = C.rowPinvMap_.Get( key );

                        C.BRMap_.Set
                        ( key, new Dense<Scalar>( Trank, Omegarank ) );

                        //_rowUSqr = ( B' A' Omega2 )' ( B' A' Omega2 )
                        // TODO: Replace this with a Herk call...
                        rowUSqr.Init();
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, LH,
                         Scalar(1), rowU.LockedBuffer(), rowU.LDim(),
                                    rowU.LockedBuffer(), rowU.LDim(),
                         Scalar(0), rowUSqr.Buffer(), rowUSqr.LDim() );

                        //_rowPinv = Omega1' (B' A' Omega2)
                        rowPinv.Init();
                        blas::Gemm
                        ( 'C', 'N', Omegarank, Trank, LH,
                          Scalar(1), B.colOmega_.LockedBuffer(),
                                     B.colOmega_.LDim(),
                                     rowU.LockedBuffer(), rowU.LDim(),
                          Scalar(0), rowPinv.Buffer(), rowPinv.LDim() );
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressPrecompute
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressReduces
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressReduces");
#endif

    const int numLevels = C.teams_->NumLevels();
    const int numReduces = numLevels-1;
    Vector<int> sizes( numReduces, 0 );
    MultiplyHMatFHHCompressReducesCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, update );

    int totalsize = 0;
    for(int i=0; i<numReduces; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressReducesPack
    ( B, C, buffer, offsetscopy,
      startLevel, endLevel, startUpdate, endUpdate, update );

    C.MultiplyHMatFHHCompressTreeReduces( buffer, sizes );

    MultiplyHMatFHHCompressReducesUnpack
    ( B, C, buffer, offsets,
      startLevel, endLevel, startUpdate, endUpdate, update );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressReducesCount
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressReduceCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    if( C.inTargetTeam_ )
                    {
                        const Dense<Scalar>& colUSqr = C.colUSqrMap_.Get( key );
                        const Dense<Scalar>& colPinv = C.colPinvMap_.Get( key );
                        sizes[C.level_] += colUSqr.Height()*colUSqr.Width();
                        sizes[C.level_] += colPinv.Height()*colPinv.Width();
                    }
                    if( C.inSourceTeam_ )
                    {
                        const Dense<Scalar>& rowUSqr = C.rowUSqrMap_.Get( key );
                        const Dense<Scalar>& rowPinv = C.rowPinvMap_.Get( key );
                        sizes[C.level_] += rowUSqr.Height()*rowUSqr.Width();
                        sizes[C.level_] += rowPinv.Height()*rowPinv.Width();
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressReducesCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressReducesPack
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressReducePack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    int size;
                    if( C.inTargetTeam_ )
                    {
                        const Dense<Scalar>& colUSqr = C.colUSqrMap_.Get( key );
                        const Dense<Scalar>& colPinv = C.colPinvMap_.Get( key );

                        size = colUSqr.Height()*colUSqr.Width();
                        MemCopy
                        ( &buffer[offsets[C.level_]], colUSqr.LockedBuffer(),
                          size );
                        offsets[C.level_] += size;

                        size = colPinv.Height()*colPinv.Width();
                        MemCopy
                        ( &buffer[offsets[C.level_]], colPinv.LockedBuffer(),
                          size );
                        offsets[C.level_] += size;
                    }
                    if( C.inSourceTeam_ )
                    {
                        const Dense<Scalar>& rowUSqr = C.rowUSqrMap_.Get( key );
                        const Dense<Scalar>& rowPinv = C.rowPinvMap_.Get( key );

                        size = rowUSqr.Height()*rowUSqr.Width();
                        MemCopy
                        ( &buffer[offsets[C.level_]], rowUSqr.LockedBuffer(),
                          size );
                        offsets[C.level_] += size;

                        size = rowPinv.Height()*rowPinv.Width();
                        MemCopy
                        ( &buffer[offsets[C.level_]], rowPinv.LockedBuffer(),
                          size );
                        offsets[C.level_] += size;
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressReducesPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), buffer, offsets,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressTreeReduces
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressTreeReduces");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressReducesUnpack
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  const Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressReducesUnpack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    MPI_Comm team = C.teams_->Team( C.level_ );
                    const int teamRank = mpi::CommRank( team );
                    const int key = A.sourceOffset_;
                    int size;
                    if( C.inTargetTeam_ && teamRank == 0 )
                    {
                        Dense<Scalar>& colUSqr = C.colUSqrMap_.Get( key );
                        Dense<Scalar>& colPinv = C.colPinvMap_.Get( key );

                        size = colUSqr.Height()*colUSqr.Width();
                        MemCopy
                        ( colUSqr.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;

                        size = colPinv.Height()*colPinv.Width();
                        MemCopy
                        ( colPinv.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;
                    }
                    if( C.inSourceTeam_ && teamRank == 0 )
                    {
                        Dense<Scalar>& rowUSqr = C.rowUSqrMap_.Get( key );
                        Dense<Scalar>& rowPinv = C.rowPinvMap_.Get( key );

                        size = rowUSqr.Height()*rowUSqr.Width();
                        MemCopy
                        ( rowUSqr.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;

                        size = rowPinv.Height()*rowPinv.Width();
                        MemCopy
                        ( rowPinv.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressReducesUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), buffer, offsets,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressMidcompute
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  Real epsilon,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressMidcompute");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                mpi::Comm team = teams_->Team( C.level_ );
                const int teamRank = mpi::CommRank( team );
                if( teamRank != 0 )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    if( C.inTargetTeam_ )
                    {
                        Dense<Scalar>& USqr = C.colUSqrMap_.Get( key );
                        Dense<Scalar>& Pinv = C.colPinvMap_.Get( key );
                        Dense<Scalar>& BL = C.BLMap_.Get( key );

                        const int k = USqr.Height();
                        Vector<Real> USqrEig( k );
                        lapack::EVD
                        ( 'V', 'U', k,
                          USqr.Buffer(), USqr.LDim(), &USqrEig[0] );

                        //colOmegaT = Omega2' T1
                        Dense<Scalar> OmegaT;
                        hmat_tools::Copy( Pinv, OmegaT );

                        Real maxEig = 0;
                        if( k > 0 )
                            maxEig = std::max( USqrEig[k-1], Real(0) );

                        // TODO: Iterate backwards to compute cutoff in manner
                        //       similar to AdjointPseudoInverse
                        const Real tolerance = sqrt(epsilon*maxEig*k);
                        for( int j=0; j<k; j++ )
                        {
                            const Real omega = std::max( USqrEig[j], Real(0) );
                            const Real sqrtOmega = sqrt( omega );
                            if( sqrtOmega > tolerance )
                                for( int i=0; i<k; ++i )
                                    USqr.Set(i,j,USqr.Get(i,j)/sqrtOmega);
                            else
                                MemZero( USqr.Buffer(0,j), k );
                        }

                        Pinv.Init();
                        blas::Gemm
                        ( 'N', 'N', OmegaT.Height(), k, k,
                         Scalar(1), OmegaT.LockedBuffer(), OmegaT.LDim(),
                                    USqr.LockedBuffer(), USqr.LDim(),
                         Scalar(0), Pinv.Buffer(), Pinv.LDim() );

                        lapack::AdjointPseudoInverse
                        ( Pinv.Height(), Pinv.Width(),
                          Pinv.Buffer(), Pinv.LDim(), epsilon );

                        Dense<Scalar> Ztmp( k, Pinv.Height() );
                        Ztmp.Init();
                        blas::Gemm
                        ( 'N', 'C', k, Pinv.Height(), k,
                         Scalar(1), USqr.LockedBuffer(), USqr.LDim(),
                                    Pinv.LockedBuffer(), Pinv.LDim(),
                         Scalar(0), Ztmp.Buffer(), Ztmp.LDim() );

                        BL.Init();
                        blas::Gemm
                        ( 'N', 'N', k, OmegaT.Width(), Ztmp.Width(),
                         Scalar(1), Ztmp.LockedBuffer(), Ztmp.LDim(),
                                    OmegaT.LockedBuffer(), OmegaT.LDim(),
                         Scalar(0), BL.Buffer(), BL.LDim() );
                    }
                    if( C.inSourceTeam_ )
                    {
                        Dense<Scalar>& USqr = C.rowUSqrMap_.Get( key );
                        Dense<Scalar>& Pinv = C.rowPinvMap_.Get( key );
                        Dense<Scalar>& BR = C.BRMap_.Get( key );

                        const int k = USqr.Height();
                        Vector<Real> USqrEig( k );
                        lapack::EVD
                        ( 'V', 'U', k,
                          USqr.Buffer(), USqr.LDim(), &USqrEig[0] );

                        //colOmegaT = Omega2' T1
                        Dense<Scalar> OmegaT;
                        hmat_tools::Copy( Pinv, OmegaT );

                        Real maxEig = 0;
                        if( k > 0 )
                            maxEig = std::max( USqrEig[k-1], Real(0) );

                        // TODO: Iterate backwards to compute cutoff in manner
                        //       similar to AdjointPseudoInverse
                        const Real tolerance = sqrt(epsilon*maxEig*k);
                        for( int j=0; j<k; j++ )
                        {
                            const Real omega = std::max( USqrEig[j], Real(0) );
                            const Real sqrtOmega = sqrt( omega );
                            if( sqrtOmega > tolerance )
                                for( int i=0; i<k; ++i )
                                    USqr.Set(i,j,USqr.Get(i,j)/sqrtOmega);
                            else
                                MemZero( USqr.Buffer(0,j), k );
                        }

                        Pinv.Init();
                        blas::Gemm
                        ( 'N', 'N', OmegaT.Height(), k, k,
                         Scalar(1), OmegaT.LockedBuffer(), OmegaT.LDim(),
                                    USqr.LockedBuffer(), USqr.LDim(),
                         Scalar(0), Pinv.Buffer(), Pinv.LDim() );

                        lapack::AdjointPseudoInverse
                        ( Pinv.Height(), Pinv.Width(),
                          Pinv.Buffer(), Pinv.LDim(), epsilon );

                        BR.Init();
                        blas::Gemm
                        ( 'N', 'C', k, Pinv.Height(), k,
                         Scalar(1), USqr.LockedBuffer(), USqr.LDim(),
                                    Pinv.LockedBuffer(), Pinv.LDim(),
                         Scalar(0), BR.Buffer(), BR.LDim() );
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressMidcompute
                            ( nodeB.Child(r,s), nodeC.Child(t,s), epsilon,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}


template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressBroadcasts
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatFHHCompressBroadcastsCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, update );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressBroadcastsPack
    ( B, C, buffer, offsetscopy,
      startLevel, endLevel, startUpdate, endUpdate, update );

    MultiplyHMatFHHCompressTreeBroadcasts( buffer, sizes );

    MultiplyHMatFHHCompressBroadcastsUnpack
    ( B, C, buffer, offsets,
      startLevel, endLevel, startUpdate, endUpdate, update );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressBroadcastsCount
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressBroadcastsCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    if( C.inTargetTeam_ )
                    {
                        const Dense<Scalar>& BL = C.BLMap_.Get( key );
                        sizes[C.level_] += BL.Height()*BL.Width();
                    }
                    if( C.inSourceTeam_ )
                    {
                        const Dense<Scalar>& BR = C.BRMap_.Get( key );
                        sizes[C.level_] += BR.Height()*BR.Width();
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressBroadcastsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressBroadcastsPack
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressBroadcastsPack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    mpi::Comm team = C.teams_->Team( C.level_ );
                    const int teamRank = mpi::CommRank( team );
                    if( teamRank == 0 )
                    {
                        const int key = A.sourceOffset_;
                        if( C.inTargetTeam_ )
                        {
                            const Dense<Scalar>& BL = C.BLMap_.Get( key );
                            int size = BL.Height()*BL.Width();
                            MemCopy
                            ( &buffer[offsets[C.level_]], BL.LockedBuffer(),
                              size );
                            offsets[C.level_] += size;
                        }
                        if( C.inSourceTeam_ )
                        {
                            const Dense<Scalar>& BR = C.BRMap_.Get( key );
                            int size = BR.Height()*BR.Width();
                            MemCopy
                            ( &buffer[offsets[C.level_]], BR.LockedBuffer(),
                              size );
                            offsets[C.level_] += size;
                        }
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressBroadcastsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), buffer, offsets,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressTreeBroadcasts
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressTreeBroadcasts");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressBroadcastsUnpack
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  const Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressBroadcastsUnpack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.block_.type != DIST_LOW_RANK )
                    break;
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    if( C.inTargetTeam_ )
                    {
                        Dense<Scalar>& BL = C.BLMap_.Get( key );
                        int size = BL.Height()*BL.Width();
                        MemCopy
                        ( BL.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;
                    }
                    if( C.inSourceTeam_ )
                    {
                        Dense<Scalar>& BR = C.BRMap_.Get( key );
                        int size = BR.Height()*BR.Width();
                        MemCopy
                        ( BR.Buffer(), &buffer[offsets[C.level_]],
                          size );
                        offsets[C.level_] += size;
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressBroadcastsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s), buffer, offsets,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressPostcompute
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressPostcompute");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const int key = A.sourceOffset_;
                    if( C.inTargetTeam_ )
                    {
                        Dense<Scalar>& BL = C.BLMap_.Get( key );
                        Dense<Scalar>& colU = C.colXMap_.Get( key );
                        Dense<Scalar> Ztmp(colU.Height(), BL.Width());
                        Ztmp.Init();
                        blas::Gemm
                        ('N', 'N', colU.Height(), BL.Width(), colU.Width(),
                         Scalar(1), colU.LockedBuffer(), colU.LDim(),
                                    BL.LockedBuffer(), BL.LDim(),
                         Scalar(0), Ztmp.Buffer(),  Ztmp.LDim() );
                        hmat_tools::Copy( Ztmp, colU );
                    }
                    if( C.inSourceTeam_ )
                    {
                        Dense<Scalar>& BR = C.BRMap_.Get( key );
                        Dense<Scalar>& rowU = C.rowXMap_.Get( key );
                        Dense<Scalar> Ztmp(rowU.Height(), BR.Width());
                        Ztmp.Init();
                        blas::Gemm
                        ('N', 'N', rowU.Height(), BR.Width(), rowU.Width(),
                         Scalar(1), rowU.LockedBuffer(), rowU.LDim(),
                                    BR.LockedBuffer(), BR.LDim(),
                         Scalar(0), Ztmp.Buffer(),  Ztmp.LDim() );
                        hmat_tools::Copy( Ztmp, rowU );
                        hmat_tools::Conjugate( rowU );
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressPostcompute
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHCompressCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHCompressCleanup");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressCleanup
                    ( startLevel, endLevel );
        }
    }
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        if( level_ < startLevel )
            break;
        colPinvMap_.Clear();
        rowPinvMap_.Clear();
        colUSqrMap_.Clear();
        rowUSqrMap_.Clear();
        BLMap_.Clear();
        BRMap_.Clear();
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
