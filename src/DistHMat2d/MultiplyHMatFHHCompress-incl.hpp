/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompress
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompress");
#endif
    const DistHMat2d<Scalar>& A = *this;
    Real error = lapack::MachineEpsilon<Real>();
    
    MultiplyHMatFHHCompressPrecompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

    MultiplyHMatFHHCompressReduces
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
    
    C.MultiplyHMatFHHCompressMidcompute
    ( error, startLevel, endLevel );

    MultiplyHMatFHHCompressBroadcasts
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0  );

    MultiplyHMatFHHCompressPostcompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

    C.MultiplyHMatFHHCompressCleanup
    ( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressPrecompute
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressPrecompute");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;
    const int rank = SampleRank( C.MaxRank() );
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
                    const unsigned teamLevel = C.teams_->TeamLevel(C.level_);
                    if( C.inTargetTeam_ ) 
                    {
                        const int key = A.sourceOffset_;
                        const Dense<Scalar>& colU = C.colXMap_.Get( key );
                        int Trank = colU.Width();
                        int Omegarank = A.rowOmega_.Width();

                        C.colUSqrMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );
                        Dense<Scalar>& colUSqr = C.colUSqrMap_.Get( key );

                        C.colPinvMap_.Set
                        ( key, new Dense<Scalar>( Omegarank, Trank ) );
                        Dense<Scalar>& colPinv = C.colPinvMap_.Get( key );

                        C.BLMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );

                        //_colUSqr = ( A B Omega1)' ( A B Omega1 )
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, A.LocalHeight(),
                         Scalar(1), colU.LockedBuffer(), colU.LDim(),
                                    colU.LockedBuffer(), colU.LDim(),
                         Scalar(0), colUSqr.Buffer(), colUSqr.LDim() );

                        //_colPinv = Omega2' (A B Omega1)
                        blas::Gemm
                        ( 'C', 'N', Omegarank, Trank, A.LocalHeight(),
                          Scalar(1), A.rowOmega_.LockedBuffer(), A.rowOmega_.LDim(),
                                     colU.LockedBuffer(), colU.LDim(),
                          Scalar(0), colPinv.Buffer(),      Omegarank );

                    }
                    if( C.inSourceTeam_ )
                    {
                        const int key = A.sourceOffset_;
                        const Dense<Scalar>& rowU = C.rowXMap_.Get( key );
                        int Trank = rowU.Width();
                        int Omegarank = B.colOmega_.Width();

                        C.rowUSqrMap_.Set
                        ( key, new Dense<Scalar>( Trank, Trank ) );
                        Dense<Scalar>& rowUSqr = C.rowUSqrMap_.Get( key );

                        C.rowPinvMap_.Set
                        ( key, new Dense<Scalar>( Omegarank, Trank ) );
                        Dense<Scalar>& rowPinv = C.rowPinvMap_.Get( key );

                        C.BRMap_.Set
                        ( key, new Dense<Scalar>( Trank, Omegarank ) );

                        //_rowUSqr = ( B' A' Omega2 )' ( B' A' Omega2 )
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, B.LocalWidth(),
                         Scalar(1), rowU.LockedBuffer(), rowU.LDim(),
                                    rowU.LockedBuffer(), rowU.LDim(),
                         Scalar(0), rowUSqr.Buffer(), rowUSqr.LDim() );

                        //_rowPinv = Omega1' (B' A' Omega2)
                        blas::Gemm
                        ( 'C', 'N', Omegarank, Trank, B.LocalWidth(),
                          Scalar(1), B.colOmega_.LockedBuffer(), B.colOmega_.LDim(),
                                     rowU.LockedBuffer(), rowU.LDim(),
                          Scalar(0), rowPinv.Buffer(),      Omegarank );
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReduces
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReduces");
#endif

    const int numLevels = C.teams_->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces, 0 );
    MultiplyHMatFHHCompressReducesCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, update );

    int totalsize = 0;
    for(int i=0; i<numReduces; ++i)
        totalsize += sizes[i];
    std::vector<Scalar> buffer( totalsize );
    std::vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReducesCount
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
        std::vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReducesCount");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReducesPack
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReducesPack");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressTreeReduces
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressTreeReduces");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReducesUnpack
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReducesUnpack");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressMidcompute
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressMidcompute");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;                                 
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressMidcompute
                    ( error, startLevel, endLevel);
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            // Calculate Eigenvalues of Squared Matrix               
            int sizemax=0;
            if( inTargetTeam_ )
            {
                colUSqrMap_.ResetIterator();
                const unsigned numEntries = colUSqrMap_.Size();
                for( int i=0; i<numEntries; ++i,colUSqrMap_.Increment())
                {
                    Dense<Scalar>& X = *colUSqrMap_.CurrentEntry();
                    sizemax = std::max( sizemax, X.Height() );
                }
            }
            else
            {
                rowUSqrMap_.ResetIterator();
                const unsigned numEntries = rowUSqrMap_.Size();
                for( int i=0; i<numEntries; ++i,rowUSqrMap_.Increment())
                {
                    Dense<Scalar>& X = *rowUSqrMap_.CurrentEntry();
                    sizemax = std::max( sizemax, X.Height() );
                }
            }
             
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( sizemax );
            lrwork=lapack::EVDRealWorkSize( sizemax );
            liwork=lapack::EVDIntWorkSize( sizemax );
                    
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
                                                                     
            if( inTargetTeam_ )
            {
#ifndef RELEASE
                if( colUSqrMap_.Size() != colPinvMap_.Size() )
                    throw std::logic_error
                    ("Number of colUSqr not equal to number of colPinv");
#endif
                colUSqrMap_.ResetIterator();
                const unsigned numEntries = colUSqrMap_.Size();
                for(int i=0; i<numEntries; ++i, colUSqrMap_.Increment() ) 
                {
                    int key = colUSqrMap_.CurrentKey();
                    Dense<Scalar>& USqr = *colUSqrMap_.CurrentEntry();
                    Dense<Scalar>& Pinv = colPinvMap_.Get( key );
                    Dense<Scalar>& BL = BLMap_.Get( key );
                    std::vector<Real> USqrEig(USqr.Height());
                    lapack::EVD  
                    ('V', 'U', USqr.Height(), 
                               USqr.Buffer(), USqr.LDim(),
                               &USqrEig[0],
                               &evdWork[0],     lwork,
                               &evdIntWork[0],  liwork,
                               &evdRealWork[0], lrwork );


                    //colOmegaT = Omega2' T1
                    Dense<Scalar> OmegaT;
                    hmat_tools::Copy(Pinv, OmegaT);
 
                    Real Eigmax;
                    if( USqrEig[USqrEig.size()-1] > (Real)0 )
                        Eigmax= USqrEig[USqrEig.size()-1] ;
                    else
                        Eigmax=0;
 
                    Real TruncBound = sqrt(std::abs(error)*Eigmax*USqrEig.size());
                    for(int j=0; j<USqrEig.size(); j++)
                    {
                        Real sqrteig;
                        if( USqrEig[j]>0 )
                            sqrteig=sqrt(USqrEig[j]);
                        else
                            sqrteig=-sqrt(-USqrEig[j]);
 
                        if( sqrteig > TruncBound )
                        {
                            for(int i=0; i<USqr.Height(); ++i)
                                USqr.Set(i,j,USqr.Get(i,j)/sqrteig);
                        }
                        else
                        {
                            for(int i=0; i<USqr.Height(); ++i)
                                USqr.Set(i,j,Scalar(0));
                        }
                    }
                    
                    blas::Gemm
                    ( 'N', 'N', OmegaT.Height(), USqr.Width(), USqr.Height(),
                     Scalar(1), OmegaT.LockedBuffer(), OmegaT.LDim(),
                                USqr.LockedBuffer(), USqr.LDim(),
                     Scalar(0), Pinv.Buffer(), Pinv.LDim() );
 
                    int rank=std::min(Pinv.Height(), Pinv.Width());
                    std::vector<Real> singularValues(rank);
                    std::vector<Scalar> U(rank*Pinv.Height()), VH(rank*Pinv.Width());
                    int lsvdwork=lapack::SVDWorkSize(Pinv.Height(), Pinv.Width());
                    int lrsvdwork=lapack::SVDRealWorkSize(Pinv.Height(), Pinv.Width());
                    std::vector<Scalar> svdWork(lsvdwork);
                    std::vector<Real> svdRealWork(lrsvdwork);
                    lapack::AdjointPseudoInverse
                    ( Pinv.Height(), Pinv.Width(), Pinv.Buffer(), Pinv.LDim(),
                      &singularValues[0], &U[0], Pinv.Height(), &VH[0], rank,
                      &svdWork[0], lsvdwork, &svdRealWork[0]);
 
                    Dense<Scalar> Ztmp(USqr.Height(), Pinv.Height());
                    blas::Gemm
                    ( 'N', 'C', USqr.Height(), Pinv.Height(), USqr.Width(),
                     Scalar(1), USqr.LockedBuffer(), USqr.LDim(),
                                Pinv.LockedBuffer(), Pinv.LDim(),
                     Scalar(0), Ztmp.Buffer(), Ztmp.LDim() );
                    
                    blas::Gemm
                    ( 'N', 'N', Ztmp.Height(), OmegaT.Width(), Ztmp.Width(),
                     Scalar(1), Ztmp.LockedBuffer(), Ztmp.LDim(),
                                OmegaT.LockedBuffer(), OmegaT.LDim(),
                     Scalar(0), BL.Buffer(), BL.LDim() );
                }
            }
                                                                     
            if( inSourceTeam_ )
            {
#ifndef RELEASE
                if( rowUSqrMap_.Size() != rowPinvMap_.Size() )
                    throw std::logic_error
                    ("Number of rowUSqr not equal to number of rowPinv");
#endif
                rowUSqrMap_.ResetIterator();
                const unsigned numEntries = rowUSqrMap_.Size();
                for(int i=0; i<numEntries; ++i, rowUSqrMap_.Increment() )
                {
					int key = rowUSqrMap_.CurrentKey();
                    Dense<Scalar>& USqr = *rowUSqrMap_.CurrentEntry();
                    Dense<Scalar>& Pinv = rowPinvMap_.Get( key );
                    Dense<Scalar>& BR = BRMap_.Get( key );
                    std::vector<Real> USqrEig(USqr.Height());
                    lapack::EVD                                                                 
                    ('V', 'U', USqr.Height(), 
                               USqr.Buffer(), USqr.LDim(),
                               &USqrEig[0],
                               &evdWork[0],     lwork,
                               &evdIntWork[0],  liwork,
                               &evdRealWork[0], lrwork );
                                                                                                
                    //colOmegaT = Omega2' T1
                    Dense<Scalar> OmegaT;
                    hmat_tools::Copy(Pinv, OmegaT);
                                                                                                
                    Real Eigmax;
                    if( USqrEig[USqrEig.size()-1] > (Real)0 )
                        Eigmax= USqrEig[USqrEig.size()-1] ;
                    else
                        Eigmax=0;
                                                                                                
                    Real TruncBound = sqrt(std::abs(error)*Eigmax*USqrEig.size());
                    for(int j=0; j<USqrEig.size(); j++)
                    {
                        Real sqrteig;
                        if( USqrEig[j]>0 )
                            sqrteig=sqrt(USqrEig[j]);
                        else
                            sqrteig=-sqrt(-USqrEig[j]);
                                                                                                
                        if( sqrteig > TruncBound )
                        {
                            for(int i=0; i<USqr.Height(); ++i)
                                USqr.Set(i,j,USqr.Get(i,j)/sqrteig);
                        }
                        else
                        {
                            for(int i=0; i<USqr.Height(); ++i)
                                USqr.Set(i,j,Scalar(0));
                        }
                    }
                    
                    blas::Gemm
                    ( 'N', 'N', OmegaT.Height(), USqr.Width(), USqr.Height(),
                     Scalar(1), OmegaT.LockedBuffer(), OmegaT.LDim(),
                                USqr.LockedBuffer(), USqr.LDim(),
                     Scalar(0), Pinv.Buffer(), Pinv.LDim() );
                                                                                                
                    int rank=std::min(Pinv.Height(), Pinv.Width());
                    std::vector<Real> singularValues(rank);
                    std::vector<Scalar> U(rank*Pinv.Height()), VH(rank*Pinv.Width());
                    int lsvdwork=lapack::SVDWorkSize(Pinv.Height(), Pinv.Width());
                    int lrsvdwork=lapack::SVDRealWorkSize(Pinv.Height(), Pinv.Width());
                    std::vector<Scalar> svdWork(lsvdwork);
                    std::vector<Real> svdRealWork(lrsvdwork);
                    lapack::AdjointPseudoInverse
                    ( Pinv.Height(), Pinv.Width(), Pinv.Buffer(), Pinv.LDim(),
                      &singularValues[0], &U[0], Pinv.Height(), &VH[0], rank,
                      &svdWork[0], lsvdwork, &svdRealWork[0]);
                                                                                                
                    blas::Gemm
                    ( 'N', 'C', USqr.Height(), Pinv.Height(), USqr.Width(),
                     Scalar(1), USqr.LockedBuffer(), USqr.LDim(),
                                Pinv.LockedBuffer(), Pinv.LDim(),
                     Scalar(0), BR.Buffer(), BR.LDim() );
                    //hmat_tools::Conjugate( BR );
                }
            }
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcasts
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatFHHCompressBroadcastsCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, update );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    std::vector<Scalar> buffer( totalsize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcastsCount
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
        std::vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsCount");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcastsPack
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
        std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsPack");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressTreeBroadcasts
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressTreeBroadcasts");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcastsUnpack
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsUnpack");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressPostcompute
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressPostcompute");
#endif
    const DistHMat2d<Scalar>& A = *this;
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
                        blas::Gemm
                        ('N', 'N', colU.Height(), BL.Width(), colU.Width(), 
                         Scalar(1), colU.LockedBuffer(), colU.LDim(),
                                    BL.LockedBuffer(), BL.LDim(),
                         Scalar(0), Ztmp.Buffer(),  Ztmp.LDim() );
                        hmat_tools::Copy(Ztmp, colU);
                    }                                                       
                    if( C.inSourceTeam_ )                                   
                    {                                                       
                        Dense<Scalar>& BR = C.BRMap_.Get( key );      
                        Dense<Scalar>& rowU = C.rowXMap_.Get( key );
                        Dense<Scalar> Ztmp(rowU.Height(), BR.Width());
                        blas::Gemm
                        ('N', 'N', rowU.Height(), BR.Width(), rowU.Width(), 
                         Scalar(1), rowU.LockedBuffer(), rowU.LDim(),
                                    BR.LockedBuffer(), BR.LDim(),
                         Scalar(0), Ztmp.Buffer(),  Ztmp.LDim() );
                        hmat_tools::Copy(Ztmp, rowU);
                        hmat_tools::Conjugate( rowU );
                    }                                                       
                }
            }
            else if( C.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
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
DistHMat2d<Scalar>::MultiplyHMatFHHCompressCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressCleanup");
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
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressCleanup
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK:
    case LOW_RANK_GHOST:
    {
        if( level_ < startLevel )
            break;
        colPinvMap_.Clear();
        rowPinvMap_.Clear();
        colUSqrMap_.Clear();
        rowUSqrMap_.Clear();
        BLMap_.Clear();
        BRMap_.Clear();
    }
    default:
        break;
    }
}

} // namespace dmhm
