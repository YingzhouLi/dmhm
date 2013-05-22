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
    
    // RYAN: Again...please properly format and don't check in debug code
mpi::Comm team = teams_->Team( level_ );
const int teamRank = mpi::CommRank( team );
int print;
if(teamRank==0)
    print=0;
else
    print=0;

if(print)
    std::cout << teamRank << " Sum" << std::endl;
    C.MultiplyHMatFHHCompressSum
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Precompute" << std::endl;
    MultiplyHMatFHHCompressPrecompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

if(print)
    std::cout << teamRank << " Reduces" << std::endl;
    C.MultiplyHMatFHHCompressReduces
    ( startLevel, endLevel );
    
if(print)
    std::cout << teamRank << " Midcompute" << std::endl;
    C.MultiplyHMatFHHCompressMidcompute
    ( error, startLevel, endLevel );

if(print)
    std::cout << teamRank << " Broadcasts" << std::endl;
    C.MultiplyHMatFHHCompressBroadcasts
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Postcompute" << std::endl;
    C.MultiplyHMatFHHCompressPostcompute
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Cleanup" << std::endl;
    C.MultiplyHMatFHHCompressCleanup
    ( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressSum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressSum");
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
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressSum
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        const char option = 'T';
        int totalrank=colXMap_.FirstWidth();
        
        if( inTargetTeam_ && totalrank > 0 && LH > 0 )
        {
            colU_.Resize( LH, totalrank, LH );
            hmat_tools::Scale( Scalar(0), colU_ );

            int numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                hmat_tools::Add
                ( Scalar(1), colU_, Scalar(1), U, colU_ );
            }
        }

        totalrank=rowXMap_.FirstWidth();
        if( inSourceTeam_ && totalrank > 0 && LW > 0 )
        {
            rowU_.Resize( LW, totalrank, LW );
            hmat_tools::Scale( Scalar(0), rowU_ );

            int numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& U = *rowXMap_.CurrentEntry();
                hmat_tools::Add
                ( Scalar(1), rowU_, Scalar(1), U, rowU_ );
            }
        }
        break;
    }
    default:
        break;
    }
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
{
_USqr.Print("_USqr******************************************");
_VSqr.Print("_VSqr******************************************");
}*/
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
                        int Trank = C.colU_.Width();
                        int Omegarank = A.rowOmega_.Width();

                        //_colUSqr = ( A B Omega1)' ( A B Omega1 )
                        C.colUSqr_.Resize( Trank, Trank, Trank );
                        C.colUSqrEig_.resize( Trank );
                        C.BL_.Resize(Trank, Trank, Trank);
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, A.LocalHeight(),
                         Scalar(1), C.colU_.LockedBuffer(), C.colU_.LDim(),
                                    C.colU_.LockedBuffer(), C.colU_.LDim(),
                         Scalar(0), C.colUSqr_.Buffer(), C.colUSqr_.LDim() );

                        //_colPinv = Omega2' (A B Omega1)
                        C.colPinv_.Resize( Omegarank, Trank, Omegarank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Omegarank, A.LocalHeight(),
                          Scalar(1), C.colU_.LockedBuffer(), C.colU_.LDim(),
                                     A.rowOmega_.LockedBuffer(), A.rowOmega_.LDim(),
                          Scalar(0), C.colPinv_.Buffer(),      Omegarank );

                    }
                    if( C.inSourceTeam_ )
                    {
                        int Trank = C.rowU_.Width();
                        int Omegarank = B.colOmega_.Width();

                        //_rowUSqr = ( B' A' Omega2 )' ( B' A' Omega2 )
                        C.rowUSqr_.Resize( Trank, Trank, Trank );
                        C.rowUSqrEig_.resize( Trank );
                        C.BR_.Resize(Trank, Omegarank, Trank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, B.LocalWidth(),
                         Scalar(1), C.rowU_.LockedBuffer(), C.rowU_.LDim(),
                                    C.rowU_.LockedBuffer(), C.rowU_.LDim(),
                         Scalar(0), C.rowUSqr_.Buffer(), C.rowUSqr_.LDim() );

                        //_rowPinv = (B' A' Omega2)' Omega1
                        C.rowPinv_.Resize( Trank, Omegarank, Trank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Omegarank, B.LocalWidth(),
                          Scalar(1), C.rowU_.LockedBuffer(), C.rowU_.LDim(),
                                     B.colOmega_.LockedBuffer(), B.colOmega_.LDim(),
                          Scalar(0), C.rowPinv_.Buffer(),      Trank );
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
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReduces");
#endif

    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces, 0 );
    MultiplyHMatFHHCompressReducesCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numReduces; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressReducesPack
    ( buffer, offsetscopy, startLevel, endLevel );
    
    MultiplyHMatFHHCompressTreeReduces( buffer, sizes );
    
    MultiplyHMatFHHCompressReducesUnpack
    ( buffer, offsets, startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReducesCount
( std::vector<int>& sizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReduceCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        if( level_ >= startLevel && level_ < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        sizes[level_] += colUSqr_.Height()*colUSqr_.Width();
        sizes[level_] += colPinv_.Height()*colPinv_.Width();
        sizes[level_] += rowUSqr_.Height()*rowUSqr_.Width();
        sizes[level_] += rowPinv_.Height()*rowPinv_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressReducesPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReducePack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int size=colUSqr_.Height()*colUSqr_.Width();
        if( size > 0 )
        {
            MemCopy( &buffer[offsets[level_]], colUSqr_.LockedBuffer(), size );
            offsets[level_] += size;
        }

        size=colPinv_.Height()*colPinv_.Width();
        if( size > 0 )
        {
            MemCopy( &buffer[offsets[level_]], colPinv_.LockedBuffer(), size );
            offsets[level_] += size;
        }

        size=rowUSqr_.Height()*rowUSqr_.Width();
        if( size > 0 )
        {
            MemCopy( &buffer[offsets[level_]], rowUSqr_.LockedBuffer(), size );
            offsets[level_] += size;
        }

        size=rowPinv_.Height()*rowPinv_.Width();
        if( size > 0 )
        {
            MemCopy( &buffer[offsets[level_]], rowPinv_.LockedBuffer(), size );
            offsets[level_] += size;
        }
        break;
    }
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
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressReducesUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int size=colUSqr_.Height()*colUSqr_.Width();                
            if( size > 0 )
            {
                MemCopy( colUSqr_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            size=colPinv_.Height()*colPinv_.Width();                
            if( size > 0 )
            {
                MemCopy( colPinv_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            size=rowUSqr_.Height()*rowUSqr_.Width();
            if( size > 0 )
            {
                MemCopy( rowUSqr_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            size=rowPinv_.Height()*rowPinv_.Width();                
            if( size > 0 )
            {
                MemCopy( rowPinv_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
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
                    node.Child(t,s).MultiplyHMatCompressFEigenDecomp
                    ( startLevel, endLevel);
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
            int maxSize = std::max(colUSqr_.Height(), rowUSqr_.Height());
             
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( maxSize );
            lrwork=lapack::EVDRealWorkSize( maxSize );
            liwork=lapack::EVDIntWorkSize( maxSize );
                    
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_VSqr.Print("_VSqr Before svd");*/
                                                                     
            if( colUSqr_.Height() > 0 )
            {
                lapack::EVD
                ('V', 'U', colUSqr_.Height(), 
                           colUSqr_.Buffer(), colUSqr_.LDim(),
                           &colUSqrEig_[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );

                //colOmegaT = T1' Omega2
                Dense<Scalar> colOmegaT;
                colOmegaT.Resize
                ( colPinv_.Height(), colPinv_.Width(), colPinv_.LDim() );
                MemCopy
                ( colOmegaT.Buffer(), colPinv_.LockedBuffer(),
                  colPinv_.Height()*colPinv_.Width() );

                Real maxEig;
                if( colUSqrEig_[colUSqrEig_.size()-1]>0 )
                    maxEig=sqrt( colUSqrEig_[colUSqrEig_.size()-1] );
                else
                    maxEig=0;

                for(int j=0; j<colUSqrEig_.size(); j++)
                    if( colUSqrEig_[j] > error*maxEig*colUSqrEig_.size() )
                    {
                        Real sqrteig=sqrt(colUSqrEig_[j]);
                        for(int i=0; i<colUSqr_.Height(); ++i)
                            colUSqr_.Set(i,j,colUSqr_.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<colUSqr_.Height(); ++i)
                            colUSqr_.Set(i,j,Scalar(0));
                    }
                
                blas::Gemm
                ( 'C', 'N', colUSqr_.Height(), colPinv_.Width(), colUSqr_.Width(),
                 Scalar(1), colUSqr_.LockedBuffer(), colUSqr_.LDim(),
                            colPinv_.LockedBuffer(), colPinv_.LDim(),
                 Scalar(0), colPinv_.Buffer(), colPinv_.LDim() );

                int rank=std::min(colPinv_.Height(), colPinv_.Width());
                std::vector<Real> singularValues(rank);
                std::vector<Scalar> U(rank*colPinv_.Height()), VH(rank*colPinv_.Width());
                int lsvdwork=lapack::SVDWorkSize(colPinv_.Height(), colPinv_.Width());
                int lrsvdwork=lapack::SVDRealWorkSize(colPinv_.Height(), colPinv_.Width());
                std::vector<Scalar> svdWork(lsvdwork);
                std::vector<Real> svdRealWork(lrsvdwork);
                lapack::AdjointPseudoInverse
                ( colPinv_.Height(), colPinv_.Width(), colPinv_.Buffer(), colPinv_.LDim(),
                  &singularValues[0], &U[0], colPinv_.Height(), &VH[0], rank,
                  &svdWork[0], lsvdwork, &svdRealWork[0]);

                blas::Gemm
                ( 'N', 'N', colUSqr_.Height(), colPinv_.Width(), colUSqr_.Width(),
                 Scalar(1), colUSqr_.LockedBuffer(), colUSqr_.LDim(),
                            colPinv_.LockedBuffer(), colPinv_.LDim(),
                 Scalar(0), colPinv_.Buffer(), colPinv_.LDim() );
                
                blas::Gemm
                ( 'N', 'C', colPinv_.Height(), colOmegaT.Height(), colPinv_.Width(),
                 Scalar(1), colPinv_.LockedBuffer(), colPinv_.LDim(),
                            colOmegaT.LockedBuffer(), colOmegaT.LDim(),
                 Scalar(0), BL_.Buffer(), BL_.LDim() );

            }
                                                                     
            if( rowUSqr_.Height() > 0 )
            {
                lapack::EVD
                ( 'V', 'U', rowUSqr_.Height(), 
                            rowUSqr_.Buffer(), rowUSqr_.LDim(),
                            &rowUSqrEig_[0],
                            &evdWork[0],     lwork,
                            &evdIntWork[0],  liwork,
                            &evdRealWork[0], lrwork );

                Real maxEig;
                if( rowUSqrEig_[rowUSqrEig_.size()-1]>0 )
                    maxEig=sqrt( rowUSqrEig_[rowUSqrEig_.size()-1] );
                else
                    maxEig=0;

                for(int j=0; j<rowUSqrEig_.size(); j++)
                    if( rowUSqrEig_[j] > error*maxEig*rowUSqrEig_.size() )
                    {
                        Real sqrteig=sqrt(rowUSqrEig_[j]);
                        for(int i=0; i<rowUSqr_.Height(); ++i)
                            rowUSqr_.Set(i,j,rowUSqr_.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<rowUSqr_.Height(); ++i)
                            rowUSqr_.Set(i,j,Scalar(0));
                    }
                
                blas::Gemm
                ( 'C', 'N', rowUSqr_.Height(), rowPinv_.Width(), rowUSqr_.Width(),
                 Scalar(1), rowUSqr_.LockedBuffer(), rowUSqr_.LDim(),
                            rowPinv_.LockedBuffer(), rowPinv_.LDim(),
                 Scalar(0), rowPinv_.Buffer(), rowPinv_.LDim() );

                int rank=std::min(rowPinv_.Height(), rowPinv_.Width());
                std::vector<Real> singularValues(rank);
                std::vector<Scalar> U(rank*rowPinv_.Height()), VH(rank*rowPinv_.Width());
                int lsvdwork=lapack::SVDWorkSize(rowPinv_.Height(), rowPinv_.Width());
                int lrsvdwork=lapack::SVDRealWorkSize(rowPinv_.Height(), rowPinv_.Width());
                std::vector<Scalar> svdWork(lsvdwork);
                std::vector<Real> svdRealWork(lrsvdwork);
                lapack::AdjointPseudoInverse
                ( rowPinv_.Height(), rowPinv_.Width(), rowPinv_.Buffer(), rowPinv_.LDim(),
                  &singularValues[0], &U[0], rowPinv_.Height(), &VH[0], rank,
                  &svdWork[0], lsvdwork, &svdRealWork[0]);

                blas::Gemm
                ( 'N', 'N', rowUSqr_.Height(), rowPinv_.Width(), rowUSqr_.Width(),
                 Scalar(1), rowUSqr_.LockedBuffer(), rowUSqr_.LDim(),
                            rowPinv_.LockedBuffer(), rowPinv_.LDim(),
                 Scalar(0), BR_.Buffer(), BR_.LDim() );
                
            }
/*//Print
if(level_==3 && block_.type==LOW_RANK)
std::cout << "MaxEig before pass: " << *std::max_element( USqrEig_.begin(), USqrEig_.end()) << " " << *std::max_element( VSqrEig_.begin(), VSqrEig_.end()) << std::endl;
*/
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
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatFHHCompressBroadcastsCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressBroadcastsPack
    ( buffer, offsetscopy, startLevel, endLevel );

    MultiplyHMatFHHCompressTreeBroadcasts( buffer, sizes );
    
    MultiplyHMatFHHCompressBroadcastsUnpack
    ( buffer, offsets, startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcastsCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressBroadcastsCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( BL_.Height()>0 )
            sizes[level_]+=BL_.Height()*BL_.Width();
        if( BR_.Height()>0 )
            sizes[level_]+=BR_.Height()*BR_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressBroadcastsPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( BL_.Height() > 0 )
            {
                int size = BL_.Height()*BL_.Width();
                MemCopy( &buffer[offsets[level_]], BL_.LockedBuffer(), size );
                offsets[level_] += size;
            }
            if( BR_.Height() > 0 )
            {
                int size = BR_.Height()*BR_.Width();
                MemCopy( &buffer[offsets[level_]], BR_.LockedBuffer(), size );
                offsets[level_] += size;
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
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressBroadcastsUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( BL_.Height() > 0 )                                  
        { 
            int size = BL_.Height()*BL_.Width();
            MemCopy( BL_.Buffer(), &buffer[offsets[level_]], size );
            offsets[level_] += size;
        }
        if( BR_.Height() > 0 )
        {                                                      
            int size = BR_.Height()*BR_.Width();
            MemCopy( BR_.Buffer(), &buffer[offsets[level_]], size );
            offsets[level_] += size;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatFHHCompressPostcompute
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatFHHCompressPostcompute");
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
                    node.Child(t,s).MultiplyHMatFHHCompressPostcompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )                                  
        { 
            colXMap_.Clear();
            colXMap_.Set( 0, new Dense<Scalar>( LocalHeight(), BL_.Width()));
            Dense<Scalar> &U = colXMap_.Get( 0 );
            blas::Gemm
            ('N', 'N', colU_.Height(), BL_.Width(), colU_.Width(), 
             Scalar(1), colU_.LockedBuffer(), colU_.LDim(),
                        BL_.LockedBuffer(), BL_.LDim(),
             Scalar(0), U.Buffer(),         U.LDim() );
        }
        if( inSourceTeam_ )
        {                                                      
            rowXMap_.Clear();
            rowXMap_.Set( 0, new Dense<Scalar>( LocalWidth(), BR_.Width()));
            Dense<Scalar> &U = rowXMap_.Get( 0 );
            blas::Gemm
            ('N', 'N', rowU_.Height(), BR_.Width(), rowU_.Width(), 
             Scalar(1), rowU_.LockedBuffer(), rowU_.LDim(),
                        BR_.LockedBuffer(), BR_.LDim(),
             Scalar(0), U.Buffer(),         U.LDim() );
        }
        break;
    }
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
        colU_.Resize(0,0,1);
        rowU_.Resize(0,0,1);
        colPinv_.Resize(0,0,1);
        rowPinv_.Resize(0,0,1);
        colUSqr_.Resize(0,0,1);
        rowUSqr_.Resize(0,0,1);
        colUSqrEig_.resize(0);
        rowUSqrEig_.resize(0);
        BL_.Resize(0,0,1);
        BR_.Resize(0,0,1);
    }
    default:
        break;
    }
}

} // namespace dmhm
