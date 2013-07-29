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
DistHMat3d<Scalar>::MultiplyHMatFHHPrecompute
( Scalar alpha, DistHMat3d<Scalar>& B,
                DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPrecompute");
#endif
    DistHMat3d<Scalar>& A = *this;
    if(( !A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    C.colFHHContextMap_.Set( key, new MultiplyDenseContext );
                    MultiplyDenseContext& colContext =
                        C.colFHHContextMap_.Get( key );
                    colContext.numRhs = sampleRank;
                    C.colXMap_.Set
                    ( key, new Dense<Scalar>( A.LocalHeight(), sampleRank ) );
                    Dense<Scalar>& colX = C.colXMap_.Get( key );
                    colX.Init();
                    A.MultiplyDenseInitialize( colContext, sampleRank );
                    A.MultiplyDensePrecompute
                    ( colContext, alpha, B.colT_, colX );

                    C.rowFHHContextMap_.Set( key, new MultiplyDenseContext );
                    MultiplyDenseContext& rowContext =
                        C.rowFHHContextMap_.Get( key );
                    rowContext.numRhs = sampleRank;
                    C.rowXMap_.Set
                    ( key, new Dense<Scalar>( B.LocalWidth(), sampleRank ) );
                    Dense<Scalar>& rowX = C.rowXMap_.Get( key );
                    rowX.Init();
                    B.AdjointMultiplyDenseInitialize( rowContext, sampleRank );
                    B.AdjointMultiplyDensePrecompute
                    ( rowContext, Conj(alpha), A.rowT_, rowX );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHSums
( Scalar alpha, DistHMat3d<Scalar>& B,
                DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHSums");
#endif
    DistHMat3d<Scalar>& A = *this;

    // Compute the message sizes for each reduce
    const unsigned numTeamLevels = teams_->NumLevels();
    const unsigned numReduces = numTeamLevels-1;
    Vector<int> sizes( numReduces, 0 );
    A.MultiplyHMatFHHSumsCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    Vector<Scalar> buffer( totalSize );
    Vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    Vector<int> offsetsCopy = offsets;
    A.MultiplyHMatFHHSumsPack
    ( B, C, buffer, offsetsCopy,
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the reduces with log2(p) messages
    A.teams_->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatFHHSumsUnpack
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHSumsCount
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHSumsCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseSumsCount( sizes, sampleRank );
                    B.TransposeMultiplyDenseSumsCount( sizes, sampleRank );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHSumsPack
( DistHMat3d<Scalar>& B,
  DistHMat3d<Scalar>& C,
  Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHSumsPack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseSumsPack
                    ( C.colFHHContextMap_.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseSumsPack
                    ( C.rowFHHContextMap_.Get( key ), buffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHSumsUnpack
( DistHMat3d<Scalar>& B,
  DistHMat3d<Scalar>& C,
  const Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHSumsUnpack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseSumsUnpack
                    ( C.colFHHContextMap_.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseSumsUnpack
                    ( C.rowFHHContextMap_.Get( key ), buffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHSumsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHPassData
( Scalar alpha, DistHMat3d<Scalar>& B,
                DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPassData");
#endif
    DistHMat3d<Scalar>& A = *this;

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFHHPassDataCount
    ( B, C, sendSizes, recvSizes,
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Compute the offsets
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    Vector<Scalar> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatFHHPassDataPack
    ( B, C, sendBuffer, offsets,
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    Vector<mpi::Request> recvRequests( numRecvs );
    Vector<Scalar> recvBuffer( totalRecvSize );
    int offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    mpi::Barrier( comm );

    // Start the non-blocking sends
    const int numSends = sendSizes.size();
    Vector<mpi::Request> sendRequests( numSends );
    offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    mpi::WaitAll( numRecvs, &recvRequests[0] );
    A.MultiplyHMatFHHPassDataUnpack
    ( B, C, recvBuffer, recvOffsets,
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHPassDataCount
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPassDataCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
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
        {
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                    B.TransposeMultiplyDensePassDataCount
                    ( sendSizes, recvSizes, sampleRank );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              sendSizes, recvSizes, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
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
DistHMat3d<Scalar>::MultiplyHMatFHHPassDataPack
( DistHMat3d<Scalar>& B,
  DistHMat3d<Scalar>& C,
  Vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPassDataPack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();

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
        {
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDensePassDataPack
                    ( C.colFHHContextMap_.Get( key ), sendBuffer, offsets );
                    B.TransposeMultiplyDensePassDataPack
                    ( C.rowFHHContextMap_.Get( key ),
                      A.rowT_, sendBuffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              sendBuffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
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
DistHMat3d<Scalar>::MultiplyHMatFHHPassDataUnpack
( DistHMat3d<Scalar>& B,
  DistHMat3d<Scalar>& C,
  const Vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPassDataUnpack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();

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
        {
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDensePassDataUnpack
                    ( C.colFHHContextMap_.Get( key ), recvBuffer, offsets );
                    B.TransposeMultiplyDensePassDataUnpack
                    ( C.rowFHHContextMap_.Get( key ), recvBuffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPassDataUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              recvBuffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
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
DistHMat3d<Scalar>::MultiplyHMatFHHBroadcasts
( Scalar alpha, DistHMat3d<Scalar>& B,
                DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHBroadcasts");
#endif
    DistHMat3d<Scalar>& A = *this;

    // Compute the message sizes for each broadcast
    const unsigned numTeamLevels = teams_->NumLevels();
    const unsigned numBroadcasts = numTeamLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    A.MultiplyHMatFHHBroadcastsCount
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    Vector<Scalar> buffer( totalSize );
    Vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    Vector<int> offsetsCopy = offsets;
    A.MultiplyHMatFHHBroadcastsPack
    ( B, C, buffer, offsetsCopy,
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the broadcasts with log2(p) messages
    A.teams_->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers
    A.MultiplyHMatFHHBroadcastsUnpack
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHBroadcastsCount
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<int>& sizes,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHBroadcastsCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int sampleRank = SampleRank( C.MaxRank() );
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseBroadcastsCount( sizes, sampleRank );
                    B.TransposeMultiplyDenseBroadcastsCount
                    ( sizes, sampleRank );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsCount
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHBroadcastsPack
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHBroadcastsPack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseBroadcastsPack
                    ( C.colFHHContextMap_.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseBroadcastsPack
                    ( C.rowFHHContextMap_.Get( key ), buffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsPack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHBroadcastsUnpack
( DistHMat3d<Scalar>& B,
  DistHMat3d<Scalar>& C,
  const Vector<Scalar>& buffer, Vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHBroadcastsUnpack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    A.MultiplyDenseBroadcastsUnpack
                    ( C.colFHHContextMap_.Get( key ), buffer, offsets );
                    B.TransposeMultiplyDenseBroadcastsUnpack
                    ( C.rowFHHContextMap_.Get( key ), buffer, offsets );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHBroadcastsUnpack
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHPostcompute
( Scalar alpha, DistHMat3d<Scalar>& B,
                DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPostcompute");
#endif
    DistHMat3d<Scalar>& A = *this;

    A.MultiplyHMatFHHPostcomputeC
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
    C.MultiplyHMatFHHPostcomputeCCleanup( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHPostcomputeC
( Scalar alpha, const DistHMat3d<Scalar>& B,
                      DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPostcomputeC");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
        {
            if( admissibleC )
            {
                if( A.level_ >= startLevel && A.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    // Finish computing A B Omega1
                    A.MultiplyDensePostcompute
                    ( C.colFHHContextMap_.Get( key ),
                      alpha, B.colT_, C.colXMap_.Get( key ) );

                    // Finish computing B' A' Omega2
                    B.AdjointMultiplyDensePostcompute
                    ( C.rowFHHContextMap_.Get( key ),
                      Conj(alpha), A.rowT_, C.rowXMap_.Get( key ) );
                }
            }
            else if( A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHPostcomputeC
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHPostcomputeCCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHPostcomputeCCleanup");
#endif
    DistHMat3d<Scalar>& C = *this;
    C.colFHHContextMap_.Clear();
    C.rowFHHContextMap_.Clear();

    switch( C.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        if( C.level_+1 < endLevel )
        {
            Node& nodeC = *C.block_.data.N;
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    nodeC.Child(t,s).MultiplyHMatFHHPostcomputeCCleanup
                    ( startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHFinalize
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalize");
#endif
    const DistHMat3d<Scalar>& A = *this;

    const int r = SampleRank( C.MaxRank() );
    const unsigned numTeamLevels = C.teams_->NumLevels();
    Vector<int> numQRs(numTeamLevels,0),
                     numTargetFHH(numTeamLevels,0),
                     numSourceFHH(numTeamLevels,0);
    C.MultiplyHMatFHHFinalizeCounts
    ( numQRs, numTargetFHH, numSourceFHH, startLevel, endLevel );

    // Set up the space for the packed 2r x r matrices and taus.
    int numTotalQRs=0, numQRSteps=0, qrTotalSize=0, tauTotalSize=0;
    Vector<int> XOffsets(numTeamLevels), halfHeightOffsets(numTeamLevels),
                     qrOffsets(numTeamLevels), tauOffsets(numTeamLevels);
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        mpi::Comm team = C.teams_->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );

        XOffsets[teamLevel] = numTotalQRs;
        halfHeightOffsets[teamLevel] = 2*numQRSteps;
        qrOffsets[teamLevel] = qrTotalSize;
        tauOffsets[teamLevel] = tauTotalSize;

        numTotalQRs += numQRs[teamLevel];
        numQRSteps += numQRs[teamLevel]*log2TeamSize;
        qrTotalSize += numQRs[teamLevel]*log2TeamSize*(r*r+r);
        tauTotalSize += numQRs[teamLevel]*(log2TeamSize+1)*r;
    }

    Vector<Dense<Scalar>*> Xs( numTotalQRs );
    Vector<int> halfHeights( 2*numQRSteps );
    Vector<Scalar> qrBuffer( qrTotalSize ), tauBuffer( tauTotalSize ),
                        qrWork( lapack::QRWorkSize( r ) );

    // Form our contributions to Omega2' (alpha A B Omega1) updates here,
    // before we overwrite the colXMap_ and rowXMap_ results.
    // The distributed summations will not occur until after the parallel
    // QR factorizations.
    Vector<int> leftOffsets(numTeamLevels), middleOffsets(numTeamLevels),
                     rightOffsets(numTeamLevels);
    int totalAllReduceSize = 0;
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        leftOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;

        middleOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numTargetFHH[teamLevel]*r*r;

        rightOffsets[teamLevel] = totalAllReduceSize;
        totalAllReduceSize += numSourceFHH[teamLevel]*r*r;
    }

    // Compute the local contributions to the middle updates,
    // Omega2' (alpha A B Omega1)
    Vector<Scalar> allReduceBuffer( totalAllReduceSize );
    {
        Vector<int> middleOffsetsCopy = middleOffsets;

        A.MultiplyHMatFHHFinalizeMiddleUpdates
        ( B, C, allReduceBuffer, middleOffsetsCopy,
          startLevel, endLevel, startUpdate, endUpdate, 0 );
    }

    // Perform the large local QR's and pack into the QR buffer as appropriate
    {
        Vector<int> XOffsetsCopy, tauOffsetsCopy;
        XOffsetsCopy = XOffsets;
        tauOffsetsCopy = tauOffsets;

        C.MultiplyHMatFHHFinalizeLocalQR
        ( Xs, XOffsetsCopy, tauBuffer, tauOffsetsCopy, qrWork,
          startLevel, endLevel );
    }

    C.MultiplyHMatParallelQR
    ( numQRs, Xs, XOffsets, halfHeights, halfHeightOffsets,
      qrBuffer, qrOffsets, tauBuffer, tauOffsets, qrWork );

    // Explicitly form the Q's
    Dense<Scalar> Z( 2*r, r );
    for( unsigned teamLevel=0; teamLevel<numTeamLevels; ++teamLevel )
    {
        mpi::Comm team = C.teams_->Team( teamLevel );
        const unsigned teamSize = mpi::CommSize( team );
        const unsigned log2TeamSize = Log2( teamSize );
        const unsigned teamRank = mpi::CommRank( team );

        Dense<Scalar>** XLevel = &Xs[XOffsets[teamLevel]];
        const int* halfHeightsLevel =
            &halfHeights[halfHeightOffsets[teamLevel]];
        const Scalar* qrLevel = &qrBuffer[qrOffsets[teamLevel]];
        const Scalar* tauLevel = &tauBuffer[tauOffsets[teamLevel]];

        for( int k=0; k<numQRs[teamLevel]; ++k )
        {
            Dense<Scalar>& X = *XLevel[k];
            const int* halfHeightsPiece = &halfHeightsLevel[k*log2TeamSize*2];
            const Scalar* qrPiece = &qrLevel[k*log2TeamSize*(r*r+r)];
            const Scalar* tauPiece = &tauLevel[k*(log2TeamSize+1)*r];

           if( log2TeamSize > 0 )
           {
                const int* lastHalfHeightsStage =
                    &halfHeightsPiece[(log2TeamSize-1)*2];
                const Scalar* lastQRStage = &qrPiece[(log2TeamSize-1)*(r*r+r)];
                const Scalar* lastTauStage = &tauPiece[log2TeamSize*r];

                const int sLast = lastHalfHeightsStage[0];
                const int tLast = lastHalfHeightsStage[1];

                // Form the identity matrix in the top r x r submatrix
                // of a zeros (sLast+tLast) x r matrix.
                Z.Resize( sLast+tLast, r );
                Z.Init();
                for( int j=0; j<std::min(sLast+tLast,r); ++j )
                    Z.Set(j,j,Scalar(1) );

                // Backtransform the last stage
                qrWork.Resize( r );
                hmat_tools::ApplyPackedQFromLeft
                ( r, sLast, tLast, lastQRStage, lastTauStage, Z, &qrWork[0] );

                // Take care of the middle stages before handling the large
                // original stage.
                int sPrev=sLast, tPrev=tLast;
                for( int commStage=log2TeamSize-2; commStage>=0; --commStage )
                {
                    const int sCurr = halfHeightsPiece[commStage*2];
                    const int tCurr = halfHeightsPiece[commStage*2+1];
                    Z.Resize( sCurr+tCurr, r );
                    Z.Init();

                    const bool rootOfPrevStage =
                        !(teamRank & (1u<<(commStage+1)));
                    if( rootOfPrevStage )
                    {
                        // Zero the bottom half of Z
                        for( int j=0; j<r; ++j )
                            MemZero( Z.Buffer(sCurr,j), tCurr );
                    }
                    else
                    {
                        // Move the bottom part to the top part and zero the
                        // bottom
                        for( int j=0; j<r; ++j )
                        {
                            MemCopy
                            ( Z.Buffer(0,j), Z.LockedBuffer(sCurr,j), tCurr );
                            MemZero( Z.Buffer(sCurr,j), tCurr );
                        }
                    }
                    hmat_tools::ApplyPackedQFromLeft
                    ( r, sCurr, tCurr, &qrPiece[commStage*(r*r+r)],
                      &tauPiece[(commStage+1)*r], Z, &qrWork[0] );

                    sPrev = sCurr;
                    tPrev = tCurr;
                }

                // Take care of the original stage. Do so by forming Y := X,
                // then zeroing X and placing our piece of Z at its top.
                const int m = X.Height();
                Dense<Scalar> Y;
                hmat_tools::Copy( X, Y );
                X.Init();
                const bool rootOfPrevStage = !(teamRank & 0x1);
                if( rootOfPrevStage )
                {
                    // Copy the first sPrev rows of the top half of Z into
                    // the top of X
                    for( int j=0; j<r; ++j )
                        MemCopy( X.Buffer(0,j), Z.LockedBuffer(0,j), sPrev );
                }
                else
                {
                    // Copy the first tPrev rows of the bottom part of Z into
                    // the top of X
                    for( int j=0; j<r; ++j )
                        MemCopy
                        ( X.Buffer(0,j), Z.LockedBuffer(sPrev,j), tPrev );
                }
                qrWork.Resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, std::min(m,r),
                  Y.LockedBuffer(), Y.LDim(), &tauPiece[0],
                  X.Buffer(),       X.LDim(), &qrWork[0], qrWork.Size() );
            }
            else // this team only contains one process
            {
                // Make a copy of X and then form the left part of identity.
                const int m = X.Height();
                Dense<Scalar> Y;
                hmat_tools::Copy( X, Y );
                X.Init();
                for( int j=0; j<std::min(m,r); ++j )
                    X.Set(j,j,Scalar(1));
                // Backtransform the last stage
                qrWork.Resize( lapack::ApplyQWorkSize('L',m,r) );
                lapack::ApplyQ
                ( 'L', 'N', m, r, std::min(m,r),
                  Y.LockedBuffer(), Y.LDim(), &tauPiece[0],
                  X.Buffer(),       X.LDim(), &qrWork[0], qrWork.Size() );
            }
        }
    }
    XOffsets.Clear();
    qrOffsets.Clear(); qrBuffer.Clear();
    tauOffsets.Clear(); tauBuffer.Clear();
    qrWork.Clear();
    Z.Clear();

    // Form our local contributions to Q1' Omega2 and Q2' Omega1
    {
        Vector<int> leftOffsetsCopy, rightOffsetsCopy;
        leftOffsetsCopy = leftOffsets;
        rightOffsetsCopy = rightOffsets;

        A.MultiplyHMatFHHFinalizeOuterUpdates
        ( B, C, allReduceBuffer, leftOffsetsCopy, rightOffsetsCopy,
          startLevel, endLevel, startUpdate, endUpdate, 0 );
    }

    // Perform a custom AllReduce on the buffers to finish forming
    // Q1' Omega2, Omega2' (alpha A B Omega1), and Q2' Omega1
    {
        // Generate offsets and sizes for each entire level
        const unsigned numAllReduces = numTeamLevels-1;
        Vector<int> sizes(numAllReduces);
        for( unsigned teamLevel=0; teamLevel<numAllReduces; ++teamLevel )
            sizes[teamLevel] = (2*numTargetFHH[teamLevel]+
                                  numSourceFHH[teamLevel])*r*r;

        A.teams_->TreeSums( allReduceBuffer, sizes );
    }

    // Finish forming the low-rank approximation
    Vector<Scalar> U( r*r ), VH( r*r ),
                        svdWork( lapack::SVDWorkSize(r,r) );
    Vector<Real> singularValues( r ),
                      svdRealWork( lapack::SVDRealWorkSize(r,r) );
    A.MultiplyHMatFHHFinalizeFormLowRank
    ( B, C, allReduceBuffer, leftOffsets, middleOffsets, rightOffsets,
      singularValues, U, VH, svdWork, svdRealWork,
      startLevel, endLevel, startUpdate, endUpdate, 0 );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHFinalizeCounts
( Vector<int>& numQRs,
  Vector<int>& numTargetFHH, Vector<int>& numSourceFHH,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalizeCounts");
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
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    node.Child(t,s).MultiplyHMatFHHFinalizeCounts
                    ( numQRs, numTargetFHH, numSourceFHH,
                      startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
        // TODO: Think about avoiding the expensive F += H H proceduce in the
        //       case where there is already a dense update. We could simply
        //       add H H onto the dense update.
        if( level_ >= startLevel && level_ < endLevel )
        {
            if( inTargetTeam_ )
            {
                const unsigned teamLevel = teams_->TeamLevel(level_);
                numQRs[teamLevel] += colXMap_.Size();
                numTargetFHH[teamLevel] += colXMap_.Size();
            }
            if( inSourceTeam_ )
            {
                const unsigned teamLevel = teams_->TeamLevel(level_);
                numQRs[teamLevel] += rowXMap_.Size();
                numSourceFHH[teamLevel] += rowXMap_.Size();
            }
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFHHFinalizeMiddleUpdates
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<Scalar>& allReduceBuffer,
        Vector<int>& middleOffsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalizeMiddleUpdates");
#endif
    const DistHMat3d<Scalar>& A = *this;
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
                if( C.inTargetTeam_ &&
                    C.level_ >= startLevel && C.level_ < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    // Handle the middle update, Omega2' (alpha A B Omega1)
                    const int key = A.sourceOffset_;
                    const Dense<Scalar>& X = C.colXMap_.Get( key );
                    const Dense<Scalar>& Omega2 = A.rowOmega_;
                    const unsigned teamLevel = C.teams_->TeamLevel(C.level_);
                    Scalar* middleUpdate =
                        &allReduceBuffer[middleOffsets[teamLevel]];
                    blas::Gemm
                    ( 'C', 'N', rank, rank, X.Height(),
                      Scalar(1), Omega2.LockedBuffer(), Omega2.LDim(),
                                 X.LockedBuffer(),      X.LDim(),
                      Scalar(0), middleUpdate,          rank );
                    middleOffsets[teamLevel] += rank*rank;
                }
            }
            else if( C.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHFinalizeMiddleUpdates
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              allReduceBuffer, middleOffsets,
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
DistHMat3d<Scalar>::MultiplyHMatFHHFinalizeLocalQR
( Vector<Dense<Scalar>*>& Xs, Vector<int>& XOffsets,
  Vector<Scalar>& tauBuffer, Vector<int>& tauOffsets,
  Vector<Scalar>& qrWork,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalizeLocalQR");
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
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    node.Child(t,s).MultiplyHMatFHHFinalizeLocalQR
                    ( Xs, XOffsets, tauBuffer, tauOffsets, qrWork,
                      startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ >= startLevel && level_ < endLevel )
        {
            mpi::Comm team = teams_->Team( level_ );
            const unsigned teamLevel = teams_->TeamLevel(level_);
            const int log2TeamSize = Log2( mpi::CommSize( team ) );
            const int r = SampleRank( MaxRank() );

            if( inTargetTeam_ )
            {
                colXMap_.ResetIterator();
                const unsigned numEntries = colXMap_.Size();
                for( unsigned i=0; i<numEntries; ++i,colXMap_.Increment() )
                {
                    Dense<Scalar>& X = *colXMap_.CurrentEntry();
                    Xs[XOffsets[teamLevel]++] = &X;

                    lapack::QR
                    ( X.Height(), X.Width(), X.Buffer(), X.LDim(),
                      &tauBuffer[tauOffsets[teamLevel]],
                      &qrWork[0], qrWork.Size() );
                    tauOffsets[teamLevel] += (log2TeamSize+1)*r;
                }
            }
            if( inSourceTeam_ )
            {
                rowXMap_.ResetIterator();
                const int numEntries = rowXMap_.Size();
                for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
                {
                    Dense<Scalar>& X = *rowXMap_.CurrentEntry();
                    Xs[XOffsets[teamLevel]++] = &X;

                    lapack::QR
                    ( X.Height(), X.Width(), X.Buffer(), X.LDim(),
                      &tauBuffer[tauOffsets[teamLevel]],
                      &qrWork[0], qrWork.Size() );
                    tauOffsets[teamLevel] += (log2TeamSize+1)*r;
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
DistHMat3d<Scalar>::MultiplyHMatFHHFinalizeOuterUpdates
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<Scalar>& allReduceBuffer,
        Vector<int>& leftOffsets,
        Vector<int>& rightOffsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalizeOuterUpdates");
#endif
    const DistHMat3d<Scalar>& A = *this;
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
                    const int key = A.sourceOffset_;
                    const unsigned teamLevel = C.teams_->TeamLevel(C.level_);
                    if( C.inTargetTeam_ )
                    {
                        // Handle the left update, Q1' Omega2
                        const Dense<Scalar>& Q1 = C.colXMap_.Get( key );
                        const Dense<Scalar>& Omega2 = A.rowOmega_;
                        Scalar* leftUpdate =
                            &allReduceBuffer[leftOffsets[teamLevel]];
                        blas::Gemm
                        ( 'C', 'N', Q1.Width(), rank, A.LocalHeight(),
                          Scalar(1), Q1.LockedBuffer(),     Q1.LDim(),
                                     Omega2.LockedBuffer(), Omega2.LDim(),
                          Scalar(0), leftUpdate,            rank );
                        leftOffsets[teamLevel] += rank*rank;
                    }
                    if( C.inSourceTeam_ )
                    {
                        // Handle the right update, Q2' Omega1
                        const Dense<Scalar>& Q2 = C.rowXMap_.Get( key );
                        const Dense<Scalar>& Omega1 = B.colOmega_;
                        Scalar* rightUpdate =
                            &allReduceBuffer[rightOffsets[teamLevel]];

                        blas::Gemm
                        ( 'C', 'N', Q2.Width(), rank, B.LocalWidth(),
                          Scalar(1), Q2.LockedBuffer(),     Q2.LDim(),
                                     Omega1.LockedBuffer(), Omega1.LDim(),
                          Scalar(0), rightUpdate,           rank );
                        rightOffsets[teamLevel] += rank*rank;
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHFinalizeOuterUpdates
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              allReduceBuffer, leftOffsets, rightOffsets,
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
DistHMat3d<Scalar>::MultiplyHMatFHHFinalizeFormLowRank
( const DistHMat3d<Scalar>& B,
        DistHMat3d<Scalar>& C,
        Vector<Scalar>& allReduceBuffer,
        Vector<int>& leftOffsets,
        Vector<int>& middleOffsets,
        Vector<int>& rightOffsets,
        Vector<Real>& singularValues,
        Vector<Scalar>& U,
        Vector<Scalar>& VH,
        Vector<Scalar>& svdWork,
        Vector<Real>& svdRealWork,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFHHFinalizeFormLowRank");
#endif
    const DistHMat3d<Scalar>& A = *this;
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
                    const int key = A.sourceOffset_;
                    const unsigned teamLevel = C.teams_->TeamLevel(C.level_);
                    if( C.inTargetTeam_ )
                    {
                        // Form Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        // in the place of X.
                        Dense<Scalar>& X = C.colXMap_.Get( key );
                        Scalar* leftUpdate =
                            &allReduceBuffer[leftOffsets[teamLevel]];
                        const Scalar* middleUpdate =
                            &allReduceBuffer[middleOffsets[teamLevel]];

                        lapack::AdjointPseudoInverse
                        ( X.Width(), rank, leftUpdate, rank, &singularValues[0],
                          &U[0], rank, &VH[0], rank, &svdWork[0],
                          svdWork.Size(), &svdRealWork[0] );

                        // We can use the VH space to hold the product
                        // pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        blas::Gemm
                        ( 'N', 'N', X.Width(), rank, rank,
                          Scalar(1), leftUpdate,   rank,
                                     middleUpdate, rank,
                          Scalar(0), &VH[0],       rank );

                        // Q1 := X.
                        Dense<Scalar> Q1;
                        hmat_tools::Copy( X, Q1 );
                        X.Resize( X.Height(), rank );
                        X.Init();

                        // Form
                        // X := Q1 pinv(Q1' Omega2)' (Omega2' alpha A B Omega1)
                        blas::Gemm
                        ( 'N', 'N', Q1.Height(), rank, Q1.Width(),
                          Scalar(1), Q1.LockedBuffer(), Q1.LDim(),
                                     &VH[0],            rank,
                          Scalar(0), X.Buffer(),        X.LDim() );

                        leftOffsets[teamLevel] += rank*rank;
                        middleOffsets[teamLevel] += rank*rank;
                    }
                    if( C.inSourceTeam_ )
                    {
                        // Form Q2 pinv(Q2' Omega1)' or its conjugate
                        Dense<Scalar>& X = C.rowXMap_.Get( key );
                        Scalar* rightUpdate =
                            &allReduceBuffer[rightOffsets[teamLevel]];

                        lapack::AdjointPseudoInverse
                        ( X.Width(), rank, rightUpdate, rank,
                          &singularValues[0],
                          &U[0], rank, &VH[0], rank,
                          &svdWork[0], svdWork.Size(), &svdRealWork[0] );

                        // Q2 := X
                        Dense<Scalar> Q2;
                        hmat_tools::Copy( X, Q2 );
                        X.Resize( X.Height(), rank );
                        X.Init();

                        blas::Gemm
                        ( 'N', 'N', Q2.Height(), rank, Q2.Width(),
                          Scalar(1), Q2.LockedBuffer(), Q2.LDim(),
                                     rightUpdate,       rank,
                          Scalar(0), X.Buffer(),        X.LDim() );
                        hmat_tools::Conjugate( X );
                        rightOffsets[teamLevel] += rank*rank;
                    }
                }
            }
            else if( C.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFHHFinalizeFormLowRank
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              allReduceBuffer,
                              leftOffsets, middleOffsets, rightOffsets,
                              singularValues, U, VH, svdWork, svdRealWork,
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

} // namespace dmhm
