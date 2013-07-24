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
DistHMat3d<Scalar>::MultiplyHMatFormGhostRanks
( DistHMat3d<Scalar>& B ) 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFormGhostRanks");
#endif
    DistHMat3d<Scalar>& A = *this;

    // Count the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatFormGhostRanksCount( B, sendSizes, recvSizes );

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
    Vector<int> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatFormGhostRanksPack( B, sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = A.teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    Vector<mpi::Request> recvRequests( numRecvs );
    Vector<int> recvBuffer( totalRecvSize );
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
#ifndef RELEASE
    const int commRank = mpi::CommRank( comm );
    if( commRank == 0 )
    {
        std::cerr << "\n"
                  << "Forming ranks requires process 0 to send to "
                  << sendSizes.size() << " processes and recv from "
                  << recvSizes.size() << std::endl;
    }
#endif
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
    A.MultiplyHMatFormGhostRanksUnpack( B, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatFormGhostRanksCount
( const DistHMat3d<Scalar>& B,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFormGhostRanksCount");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( A.sourceRoot_ != B.targetRoot_ )
    {
        std::ostringstream s;
        s << "A.sourceRoot_=" << A.sourceRoot_ << ", B.targetRoot_="
          << B.targetRoot_ << ", level=" << A.level_;
        throw std::logic_error( s.str().c_str() );
    }
    if( !A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_ )
        return;

    mpi::Comm team = A.teams_->Team( A.level_ );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A.targetOffset_, A.sourceOffset_ ),
                       BOffsets( B.targetOffset_, B.sourceOffset_ );

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
            // Check if C will be admissible
            const bool admissibleC = A.Admissible
            ( B.xSource_, A.xTarget_, B.ySource_, A.yTarget_, B.zSource_, A.zTarget_ );

            if( !admissibleC )
            {
                // Recurse
                const Node& nodeA = *A.block_.data.N;        
                const Node& nodeB = *B.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksCount
                            ( nodeB.Child(r,s), sendSizes, recvSizes );
            }
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B.inSourceTeam_ &&
                A.targetRoot_ != B.targetRoot_ && 
                A.targetRoot_ != B.sourceRoot_ )
                AddToMap( sendSizes, A.targetRoot_+teamRank, 1 );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B.sourceRoot_+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        // Check if we need to send to B's source team
        if( A.inTargetTeam_ &&
            B.sourceRoot_ != A.targetRoot_ &&
            B.sourceRoot_ != A.sourceRoot_ )
            AddToMap( sendSizes, B.sourceRoot_+teamRank, 1 );

        switch( B.block_.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ &&
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                AddToMap( sendSizes, A.targetRoot_+teamRank, 1 );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B.sourceRoot_+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        AddToMap( recvSizes, A.targetRoot_+teamRank, 1 );

        switch( B.block_.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ && 
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                AddToMap( sendSizes, A.targetRoot_+teamRank, 1 );
            break;
        }
        default:
            break;
        }
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ && 
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                AddToMap( sendSizes, A.targetRoot_, 1 );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            AddToMap( recvSizes, B.sourceRoot_, 1 );
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
DistHMat3d<Scalar>::MultiplyHMatFormGhostRanksPack
( const DistHMat3d<Scalar>& B,
  Vector<int>& sendBuffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFormGhostRanksPack");
#endif
    const DistHMat3d<Scalar>& A = *this;
    if( !A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_ )
        return;

    mpi::Comm team = A.teams_->Team( A.level_ );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A.targetOffset_, A.sourceOffset_ ),
                       BOffsets( B.targetOffset_, B.sourceOffset_ );

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
            // Check if C will be admissible
            const bool admissibleC = A.Admissible
            ( B.xSource_, A.xTarget_, B.ySource_, A.yTarget_, B.zSource_, A.zTarget_ );

            if( !admissibleC )
            {
                // Recurse
                const Node& nodeA = *A.block_.data.N;        
                const Node& nodeB = *B.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksPack
                            ( nodeB.Child(r,s), sendBuffer, offsets );
            }
            break;
        }
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Test if we need to send to A's target team
            if( B.inSourceTeam_ &&
                A.targetRoot_ != B.targetRoot_ && 
                A.targetRoot_ != B.sourceRoot_ )
                sendBuffer[offsets[A.targetRoot_+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        // Check if we need to send to B's source team
        if( A.inTargetTeam_ && 
            B.sourceRoot_ != A.targetRoot_ &&
            B.sourceRoot_ != A.sourceRoot_ )
            sendBuffer[offsets[B.sourceRoot_+teamRank]++] = A.Rank();

        switch( B.block_.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ && 
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                sendBuffer[offsets[A.targetRoot_+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_LOW_RANK:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ && 
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                sendBuffer[offsets[A.targetRoot_+teamRank]++] = B.Rank();
            break;
        }
        default:
            break;
        }
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
        {
            // Check if we need to send to A's target team
            if( B.inSourceTeam_ && 
                A.targetRoot_ != B.targetRoot_ &&
                A.targetRoot_ != B.sourceRoot_ )
                sendBuffer[offsets[A.targetRoot_]++] = B.Rank();
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
DistHMat3d<Scalar>::MultiplyHMatFormGhostRanksUnpack
( DistHMat3d<Scalar>& B,
  const Vector<int>& recvBuffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatFormGhostRanksUnpack");
#endif
    DistHMat3d<Scalar>& A = *this;
    if( !A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_ )
        return;

    mpi::Comm team = A.teams_->Team( A.level_ );
    const int teamRank = mpi::CommRank( team );
    std::pair<int,int> AOffsets( A.targetOffset_, A.sourceOffset_ ),
                       BOffsets( B.targetOffset_, B.sourceOffset_ );

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
            // Check if C will be admissible
            const bool admissibleC = A.Admissible
            ( B.xSource_, A.xTarget_, B.ySource_, A.yTarget_, B.zSource_, A.zTarget_ );

            if( !admissibleC )
            {
                // Recurse
                Node& nodeA = *A.block_.data.N;        
                Node& nodeB = *B.block_.data.N;
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        for( int r=0; r<8; ++r )
                            nodeA.Child(t,r).MultiplyHMatFormGhostRanksUnpack
                            ( nodeB.Child(r,s), recvBuffer, offsets );
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B.sourceRoot_+teamRank]++] );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        switch( B.block_.type )
        {
        case DIST_LOW_RANK_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B.sourceRoot_+teamRank]++] );
            break;
        }
        default:
            break;
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    {
        A.SetGhostRank( recvBuffer[offsets[A.targetRoot_+teamRank]++] );
        break;
    }
    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        {
            B.SetGhostRank( recvBuffer[offsets[B.sourceRoot_]++] );
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

} // namespace dmhm
