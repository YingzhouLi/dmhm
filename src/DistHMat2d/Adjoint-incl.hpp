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
DistHMat2d<Scalar>::Adjoint()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Adjoint");
#endif
    // This requires communication and is not yet written
    throw std::logic_error("DistHMat2d::Adjoint is not yet written");
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointFrom( const DistHMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointFrom");
#endif
    AdjointCopy( B );
    AdjointPassData( B );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointCopy( const DistHMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointCopy");
#endif
    DistHMat2d<Scalar>& A = *this;

    A.numLevels_ = B.numLevels_;
    A.maxRank_ = B.maxRank_;
    A.targetOffset_ = B.sourceOffset_;
    A.sourceOffset_ = B.targetOffset_;
    A.stronglyAdmissible_ = B.stronglyAdmissible_;

    A.xSizeTarget_ = B.xSizeSource_;
    A.ySizeTarget_ = B.ySizeSource_;
    A.xSizeSource_ = B.xSizeTarget_;
    A.ySizeSource_ = B.ySizeTarget_;

    A.xTarget_ = B.xSource_;
    A.yTarget_ = B.ySource_;
    A.xSource_ = B.xTarget_;
    A.ySource_ = B.yTarget_;

    A.teams_ = B.teams_;
    A.level_ = B.level_;
    A.inTargetTeam_ = B.inSourceTeam_;
    A.inSourceTeam_ = B.inTargetTeam_;
    A.targetRoot_ = B.sourceRoot_;
    A.sourceRoot_ = B.targetRoot_;

    A.block_.Clear();
    A.block_.type = B.block_.type;

    switch( B.block_.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        A.block_.data.N = A.NewNode();
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int j=0; j<16; ++j )
            nodeA.children[j] = new DistHMat2d<Scalar>;

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointCopy( nodeB.Child(s,t) );
        break;
    }
    case DIST_LOW_RANK:
    {
        A.block_.data.DF = new DistLowRank;
        DistLowRank& DFA = *A.block_.data.DF;
        const DistLowRank& DFB = *B.block_.data.DF;

        DFA.rank = DFB.rank;
        if( B.inTargetTeam_ )
            hmat_tools::Conjugate( DFB.ULocal, DFA.VLocal );
        if( B.inSourceTeam_ )
            hmat_tools::Conjugate( DFB.VLocal, DFA.ULocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        A.block_.data.SF = new SplitLowRank;
        SplitLowRank& SFA = *A.block_.data.SF;
        const SplitLowRank& SFB = *B.block_.data.SF;

        SFA.rank = SFB.rank;
        hmat_tools::Conjugate( SFB.D, SFA.D );
        break;
    }
    case LOW_RANK:
    {
        A.block_.data.F = new LowRank<Scalar>;
        hmat_tools::Adjoint( *B.block_.data.F, *A.block_.data.F );
        break;
    }
    case SPLIT_DENSE:
    {
        A.block_.data.SD = new SplitDense;
        break;
    }
    case DENSE:
    {
        A.block_.data.D = new Dense<Scalar>;
        hmat_tools::Adjoint( *B.block_.data.D, *A.block_.data.D );
        break;
    }
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointPassData( const DistHMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointPassData");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    AdjointPassDataCount
    ( B, sendSizes, recvSizes );

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
    std::vector<Scalar> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    AdjointPassDataPack
    ( B, sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
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
    std::vector<mpi::Request> sendRequests( numSends );
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
    AdjointPassDataUnpack( B, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::AdjointPassDataCount
( const DistHMat2d<Scalar>& B,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointPassDataCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;
    const DistHMat2d<Scalar>& A = *this;

    switch( B.block_.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataCount
                ( nodeB.Child(s,t), sendSizes, recvSizes );
        break;
    }
    case SPLIT_DENSE:
    {
        if( B.inSourceTeam_ )
            AddToMap( sendSizes, B.targetRoot_, B.Height()*B.Width() );
        if( A.inSourceTeam_ )
            AddToMap( recvSizes, A.targetRoot_, A.Height()*B.Width() );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::AdjointPassDataPack
( const DistHMat2d<Scalar>& B,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointPassDataPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;
    const DistHMat2d<Scalar>& A = *this;

    switch( B.block_.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataPack
                ( nodeB.Child(s,t), buffer, offsets );
        break;
    }
    case SPLIT_DENSE:
    {
        if( B.inSourceTeam_ )
        {
            SplitDense& SD = *B.block_.data.SD;
            std::memcpy
            ( &buffer[offsets[B.targetRoot_]], SD.D.LockedBuffer(),
              SD.D.Height()*SD.D.Width()*sizeof(Scalar) );
            offsets[B.targetRoot_] += SD.D.Height()*SD.D.Width();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::AdjointPassDataUnpack
( const DistHMat2d<Scalar>& B,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointPassDataUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;
    DistHMat2d<Scalar>& A = *this;

    switch( B.block_.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataUnpack
                ( nodeB.Child(s,t), buffer, offsets );
        break;
    }
    case SPLIT_DENSE:
    {
        if( A.inSourceTeam_ )
        {
            SplitDense& SD = *A.block_.data.SD;
            SD.D.Resize(A.Height(), A.Width());
            const int m = A.Width();
            const int n = A.Height();
            const int LDim = A.Width();
            const int DLDim = SD.D.LDim();
            Scalar* RESTRICT ABuffer = SD.D.Buffer();
            for( int j=0; j<n; ++j)
                for( int i=0; i<m; ++i)
                    ABuffer[j+i*DLDim] = 
                        Conj(buffer[offsets[A.targetRoot_]+i+j*LDim]);
            offsets[A.targetRoot_] += m*n;
        }
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
