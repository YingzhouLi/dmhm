/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

namespace dmhm {

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::Adjoint()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Adjoint");
#endif
    // This requires communication and is not yet written
    throw std::logic_error("DistQuasi2dHMat::Adjoint is not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::AdjointFrom( const DistQuasi2dHMat<Scalar>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointFrom");
#endif
    AdjointCopy( B );
    AdjointPassData( B );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::AdjointCopy( const DistQuasi2dHMat<Scalar>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointCopy");
#endif
    DistQuasi2dHMat<Scalar>& A = *this;

    A._numLevels = B._numLevels;
    A._maxRank = B._maxRank;
    A._targetOffset = B._sourceOffset;
    A._sourceOffset = B._targetOffset;
    A._stronglyAdmissible = B._stronglyAdmissible;

    A._xSizeTarget = B._xSizeSource;
    A._ySizeTarget = B._ySizeSource;
    A._xSizeSource = B._xSizeTarget;
    A._ySizeSource = B._ySizeTarget;
    A._zSize = B._zSize;

    A._xTarget = B._xSource;
    A._yTarget = B._ySource;
    A._xSource = B._xTarget;
    A._ySource = B._yTarget;

    A._teams = B._teams;
    A._level = B._level;
    A._inTargetTeam = B._inSourceTeam;
    A._inSourceTeam = B._inTargetTeam;
    A._targetRoot = B._sourceRoot;
    A._sourceRoot = B._targetRoot;

    A._block.Clear();
    A._block.type = B._block.type;

    switch( B._block.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        A._block.data.N = A.NewNode();
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int j=0; j<16; ++j )
            nodeA.children[j] = new DistQuasi2dHMat<Scalar>;

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointCopy( nodeB.Child(s,t) );
        break;
    }
    case DIST_LOW_RANK:
    {
        A._block.data.DF = new DistLowRank;
        DistLowRank& DFA = *A._block.data.DF;
        const DistLowRank& DFB = *B._block.data.DF;

        DFA.rank = DFB.rank;
        if( B._inTargetTeam )
            hmat_tools::Conjugate( DFB.ULocal, DFA.VLocal );
        if( B._inSourceTeam )
            hmat_tools::Conjugate( DFB.VLocal, DFA.ULocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        A._block.data.SF = new SplitLowRank;
        SplitLowRank& SFA = *A._block.data.SF;
        const SplitLowRank& SFB = *B._block.data.SF;

        SFA.rank = SFB.rank;
        hmat_tools::Conjugate( SFB.D, SFA.D );
        break;
    }
    case LOW_RANK:
    {
        A._block.data.F = new LowRank<Scalar>;
        hmat_tools::Adjoint( *B._block.data.F, *A._block.data.F );
        break;
    }
    case SPLIT_DENSE:
    {
        A._block.data.SD = new SplitDense;
        break;
    }
    case DENSE:
    {
        A._block.data.D = new Dense<Scalar>;
        hmat_tools::Adjoint( *B._block.data.D, *A._block.data.D );
        break;
    }
    }

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::AdjointPassData( const DistQuasi2dHMat<Scalar>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointPassData");
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
    MPI_Comm comm = _teams->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<MPI_Request> recvRequests( numRecvs );
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
    std::vector<MPI_Request> sendRequests( numSends );
    offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    for( int i=0; i<numRecvs; ++i )
        mpi::Wait( recvRequests[i] );
    AdjointPassDataUnpack
    ( B, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void 
DistQuasi2dHMat<Scalar>::AdjointPassDataCount
( const DistQuasi2dHMat<Scalar>& B,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointPassDataCount");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const DistQuasi2dHMat<Scalar>& A = *this;

    switch( B._block.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataCount
                ( nodeB.Child(s,t), sendSizes, recvSizes );
        break;
    }
    case SPLIT_DENSE:
    {
        if( B._inSourceTeam )
            AddToMap( sendSizes, B._targetRoot, B.Height()*B.Width() );
        if( A._inSourceTeam )
            AddToMap( recvSizes, A._targetRoot, A.Height()*B.Width() );
        break;
    }
    default:
        break;
    }

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void 
DistQuasi2dHMat<Scalar>::AdjointPassDataPack
( const DistQuasi2dHMat<Scalar>& B,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointPassDataPack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const DistQuasi2dHMat<Scalar>& A = *this;

    switch( B._block.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataPack
                ( nodeB.Child(s,t), buffer, offsets );
        break;
    }
    case SPLIT_DENSE:
    {
        if( B._inSourceTeam )
        {
            SplitDense& SD = *B._block.data.SD;
            std::memcpy
            ( &buffer[offsets[B._targetRoot]], SD.D.LockedBuffer(),
              SD.D.Height()*SD.D.Width()*sizeof(Scalar) );
            offsets[B._targetRoot] += SD.D.Height()*SD.D.Width();
        }
        break;
    }
    default:
        break;
    }

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void 
DistQuasi2dHMat<Scalar>::AdjointPassDataUnpack
( const DistQuasi2dHMat<Scalar>& B,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointPassDataUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    DistQuasi2dHMat<Scalar>& A = *this;

    switch( B._block.type )
    {
    case DIST_NODE:
    case NODE:
    case SPLIT_NODE:
    {
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).AdjointPassDataUnpack
                ( nodeB.Child(s,t), buffer, offsets );
        break;
    }
    case SPLIT_DENSE:
    {
        if( A._inSourceTeam )
        {
            SplitDense& SD = *A._block.data.SD;
            SD.D.Resize(A.Height(), A.Width());
            const int m = A.Width();
            const int n = A.Height();
            const int LDim = A.Width();
            const int DLDim = SD.D.LDim();
            Scalar* RESTRICT ABuffer = SD.D.Buffer();
            for( int j=0; j<n; ++j)
                for( int i=0; i<m; ++i)
                    ABuffer[j+i*DLDim] = 
                        Conj(buffer[offsets[A._targetRoot]+i+j*LDim]);
            offsets[A._targetRoot] += m*n;
        }
        break;
    }
    default:
        break;
    }

#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
