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
DistHMat2d<Scalar>::MultiplyHMatRandomCompress
()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompress");
#endif

#ifdef MEMORY_INFO
    PrintGlobal( PeakMemoryUsage()/1024./1024.,
                 "Peak Memory Before Compress(MB): " );
    NewMemoryCount( 2 );
#endif
    // Compress low-rank F matrix into much lower form.
    // Our low-rank matrix is UV', we want to project U and V onto a small
    // colums spaces( random test vectors ).
    MultiplyHMatCompressLowRankCountAndResize(0);
    MultiplyHMatCompressLowRankImport(0);

#ifdef MEMORY_INFO
    PrintGlobal( PeakMemoryUsage( 2 )/1024./1024.,
                 "Peak Memory Of Resize And Import(MB): " );
    PrintGlobal( PeakMemoryUsage()/1024./1024.,
                 "Peak Memory After Import(MB): " );
    PrintGlobalMemoryInfo();
    EraseMemoryCount( 2 );
    NewMemoryCount( 1 );
#endif

    // Generate Random test vectors, calculate Omega1'U and V'Omega2.
    MultiplyHMatRandomCompressPrecompute();
    MultiplyHMatRandomCompressReducesOmegaTUV();
    MultiplyHMatRandomCompressPassOmegaTUV();
    MultiplyHMatRandomCompressBroadcastsOmegaTUV();

    // HERE: memoryCheck
    MultiplyHMatRandomCompressMidcompute();

    MultiplyHMatRandomCompressReducesTSqr();

    const Real midcomputeTol = MidcomputeTolerance<Real>();
    const Real compressionTol = CompressionTolerance<Real>();
    MultiplyHMatRandomCompressPostcompute( midcomputeTol, compressionTol );


    // Broadcastsnum and broadcasts in compression to
    // broad cast BL_ and BR_
    MultiplyHMatRandomCompressBroadcastsNum();
    MultiplyHMatRandomCompressBroadcasts();
    // Compute the final U and V store in the usual space.
    MultiplyHMatRandomCompressFinalcompute();

#ifdef MEMORY_INFO
    PrintGlobal( PeakMemoryUsage()/1024./1024., "Peak Memory(MB): " );
    PrintGlobal( PeakMemoryUsage( 1 )/1024./1024.,
                 "Temp Peak Memory(MB): " );
    EraseMemoryCount( 1 );
#endif
    // Clean up all the space used in this file
    // Also, clean up the colXMap_, rowXMap_, UMap_, VMap_, ZMap_
    MultiplyHMatCompressFCleanup();
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPrecompute()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressPrecompute");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    const int sampleRank = SampleRank( MaxRank() );

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressPrecompute();
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        const int LH=LocalHeight();
        const int LW=LocalWidth();
        const int totalrank = DF.rank;

        if( inTargetTeam_ && totalrank > MaxRank() )
        {
            rowOmega_.Resize( LH, sampleRank );
            ParallelGaussianRandomVectors( rowOmega_ );
            OmegaTU_.Resize( sampleRank, totalrank );
            hmat_tools::TransposeMultiply
            ( Scalar(1), rowOmega_, DF.ULocal, OmegaTU_ );
        }

        if( inSourceTeam_ && totalrank > MaxRank() )
        {
            colOmega_.Resize( LW, sampleRank );
            ParallelGaussianRandomVectors( colOmega_ );
            OmegaTV_.Resize( sampleRank, totalrank );
            hmat_tools::TransposeMultiply
            ( Scalar(1), colOmega_, DF.VLocal, OmegaTV_ );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        SplitLowRank &SF = *block_.data.SF;
        int LH=Height();
        int LW=Width();
        int totalrank = SF.rank;

        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
                rowOmega_.Resize( LH, sampleRank );
                ParallelGaussianRandomVectors( rowOmega_ );
                OmegaTU_.Resize( sampleRank, totalrank );
                hmat_tools::TransposeMultiply
                ( Scalar(1), rowOmega_, SF.D, OmegaTU_ );
            }

            if( inSourceTeam_ )
            {
                colOmega_.Resize( LW, sampleRank );
                ParallelGaussianRandomVectors( colOmega_ );
                OmegaTV_.Resize( sampleRank, totalrank );
                hmat_tools::TransposeMultiply
                ( Scalar(1), colOmega_, SF.D, OmegaTV_ );
            }
        }
        break;
    }
    case LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        LowRank<Scalar> &F = *block_.data.F;
        int LH=Height();
        int LW=Width();
        int totalrank = F.U.Width();

        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            rowOmega_.Resize( LH, sampleRank );
            ParallelGaussianRandomVectors( rowOmega_ );
            OmegaTU_.Resize( sampleRank, totalrank );
            hmat_tools::TransposeMultiply
            ( Scalar(1), rowOmega_, F.U, OmegaTU_ );

            colOmega_.Resize( LW, sampleRank );
            ParallelGaussianRandomVectors( colOmega_ );
            OmegaTV_.Resize( sampleRank, totalrank );
            hmat_tools::TransposeMultiply
            ( Scalar(1), colOmega_, F.V, OmegaTV_ );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesOmegaTUV()
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesOmegaTUV");
#endif

    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    Vector<int> sizes( numReduces, 0 );
    MultiplyHMatRandomCompressReducesOmegaTUVCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numReduces; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatRandomCompressReducesOmegaTUVPack( buffer, offsetscopy );

    MultiplyHMatRandomCompressTreeReducesOmegaTUV( buffer, sizes );

    MultiplyHMatRandomCompressReducesOmegaTUVUnpack( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesOmegaTUVCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesOmegaTUVCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesOmegaTUVCount
                ( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
            sizes[level_] += OmegaTU_.Height()*OmegaTU_.Width();
        if( inSourceTeam_ )
            sizes[level_] += OmegaTV_.Height()*OmegaTV_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesOmegaTUVPack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesOmegaTUVPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesOmegaTUVPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( inTargetTeam_ )
        {
            int size=OmegaTU_.Height()*OmegaTU_.Width();
            MemCopy( &buffer[offsets[level_]], OmegaTU_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                OmegaTU_.Clear();
        }

        if( inSourceTeam_ )
        {
            int size=OmegaTV_.Height()*OmegaTV_.Width();
            MemCopy( &buffer[offsets[level_]], OmegaTV_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                OmegaTV_.Clear();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressTreeReducesOmegaTUV
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressTreeReducesOmegaTUV");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesOmegaTUVUnpack
( const Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesOmegaTUVUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesOmegaTUVUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
            {
                int size=OmegaTU_.Height()*OmegaTU_.Width();
                MemCopy( OmegaTU_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            if( inSourceTeam_ )
            {
                int size=OmegaTV_.Height()*OmegaTV_.Width();
                MemCopy( OmegaTV_.Buffer(), &buffer[offsets[level_]], size );
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPassOmegaTUV()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressPassOmegaTUV");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendsizes, recvsizes;
    MultiplyHMatRandomCompressPassOmegaTUVCount( sendsizes, recvsizes );

    // Compute the offsets
    int totalSendsize=0, totalRecvsize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendsizes.begin(); it!=sendsizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendsize;
        totalSendsize += it->second;
    }
    for( it=recvsizes.begin(); it!=recvsizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvsize;
        totalRecvsize += it->second;
    }

    // Fill the send buffer
    Vector<Scalar> sendBuffer(totalSendsize);
    std::map<int,int> offsets = sendOffsets;
    MultiplyHMatRandomCompressPassOmegaTUVPack( sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvsizes.size();
    Vector<mpi::Request> recvRequests( numRecvs );
    Vector<Scalar> recvBuffer( totalRecvsize );
    int offset = 0;
    for( it=recvsizes.begin(); it!=recvsizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvsizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    mpi::Barrier( comm );

    // Start the non-blocking sends
    const int numSends = sendsizes.size();
    Vector<mpi::Request> sendRequests( numSends );
    offset = 0;
    for( it=sendsizes.begin(); it!=sendsizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendsizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    mpi::WaitAll( numRecvs, &recvRequests[0] );
    MultiplyHMatRandomCompressPassOmegaTUVUnpack( recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPassOmegaTUVCount
( std::map<int,int>& sendsizes, std::map<int,int>& recvsizes )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressPassOmegaTUVCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressPassOmegaTUVCount
                ( sendsizes, recvsizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inSourceTeam_ )
            {
                AddToMap
                ( sendsizes, targetRoot_,
                  OmegaTV_.Height()*OmegaTV_.Width() );

                AddToMap
                ( recvsizes, targetRoot_, totalrank*sampleRank );
            }
            else
            {
                AddToMap
                ( sendsizes, sourceRoot_,
                  OmegaTU_.Height()*OmegaTU_.Width() );

                AddToMap
                ( recvsizes, sourceRoot_, totalrank*sampleRank );
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( !haveDenseUpdate_ )
        {
            if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
            {
                if( inSourceTeam_ )
                {
                    AddToMap
                    ( sendsizes, targetRoot_,
                      OmegaTV_.Height()*OmegaTV_.Width() );

                    AddToMap
                    ( recvsizes, targetRoot_, totalrank*sampleRank );
                }
                else
                {
                    AddToMap
                    ( sendsizes, sourceRoot_,
                      OmegaTU_.Height()*OmegaTU_.Width() );

                    AddToMap
                    ( recvsizes, sourceRoot_, totalrank*sampleRank );
                }
            }
            else if( totalrank > MaxRank() )
            {
                if( inSourceTeam_ )
                {
                    AddToMap
                    ( sendsizes, targetRoot_, SF.D.Height()*SF.D.Width() );
                    AddToMap
                    ( recvsizes, targetRoot_, LH*SF.D.Width() );
                }
                else
                {
                    AddToMap
                    ( sendsizes, sourceRoot_, SF.D.Height()*SF.D.Width() );
                    AddToMap
                    ( recvsizes, sourceRoot_, LW*SF.D.Width() );
                }
            }
        }
        else
        {
            if( inTargetTeam_ )
            {
                AddToMap( sendsizes, sourceRoot_, Height()*SF.D.Width() );
                AddToMap
                ( recvsizes, sourceRoot_,
                  Width()*SF.D.Width()+Width()*Height());
            }
            else
            {
                AddToMap
                ( sendsizes, targetRoot_,
                  Width()*SF.D.Width()+Width()*Height());
                AddToMap( recvsizes, targetRoot_, Height()*SF.D.Width() );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( inTargetTeam_ )
        {
            UMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            AddToMap( sendsizes, sourceRoot_, Height()*U.Width() );
        }
        else
        {
            VMap_.ResetIterator();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();
            AddToMap( recvsizes, targetRoot_, Height()*V.Width() );
        }

        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPassOmegaTUVPack
( Vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressPassOmegaTUVPack");
#endif
    if(  Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressPassOmegaTUVPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ && inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
            {
                const int size = OmegaTU_.Height()*OmegaTU_.Width();
                MemCopy
                ( &buffer[offsets[sourceRoot_]],
                  OmegaTU_.LockedBuffer(), size );
                  offsets[sourceRoot_] += size;
            }
            else
            {
                const int size = OmegaTV_.Height()*OmegaTV_.Width();
                MemCopy
                ( &buffer[offsets[targetRoot_]],
                  OmegaTV_.LockedBuffer(), size );
                  offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;
        const int sampleRank = SampleRank( MaxRank() );

        if( !haveDenseUpdate_ )
        {
            if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
            {
                if( inTargetTeam_ )
                {
                    const int size = OmegaTU_.Height()*OmegaTU_.Width();
                    MemCopy
                    ( &buffer[offsets[sourceRoot_]],
                      OmegaTU_.LockedBuffer(), size );
                    offsets[sourceRoot_] += size;
                }
                else
                {
                    const int size = OmegaTV_.Height()*OmegaTV_.Width();
                    MemCopy
                    ( &buffer[offsets[targetRoot_]],
                      OmegaTV_.LockedBuffer(), size );
                    offsets[targetRoot_] += size;
                }
            }
            else if( totalrank > MaxRank() )
            {
                if( inSourceTeam_ )
                {
                    MemCopy
                    ( &buffer[offsets[targetRoot_]], SF.D.LockedBuffer(),
                      SF.D.Height()*SF.D.Width() );
                    offsets[targetRoot_] += SF.D.Height()*SF.D.Width();
                }
                else
                {
                    MemCopy
                    ( &buffer[offsets[sourceRoot_]], SF.D.LockedBuffer(),
                      SF.D.Height()*SF.D.Width() );
                    offsets[sourceRoot_] += SF.D.Height()*SF.D.Width();
                }
            }
        }
        else
        {
            const int m = Height();
            const int n = Width();
            if( inTargetTeam_ )
            {
                int size = m*SF.D.Width();
                MemCopy
                ( &buffer[offsets[sourceRoot_]], SF.D.LockedBuffer(), size );
                offsets[sourceRoot_] += size;
            }
            else
            {
                int size = n*SF.D.Width();
                MemCopy
                ( &buffer[offsets[targetRoot_]], SF.D.LockedBuffer(), size );
                offsets[targetRoot_] += size;

                size = n*m;
                MemCopy
                ( &buffer[offsets[targetRoot_]], D_.LockedBuffer(), size );
                offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( inTargetTeam_ )
        {
            UMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            if( Height() != U.Height() )
                throw std::logic_error
                ("Packing SPLIT_DENSE, the height does not fit");
            int size=U.Height()*U.Width();
            MemCopy
            ( &buffer[offsets[sourceRoot_]], U.LockedBuffer(), size );
            offsets[sourceRoot_] += size;
            UMap_.Clear();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPassOmegaTUVUnpack
( const Vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressPassOmegaTUVUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressPassOmegaTUVUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ && inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
            {
                const int size = sampleRank*totalrank;
                OmegaTV_.Resize( sampleRank, totalrank );
                MemCopy
                ( OmegaTV_.Buffer(), &buffer[offsets[sourceRoot_]], size );
                offsets[sourceRoot_] += size;
            }
            else
            {
                const int size = sampleRank*totalrank;
                OmegaTU_.Resize( sampleRank, totalrank );
                MemCopy
                ( OmegaTU_.Buffer(), &buffer[offsets[targetRoot_]], size );
                offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;
        const int sampleRank = SampleRank( MaxRank() );

        if( !haveDenseUpdate_ )
        {
            if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
            {
                if( inTargetTeam_ )
                {
                    const int size = sampleRank*totalrank;
                    OmegaTV_.Resize( sampleRank, totalrank );
                    MemCopy
                    ( OmegaTV_.Buffer(), &buffer[offsets[sourceRoot_]], size );
                    offsets[sourceRoot_] += size;
                }
                else
                {
                    const int size = sampleRank*totalrank;
                    OmegaTU_.Resize( sampleRank, totalrank );
                    MemCopy
                    ( OmegaTU_.Buffer(), &buffer[offsets[targetRoot_]], size );
                    offsets[targetRoot_] += size;
                }
            }
            else if( totalrank > MaxRank() )
            {
                if( inSourceTeam_ )
                {
                    SFD_.Resize( LH, totalrank );
                    SFD_.Init();
                    MemCopy
                    ( SFD_.Buffer(), &buffer[offsets[targetRoot_]],
                      SFD_.Height()*SFD_.Width() );
                    offsets[targetRoot_] += SFD_.Height()*SFD_.Width();
                }
                else
                {
                    SFD_.Resize( LW, totalrank );
                    SFD_.Init();
                    MemCopy
                    ( SFD_.Buffer(), &buffer[offsets[sourceRoot_]],
                      SFD_.Height()*SFD_.Width() );
                    offsets[sourceRoot_] += SFD_.Height()*SFD_.Width();
                }
            }
        }
        else
        {
            const int m = Height();
            const int n = Width();
            if( inTargetTeam_ )
            {
                int size = n*SF.D.Width();
                SFD_.Resize( n, SF.D.Width() );
                SFD_.Init();
                MemCopy
                ( SFD_.Buffer(), &buffer[offsets[sourceRoot_]], size );
                offsets[sourceRoot_] += size;

                size = n*m;
                D_.Resize(m, n);
                D_.Init();
                MemCopy
                ( D_.Buffer(), &buffer[offsets[sourceRoot_]], size );
                offsets[sourceRoot_] += size;
            }
            else
            {
                int size = m*SF.D.Width();
                SFD_.Resize(m, SF.D.Width());
                SFD_.Init();
                MemCopy
                ( SFD_.Buffer(), &buffer[offsets[targetRoot_]], size );
                offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( inSourceTeam_ )
        {
            UMap_.Set( 0, new Dense<Scalar>);
            Dense<Scalar>& U = UMap_.Get(0);
            VMap_.ResetIterator();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();
            U.Resize(Height(), V.Width(), Height());
            U.Init();
            int size=U.Height()*U.Width();
            MemCopy
            ( U.Buffer(), &buffer[offsets[targetRoot_]],  size );
            offsets[targetRoot_] += size;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsOmegaTUV()
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressBroadcastsOmegaTUV");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatRandomCompressBroadcastsOmegaTUVCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatRandomCompressBroadcastsOmegaTUVPack( buffer, offsetscopy );

    MultiplyHMatRandomCompressTreeBroadcastsOmegaTUV( buffer, sizes );

    MultiplyHMatRandomCompressBroadcastsOmegaTUVUnpack( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsOmegaTUVCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressBroadcastsOmegaTUVCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child
                (t,s).MultiplyHMatRandomCompressBroadcastsOmegaTUVCount
                ( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
            sizes[level_] += totalrank*sampleRank;
        if( inSourceTeam_ )
            sizes[level_] += totalrank*sampleRank;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsOmegaTUVPack
( Vector<Scalar>& buffer, Vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressBroadcastsOmegaTUVPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child
                (t,s).MultiplyHMatRandomCompressBroadcastsOmegaTUVPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( inTargetTeam_ )
            {
                int size = OmegaTV_.Height()*OmegaTV_.Width();
                MemCopy
                ( &buffer[offsets[level_]], OmegaTV_.LockedBuffer(), size );
                offsets[level_] += size;
            }
            if( inSourceTeam_ )
            {
                int size = OmegaTU_.Height()*OmegaTU_.Width();
                MemCopy
                ( &buffer[offsets[level_]], OmegaTU_.LockedBuffer(), size );
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressTreeBroadcastsOmegaTUV
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressTreeBroadcastsOmegaTUV");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsOmegaTUVUnpack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressBroadcastsOmegaTUVUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child
                (t,s).MultiplyHMatRandomCompressBroadcastsOmegaTUVUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
        {
            const int size = sampleRank*totalrank;
            OmegaTV_.Resize( sampleRank, totalrank );
            MemCopy( OmegaTV_.Buffer(), &buffer[offsets[level_]], size );
            offsets[level_] += size;
        }
        if( inSourceTeam_ )
        {
            const int size = sampleRank*totalrank;
            OmegaTU_.Resize( sampleRank, totalrank );
            MemCopy( OmegaTU_.Buffer(), &buffer[offsets[level_]], size );
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressMidcompute()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressMidcompute");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressMidcompute();
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
        {
            hmat_tools::MultiplyTranspose
            ( Scalar(1), DF.ULocal, OmegaTV_, colT_ );
            DF.ULocal.Clear();

            hmat_tools::AdjointMultiply
            ( Scalar(1), colT_, colT_, colTSqr_ );
        }
        if( inSourceTeam_ )
        {
            hmat_tools::MultiplyTranspose
            ( Scalar(1), DF.VLocal, OmegaTU_, rowT_ );
            DF.VLocal.Clear();

            hmat_tools::AdjointMultiply
            ( Scalar(1), rowT_, rowT_, rowTSqr_ );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        const int totalrank = SF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SF.D, OmegaTV_, colT_ );
                SF.D.Clear();

                hmat_tools::AdjointMultiply
                ( Scalar(1), colT_, colT_, colTSqr_ );
            }
            else
            {
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SF.D, OmegaTU_, rowT_ );
                SF.D.Clear();

                hmat_tools::AdjointMultiply
                ( Scalar(1), rowT_, rowT_, rowTSqr_ );
            }
        }
        break;
    }
    case LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        LowRank<Scalar> &F = *block_.data.F;
        const int LH = Height();
        const int LW = Width();
        const int totalrank = F.U.Width();
        const int sampleRank = SampleRank( MaxRank() );
        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            hmat_tools::MultiplyTranspose
            ( Scalar(1), F.U, OmegaTV_, colT_ );

            hmat_tools::AdjointMultiply
            ( Scalar(1), colT_, colT_, colTSqr_ );

            hmat_tools::MultiplyTranspose
            ( Scalar(1), F.V, OmegaTU_, rowT_ );

            hmat_tools::AdjointMultiply
            ( Scalar(1), rowT_, rowT_, rowTSqr_ );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesTSqr()
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesTSqr");
#endif

    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    Vector<int> sizes( numReduces, 0 );
    MultiplyHMatRandomCompressReducesTSqrCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numReduces; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatRandomCompressReducesTSqrPack( buffer, offsetscopy );

    MultiplyHMatRandomCompressTreeReducesTSqr( buffer, sizes );

    MultiplyHMatRandomCompressReducesTSqrUnpack( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesTSqrCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesTSqrCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesTSqrCount
                ( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
            sizes[level_] += colTSqr_.Height()*colTSqr_.Width();
        if( inSourceTeam_ )
            sizes[level_] += rowTSqr_.Height()*rowTSqr_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesTSqrPack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesTSqrPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesTSqrPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( inTargetTeam_ )
        {
            int size=colTSqr_.Height()*colTSqr_.Width();
            MemCopy( &buffer[offsets[level_]], colTSqr_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                colTSqr_.Clear();
        }

        if( inSourceTeam_ )
        {
            int size=rowTSqr_.Height()*rowTSqr_.Width();
            MemCopy( &buffer[offsets[level_]], rowTSqr_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                rowTSqr_.Clear();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressTreeReducesTSqr
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressTreeReducesTSqr");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressReducesTSqrUnpack
( const Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry
        ("DistHMat2d::MultiplyHMatRandomCompressReducesTSqrUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressReducesTSqrUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
            {
                int size=colTSqr_.Height()*colTSqr_.Width();
                MemCopy( colTSqr_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            if( inSourceTeam_ )
            {
                int size=rowTSqr_.Height()*rowTSqr_.Width();
                MemCopy( rowTSqr_.Buffer(), &buffer[offsets[level_]], size );
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressPostcompute
( Real epsilon, Real relTol )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressPostcompute");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s)
                node.Child(t,s).MultiplyHMatRandomCompressPostcompute
                ( epsilon, relTol );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        const int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        const int sampleRank = SampleRank( MaxRank() );
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
            {
                const int k = colTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  colTSqr_.Buffer(), colTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            colTSqr_.Set
                            (i,j,colTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( colTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), Omega, colTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), colTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                for( int j=0; j<Sigma.Size(); ++j )
                    for( int i=0; i<sampleRank; ++i )
                        U.Set(i,j,U.Get(i,j)*Sigma[j]);

                BL_.Resize( k, Sigma.Size() );
                hmat_tools::Multiply( Scalar(1), Ztmp, U, BL_ );

                colTSqr_.Clear();
            }

            if( inSourceTeam_ )
            {
                const int k = rowTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  rowTSqr_.Buffer(), rowTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            rowTSqr_.Set
                            (i,j,rowTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( rowTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> OmegaT;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTV_, OmegaTU_, OmegaT );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), OmegaT, rowTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), rowTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                hmat_tools::MultiplyTranspose
                ( Scalar(1), Ztmp, VH, BR_ );

                rowTSqr_.Clear();
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        const int totalrank = SF.rank;
        const int sampleRank = SampleRank( MaxRank() );
        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
                const int k = colTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  colTSqr_.Buffer(), colTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            colTSqr_.Set
                            (i,j,colTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( colTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), Omega, colTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), colTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                for( int j=0; j<Sigma.Size(); ++j )
                    for( int i=0; i<sampleRank; ++i )
                        U.Set(i,j,U.Get(i,j)*Sigma[j]);

                BL_.Resize( k, Sigma.Size() );
                hmat_tools::Multiply( Scalar(1), Ztmp, U, BL_ );

                colTSqr_.Clear();
            }

            if( inSourceTeam_ )
            {
                const int k = rowTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  rowTSqr_.Buffer(), rowTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            rowTSqr_.Set
                            (i,j,rowTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( rowTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> OmegaT;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTV_, OmegaTU_, OmegaT );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), OmegaT, rowTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), rowTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                hmat_tools::MultiplyTranspose
                ( Scalar(1), Ztmp, VH, BR_ );

                rowTSqr_.Clear();
            }

        }
        else if( totalrank > MaxRank() )
        {
            Dense<Scalar> B;
            if( inSourceTeam_ )
            {
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SFD_, SF.D, B );
            }
            else
            {
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SF.D, SFD_, B );
            }

            const int minDim = std::min( LH, LW );
            BSqrU_.Resize( LH, minDim );
            BSqrU_.Init();
            BSqrVH_.Resize( minDim, LW );
            BSqrVH_.Init();
            BSigma_.Resize( minDim );

            const int lwork = lapack::SVDWorkSize(LH,LW);
            const int lrwork = lapack::SVDRealWorkSize(LH,LW);
            Vector<Scalar> work(lwork);
            Vector<Real> rwork(lrwork);
            lapack::SVD
            ('S', 'S' , LH, LW,
             B.Buffer(), B.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol );
            SFD_.Clear();
        }
        break;
    }
    case LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        LowRank<Scalar> &F = *block_.data.F;
        const int LH = Height();
        const int LW = Width();
        const int totalrank = F.U.Width();
        const int sampleRank = SampleRank( MaxRank() );
        if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
        {
            {
                const int k = colTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  colTSqr_.Buffer(), colTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            colTSqr_.Set
                            (i,j,colTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( colTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), Omega, colTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), colTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                for( int j=0; j<Sigma.Size(); ++j )
                    for( int i=0; i<sampleRank; ++i )
                        U.Set(i,j,U.Get(i,j)*Sigma[j]);

                BL_.Resize( k, Sigma.Size() );
                hmat_tools::Multiply( Scalar(1), Ztmp, U, BL_ );

                colTSqr_.Clear();
            }

            {
                const int k = rowTSqr_.Height();
                Vector<Real> TSqrEig( k );
                lapack::EVD
                ( 'V', 'U', k,
                  rowTSqr_.Buffer(), rowTSqr_.LDim(), &TSqrEig[0] );

                Real maxEig = std::max( TSqrEig[k-1], Real(0) );

                const Real tolerance = sqrt(epsilon*maxEig*k);
                for( int j=0; j<k; ++j )
                {
                    const Real omega = std::max( TSqrEig[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<k; ++i )
                            rowTSqr_.Set
                            (i,j,rowTSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( rowTSqr_.Buffer(0,j), k );
                }

                Dense<Scalar> Omega;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTU_, OmegaTV_, Omega );

                Dense<Scalar> OmegaT;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), OmegaTV_, OmegaTU_, OmegaT );

                Dense<Scalar> Pinv;
                hmat_tools::Multiply( Scalar(1), OmegaT, rowTSqr_, Pinv );

                lapack::AdjointPseudoInverse
                ( Pinv.Height(), Pinv.Width(),
                  Pinv.Buffer(), Pinv.LDim(), epsilon );

                Dense<Scalar> Ztmp;
                hmat_tools::MultiplyAdjoint
                ( Scalar(1), rowTSqr_, Pinv, Ztmp );

                Dense<Scalar> U( sampleRank, sampleRank );
                Dense<Scalar> VH( sampleRank, sampleRank );
                Vector<Real> Sigma( sampleRank );
                lapack::SVD
                ( 'S', 'S', sampleRank, sampleRank,
                  Omega.Buffer(), Omega.LDim(), &Sigma[0],
                  U.Buffer(), U.LDim(),
                  VH.Buffer(), VH.LDim() );

                SVDTrunc( U, Sigma, VH, relTol );

                hmat_tools::MultiplyTranspose
                ( Scalar(1), Ztmp, VH, BR_ );

                rowTSqr_.Clear();
            }
        }
        else if( totalrank > MaxRank() )
        {
            Dense<Scalar> B;
            hmat_tools::MultiplyTranspose
            ( Scalar(1), F.U, F.V, B );

            const int minDim = std::min( LH, LW );
            BSqrU_.Resize( LH, minDim );
            BSqrU_.Init();
            BSqrVH_.Resize( minDim, LW );
            BSqrVH_.Init();
            BSigma_.Resize( minDim );

            const int lwork = lapack::SVDWorkSize(LH,LW);
            const int lrwork = lapack::SVDRealWorkSize(LH,LW);
            Vector<Scalar> work(lwork);
            Vector<Real> rwork(lrwork);
            lapack::SVD
            ('S', 'S' , LH, LW,
             B.Buffer(), B.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsNum()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsNum");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatRandomCompressBroadcastsNumCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<int> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatRandomCompressBroadcastsNumPack
    ( buffer, offsetscopy );

    MultiplyHMatRandomCompressTreeBroadcastsNum( buffer, sizes );

    MultiplyHMatRandomCompressBroadcastsNumUnpack
    ( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsNumCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsNumCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsNumCount
                ( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
            sizes[level_]++;
        if( inSourceTeam_ )
            sizes[level_]++;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsNumPack
( Vector<int>& buffer, Vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsNumPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsNumPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( inTargetTeam_ )
            {
                buffer[offsets[level_]] = BL_.Width();
                offsets[level_]++;
            }
            if( inSourceTeam_ )
            {
                buffer[offsets[level_]] = BR_.Width();
                offsets[level_]++;
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressTreeBroadcastsNum
( Vector<int>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressTreeBroadcastsNum");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsNumUnpack
( Vector<int>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsNumUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsNumUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank != 0 )
        {
            if( inTargetTeam_ )
            {
                BL_.Resize
                (colT_.Width(), buffer[offsets[level_]]);
                offsets[level_]++;
            }
            if( inSourceTeam_ )
            {
                BR_.Resize
                (rowT_.Width(), buffer[offsets[level_]]);
                offsets[level_]++;
            }
        }
        else
        {
            if( inTargetTeam_ )
                offsets[level_]++;
            if( inSourceTeam_ )
                offsets[level_]++;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcasts()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatRandomCompressBroadcastsCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatRandomCompressBroadcastsPack
    ( buffer, offsetscopy );

    MultiplyHMatRandomCompressTreeBroadcasts( buffer, sizes );

    MultiplyHMatRandomCompressBroadcastsUnpack
    ( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsCount");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsCount
                ( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
            sizes[level_] += BL_.Height()*BL_.Width();
        if( inSourceTeam_ )
            sizes[level_] += BR_.Height()*BR_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsPack
( Vector<Scalar>& buffer, Vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsPack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( inTargetTeam_ )
            {
                int size = BL_.Height()*BL_.Width();
                MemCopy( &buffer[offsets[level_]], BL_.LockedBuffer(), size );
                offsets[level_] += size;
            }
            if( inSourceTeam_ )
            {
                int size = BR_.Height()*BR_.Width();
                MemCopy ( &buffer[offsets[level_]], BR_.LockedBuffer(), size );
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressTreeBroadcasts
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressTreeBroadcasts");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatRandomCompressBroadcastsUnpack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressBroadcastsUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressBroadcastsUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
        {
            int size = BL_.Height()*BL_.Width();
            MemCopy( BL_.Buffer(), &buffer[offsets[level_]], size );
            offsets[level_] += size;
        }
        if( inSourceTeam_ )
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
DistHMat2d<Scalar>::MultiplyHMatRandomCompressFinalcompute()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatRandomCompressFinalcompute");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatRandomCompressFinalcompute();
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        if( inTargetTeam_ )
        {
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &U = DF.ULocal;
            DF.rank = BL_.Width();
            hmat_tools::Multiply( Scalar(1), colT_, BL_, U );
            colT_.Clear();
            BL_.Clear();
        }
        if( inSourceTeam_ )
        {
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &V = DF.VLocal;
            DF.rank = BR_.Width();
            hmat_tools::Multiply( Scalar(1), rowT_, BR_, V );
            rowT_.Clear();
            BR_.Clear();
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( !haveDenseUpdate_ )
        {
            SplitLowRank& SF = *block_.data.SF;
            const int LH = Height();
            const int LW = Width();
            const int totalrank = SF.rank;
            const int sampleRank = SampleRank( MaxRank() );
            if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
            {
                if( inTargetTeam_ )
                {
                    Dense<Scalar> &U = SF.D;
                    SF.rank = BL_.Width();
                    hmat_tools::Multiply( Scalar(1), colT_, BL_, U );
                    colT_.Clear();
                    BL_.Clear();
                }
                if( inSourceTeam_ )
                {
                    Dense<Scalar> &V = SF.D;
                    SF.rank = BR_.Width();
                    hmat_tools::Multiply( Scalar(1), rowT_, BR_, V );
                    rowT_.Clear();
                    BR_.Clear();
                }
            }
            else if( totalrank > MaxRank() )
            {
                SF.rank = BSqrU_.Width();
                if( inTargetTeam_ )
                    hmat_tools::Copy( BSqrU_, SF.D );
                if( inSourceTeam_ )
                {
                    SF.D.Resize( LW, BSqrVH_.Height() );
                    SF.D.Init();
                    for( int j=0; j<BSqrVH_.Height(); ++j )
                        for( int i=0; i<LW; ++i )
                            SF.D.Set(i,j,BSqrVH_.Get(j,i)*BSigma_[j]);
                }

            }
        }
        else
        {
            SplitLowRank &SF = *block_.data.SF;
            const int m = Height();
            const int n = Width();
            if( inTargetTeam_ )
            {
                Dense<Scalar>& SFU = SF.D;
                Dense<Scalar>& SFV = SFD_;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SFU, SFV, Scalar(1), D_ );
            }
            else
            {
                Dense<Scalar>& SFU = SFD_;
                Dense<Scalar>& SFV = SF.D;
                hmat_tools::MultiplyTranspose
                ( Scalar(1), SFU, SFV, Scalar(1), D_ );
            }


            const int minDim = std::min(m,n);
            const int maxRank = MaxRank();
            if( minDim <= maxRank )
            {
                if( inTargetTeam_ )
                {
                    SF.rank = minDim;
                    if( m == minDim )
                    {
                        SF.D.Resize( m, m, m );
                        SF.D.Init();
                        for( int i=0; i<m; i++)
                            SF.D.Set(i,i,Scalar(1));
                    }
                    else
                        hmat_tools::Copy( D_, SF.D );
                }
                else
                {
                    SF.rank = minDim;
                    if( m == minDim )
                        hmat_tools::Transpose( D_, SF.D );
                    else
                    {
                        SF.D.Resize( n, n, n);
                        SF.D.Init();
                        for( int i=0; i<n; i++)
                            SF.D.Set(i,i,Scalar(1));
                    }
                }
            }
            else
            {
                SF.rank = maxRank;
                Vector<Real> sigma( minDim );
                Vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                Vector<Real> realwork( lapack::SVDRealWorkSize(m,n) );
                if( inTargetTeam_ )
                {
                    lapack::SVD
                    ( 'O', 'N', m, n, D_.Buffer(), D_.LDim(),
                      &sigma[0], 0, 1, 0, 1,
                      &work[0], work.Size(), &realwork[0] );

                    SF.D.Resize( m, maxRank );
                    SF.D.Init();
                    for( int j=0; j<maxRank; j++ )
                        for( int i=0; i<m; i++)
                            SF.D.Set(i,j,D_.Get(i,j)*sigma[j]);
                }
                else
                {
                    lapack::SVD
                    ( 'N', 'O', m, n, D_.Buffer(), D_.LDim(),
                      &sigma[0], 0, 1, 0, 1,
                      &work[0], work.Size(), &realwork[0] );

                    SF.D.Resize( n, maxRank );
                    SF.D.Init();
                    for( int j=0; j<maxRank; j++ )
                        for( int i=0; i<n; i++)
                            SF.D.Set(i,j,D_.Get(j,i));
                }
            }

            D_.Clear();
            SFD_.Clear();
            haveDenseUpdate_ = false;
            storedDenseUpdate_ = false;
        }
        break;
    }
    case LOW_RANK:
    {
        if( !haveDenseUpdate_ )
        {
            LowRank<Scalar> &F = *block_.data.F;
            const int LH = Height();
            const int LW = Width();
            const int totalrank = F.U.Width();
            const int sampleRank = SampleRank( MaxRank() );
            if( LH*LW > totalrank*sampleRank && totalrank > MaxRank() )
            {
                LowRank<Scalar> &F = *block_.data.F;
                Dense<Scalar> &U = F.U;
                hmat_tools::Multiply( Scalar(1), colT_, BL_, U );
                colT_.Clear();
                BL_.Clear();

                Dense<Scalar> &V = F.V;
                hmat_tools::Multiply( Scalar(1), rowT_, BR_, V );
                rowT_.Clear();
                BR_.Clear();
            }
            else if( totalrank > MaxRank() )
            {
                hmat_tools::Copy( BSqrU_, F.U );
                F.V.Resize( LW, BSqrVH_.Height() );
                F.V.Init();
                for( int j=0; j<BSqrVH_.Height(); ++j )
                    for( int i=0; i<LW; ++i )
                        F.V.Set(i,j,BSqrVH_.Get(j,i)*BSigma_[j]);
            }
        }
        else
        {
            LowRank<Scalar> &F = *block_.data.F;
            const int m = F.Height();
            const int n = F.Width();
            const int minDim = std::min( m, n );
            const int maxRank = MaxRank();

            // Add U V^[T/H] onto the dense update
            hmat_tools::MultiplyTranspose
            ( Scalar(1), F.U, F.V, Scalar(1), D_ );

            if( minDim <= maxRank )
            {
                if( m == minDim )
                {
                    // Make U := I and V := D_^[T/H]
                    F.U.Resize( minDim, minDim );
                    F.U.Init();
                    for( int j=0; j<minDim; ++j )
                        F.U.Set(j,j,Scalar(1));
                    hmat_tools::Transpose( D_, F.V );
                }
                else
                {
                    // Make U := D_ and V := I
                    hmat_tools::Copy( D_, F.U );
                    F.V.Resize( minDim, minDim );
                    F.V.Init();
                    for( int j=0; j<minDim; ++j )
                        F.V.Set(j,j,Scalar(1));
                }
            }
            else // minDim > maxRank
            {
                // Perform an SVD on the dense matrix, overwriting it with
                // the left singular vectors and VH with the adjoint of the
                // right singular vecs
                Dense<Scalar> VH( std::min(m,n), n );
                Vector<Real> sigma( minDim );
                Vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                Vector<Real> realWork( lapack::SVDRealWorkSize(m,n) );
                lapack::SVD
                ( 'O', 'S', m, n, D_.Buffer(), D_.LDim(),
                  &sigma[0], 0, 1, VH.Buffer(), VH.LDim(),
                  &work[0], work.Size(), &realWork[0] );

                // Form U with the truncated left singular vectors scaled
                // by the corresponding singular values
                F.U.Resize( m, maxRank );
                F.U.Init();
                for( int j=0; j<maxRank; ++j )
                    for( int i=0; i<m; ++i )
                        F.U.Set(i,j,sigma[j]*D_.Get(i,j));

                // Form V with the truncated right singular vectors
                F.V.Resize( n, maxRank );
                F.V.Init();
                for( int j=0; j<maxRank; ++j )
                    for( int i=0; i<n; ++i )
                        F.V.Set(i,j,VH.Get(j,i));
            }
            D_.Clear();
            haveDenseUpdate_ = false;
            storedDenseUpdate_ = false;
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( inSourceTeam_ )
        {
            SplitDense& SD = *block_.data.SD;

            UMap_.ResetIterator();
            VMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();

            hmat_tools::MultiplyTranspose
            ( Scalar(1), U, V, Scalar(1), SD.D );

            UMap_.Clear();
            VMap_.Clear();
        }
        break;
    }
    case DENSE:
    {
        Dense<Scalar>& D = *block_.data.D;

        UMap_.ResetIterator();
        VMap_.ResetIterator();
        const Dense<Scalar>& U = *UMap_.CurrentEntry();
        const Dense<Scalar>& V = *VMap_.CurrentEntry();

        hmat_tools::MultiplyTranspose
        ( Scalar(1), U, V, Scalar(1), D );

        UMap_.Clear();
        VMap_.Clear();
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
