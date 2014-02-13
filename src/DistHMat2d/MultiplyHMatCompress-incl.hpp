/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

int maxRankCount;

namespace dmhm {

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompress( Real twoNorm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompress");
#endif
    // Compress low-rank F matrix into much lower form.
    // Our low-rank matrix is UV', we want to compute eigenvalues and
    // eigenvectors of U'U and V'V.
    // U'U first stored in USqr_, then we use USqr_ to store the orthognal
    // vectors of U'U, and USqrEig_ to store the eigenvalues of U'U.
    // Everything about V are same in VSqr_ and VSqrEig_.

#ifdef MEMORY_INFO
    //PrintMemoryInfo( "MemoryInfo before Compression" );
    PrintGlobal( PeakMemoryUsage()/1024./1024.,
                 "Peak Memory Before Compress(MB): " );
    NewMemoryCount( 2 );
#endif

    maxRankCount = 0;

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Start( 2 );
#endif
    MultiplyHMatCompressLowRankCountAndResize(0);
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 2 );
    timerGlobal.Start( 3 );
#endif
    MultiplyHMatCompressLowRankImport(0);
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 3 );
    timerGlobal.Start( 4 );
#endif
#ifdef MEMORY_INFO
    PrintGlobal( double(maxRankCount), "Max Rank Number: ");
    PrintGlobal( PeakMemoryUsage( 2 )/1024./1024.,
                 "Peak Memory Of Resize And Import(MB): " );
    PrintGlobal( PeakMemoryUsage()/1024./1024.,
                 "Peak Memory After Import(MB): " );
    PrintGlobalMemoryInfo();
    EraseMemoryCount( 2 );
    NewMemoryCount( 1 );
#endif

    MultiplyHMatCompressFPrecompute();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 4 );
    timerGlobal.Start( 5 );
#endif
    MultiplyHMatCompressFReduces();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 5 );
    timerGlobal.Start( 6 );
#endif

    MultiplyHMatCompressFEigenDecomp();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 6 );
    timerGlobal.Start( 7 );
#endif
    MultiplyHMatCompressFPassMatrix();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 7 );
    timerGlobal.Start( 8 );
#endif
    MultiplyHMatCompressFPassVector();

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 8 );
    timerGlobal.Start( 9 );
#endif
    // Compute sigma_1 V1' V2 sigma_2, the middle part of UV'
    // We use B to state the mid part of UV' that is
    // B = sigma_1 V1' V2 sigma_2.
    // BSqr_ = sqrt(USqrEig_) USqr_' VSqr_ sqrt(VSqrEig_)
    // Then BSqr_ also will be used to store the eigenvectors
    // of B. BSqrEig_ stores eigenvalues of B.
    //
    const Real compressionTol = CompressionTolerance<Real>();
    MultiplyHMatCompressFMidcompute( compressionTol, twoNorm );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 9 );
    timerGlobal.Start( 10 );
#endif
    MultiplyHMatCompressFPassbackNum();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 10 );
    timerGlobal.Start( 11 );
#endif
    MultiplyHMatCompressFPassbackData();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 11 );
    timerGlobal.Start( 12 );
#endif

    // Compute USqr*sqrt(USqrEig)^-1 BSqrU BSigma = BL
    // We overwrite the USqr = USqr*sqrt(USqrEig)^-1
    // Also overwrite the BSqrU = BSqrU BSigma
    // Compute VSqr*sqrt(VSqrEig)^-1 BSqrV = BR
    // We overwrite the VSqr = VSqr*sqrt(VSqrEig)^-1
    //
    const Real pseudoinvTol = PseudoinvTolerance<Real>();
    MultiplyHMatCompressFPostcompute( pseudoinvTol );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 12 );
    timerGlobal.Start( 13 );
#endif

    MultiplyHMatCompressFBroadcastsNum();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 13 );
    timerGlobal.Start( 14 );
#endif
    MultiplyHMatCompressFBroadcasts();
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 14 );
    timerGlobal.Start( 15 );
#endif
    // Compute the final U and V store in the usual space.
    MultiplyHMatCompressFFinalcompute( compressionTol, twoNorm );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timerGlobal.Stop( 15 );
#endif

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
DistHMat2d<Scalar>::MultiplyHMatCompressLowRankCountAndResize( int rank )
{
#ifndef RELEASE
    CallStackEntry entry
    ("DistHMat2d::MultiplyHMatCompressLowRankCountAndResize");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        if( inTargetTeam_ )
        {
            const int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
                rank += UMap_.CurrentEntry()->Width();
        }
        else if( inSourceTeam_ )
        {
            const int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
                rank += VMap_.CurrentEntry()->Width();
        }

        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressLowRankCountAndResize
                ( rank );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *block_.data.DF;

        // Compute the new rank
        if( inTargetTeam_ )
        {
            // Add the F+=HH updates
            int numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
                rank += colXMap_.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
                rank += UMap_.CurrentEntry()->Width();

            // Add the rank of the original low-rank matrix
            rank += DF.ULocal.Width();
        }
        else if( inSourceTeam_ )
        {
            // Add the F+=HH updates
            int numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
                rank += rowXMap_.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
                rank += VMap_.CurrentEntry()->Width();

            // Add the rank of the original low-rank matrix
            rank += DF.VLocal.Width();
        }

        if( rank > maxRankCount )
            maxRankCount = rank;
        // Store the rank and create the space
        if( inTargetTeam_ )
        {
            const int oldRank = DF.ULocal.Width();
            const int localHeight = DF.ULocal.Height();

            Dense<Scalar> ULocalCopy;
            hmat_tools::Copy( DF.ULocal, ULocalCopy );
            DF.ULocal.Resize( localHeight, rank, localHeight );
            DF.ULocal.Init();
            MemCopy
            ( DF.ULocal.Buffer(0,rank-oldRank), ULocalCopy.LockedBuffer(),
              localHeight*oldRank );
        }
        if( inSourceTeam_ )
        {
            const int oldRank = DF.VLocal.Width();
            const int localWidth = DF.VLocal.Height();

            Dense<Scalar> VLocalCopy;
            hmat_tools::Copy( DF.VLocal, VLocalCopy );
            DF.VLocal.Resize( localWidth, rank, localWidth );
            DF.VLocal.Init();
            MemCopy
            ( DF.VLocal.Buffer(0,rank-oldRank), VLocalCopy.LockedBuffer(),
              localWidth*oldRank );
        }
        DF.rank = rank;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        // Compute the new rank
        if( inTargetTeam_ )
        {
            // Add the F+=HH updates
            int numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
                rank += colXMap_.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
                rank += UMap_.CurrentEntry()->Width();

            // Add the original rank
            rank += SF.D.Width();
        }
        else
        {
            // Add the F+=HH updates
            int numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
                rank += rowXMap_.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
                rank += VMap_.CurrentEntry()->Width();

            // Add the original rank
            rank += SF.D.Width();
        }

        if( rank > maxRankCount )
            maxRankCount = rank;
        // Create the space and store the rank if we'll need to do a QR
        if( inTargetTeam_ )
        {
            const int oldRank = SF.D.Width();
            const int height = SF.D.Height();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( SF.D, UCopy );
            SF.D.Resize( height, rank, height );
            SF.D.Init();
            MemCopy
            ( SF.D.Buffer(0,rank-oldRank), UCopy.LockedBuffer(),
              height*oldRank );
        }
        else
        {
            const int oldRank = SF.D.Width();
            const int width = SF.D.Height();

            Dense<Scalar> VCopy;
            hmat_tools::Copy( SF.D, VCopy );
            SF.D.Resize( width, rank, width );
            SF.D.Init();
            MemCopy
            ( SF.D.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              width*oldRank );
        }
        SF.rank = rank;
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        // Compute the total new rank
        {
            // Add the F+=HH updates
            int numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
                rank += colXMap_.CurrentEntry()->Width();

            // Add the low-rank updates
            numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
                rank += UMap_.CurrentEntry()->Width();

            // Add the original low-rank matrix
            rank += F.Rank();
        }

        if( rank > maxRankCount )
            maxRankCount = rank;
        // Create the space and store the updates. If there are no dense
        // updates, then mark two more matrices for QR factorization.
        {
            const int oldRank = F.Rank();
            const int height = F.Height();
            const int width = F.Width();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( F.U, UCopy );
            F.U.Resize( height, rank, height );
            F.U.Init();
            MemCopy
            ( F.U.Buffer(0,rank-oldRank), UCopy.LockedBuffer(),
              height*oldRank );

            Dense<Scalar> VCopy;
            hmat_tools::Copy( F.V, VCopy );
            F.V.Resize( width, rank, width );
            F.V.Init();
            MemCopy
            ( F.V.Buffer(0,rank-oldRank), VCopy.LockedBuffer(), width*oldRank );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( inTargetTeam_ )
        {
            // Combine all of the U's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int m = Height();
            const int numLowRankUpdates = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numLowRankUpdates; ++i,UMap_.Increment() )
                rank += UMap_.CurrentEntry()->Width();

            if( numLowRankUpdates == 0 )
                UMap_.Set( -1, new Dense<Scalar>( m, rank ) );
            else
            {
                UMap_.ResetIterator();
                Dense<Scalar>& firstU = *UMap_.CurrentEntry();
                UMap_.Increment();

                Dense<Scalar> firstUCopy;
                hmat_tools::Copy( firstU, firstUCopy );

                firstU.Resize( m, rank, m );
                firstU.Init();
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstUCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        MemCopy
                        ( firstU.Buffer(0,rOffset+j),
                          firstUCopy.LockedBuffer(0,j), m );
                }
                // Push the rest of the updates in and then erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& U = *UMap_.CurrentEntry();
                    const int r = U.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        MemCopy
                        ( firstU.Buffer(0,rOffset+j), U.LockedBuffer(0,j),
                          m );
                    UMap_.EraseCurrentEntry();
                }
            }
        }
        else
        {
            // Combine all of the V's into a single buffer with enough space
            // for the parent ranks to fit at the beginning
            const int n = Width();
            const int numLowRankUpdates = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numLowRankUpdates; ++i,VMap_.Increment() )
                rank += VMap_.CurrentEntry()->Width();

            if( numLowRankUpdates == 0 )
                VMap_.Set( -1, new Dense<Scalar>( n, rank ) );
            else
            {
                VMap_.ResetIterator();
                Dense<Scalar>& firstV = *VMap_.CurrentEntry();
                VMap_.Increment();

                Dense<Scalar> firstVCopy;
                hmat_tools::Copy( firstV, firstVCopy );

                firstV.Resize( n, rank, n );
                firstV.Init();
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstVCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        MemCopy
                        ( firstV.Buffer(0,rOffset+j),
                          firstVCopy.LockedBuffer(0,j), n );
                }
                // Push the rest of the updates in and erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& V = *VMap_.CurrentEntry();
                    const int r = V.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        MemCopy
                        ( firstV.Buffer(0,rOffset+j), V.LockedBuffer(0,j),
                          n );
                    VMap_.EraseCurrentEntry();
                }
            }
        }
        if( rank > maxRankCount )
            maxRankCount = rank;
        break;
    }
    case DENSE:
    {
        // Condense all of the U's and V's onto the dense matrix
        Dense<Scalar>& D = *block_.data.D;
        const int m = Height();
        const int n = Width();
        const int numLowRankUpdates = UMap_.Size();

        UMap_.ResetIterator();
        VMap_.ResetIterator();
        for( int update=0; update<numLowRankUpdates; ++update )
        {
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();
            hmat_tools::MultiplyTranspose
			( Scalar(1), U, V, Scalar(1), D );
            UMap_.EraseCurrentEntry();
            VMap_.EraseCurrentEntry();
        }

        // Create space for storing the parent updates
        UMap_.Set( -1, new Dense<Scalar>(m,rank) );
        VMap_.Set( -1, new Dense<Scalar>(n,rank) );
        if( rank > maxRankCount )
            maxRankCount = rank;
        break;
    }
    default:
        break;
    }
}


template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressLowRankImport( int rank )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressLowRankImport");
#endif
    if( Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        mpi::Comm team = teams_->Team( level_ );
        const int teamsize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        int newRank = rank;
        if( teamsize == 2 )
        {
            if( inTargetTeam_ )
            {
                const int tStart = (teamRank==0 ? 0 : 2);
                const int tStop = (teamRank==0 ? 2 : 4);
                const int numEntries = UMap_.Size();
                UMap_.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& ULocal = *UMap_.CurrentEntry();
                    const int r = ULocal.Width();
                    Dense<Scalar> ULocalSub;
                    for( int t=tStart,tOffset=0; t<tStop;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        ULocalSub.LockedView
                        ( ULocal, tOffset, 0, node.targetSizes[t], r );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyHMatCompressImportU
                            ( newRank, ULocalSub );
                    }
                    newRank += r;
                    UMap_.EraseCurrentEntry();
                }
            }
            else
                UMap_.Clear();

            if( inSourceTeam_ )
            {
                newRank = rank;
                const int sStart = (teamRank==0 ? 0 : 2);
                const int sStop = (teamRank==0 ? 2 : 4);
                const int numEntries = VMap_.Size();
                VMap_.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& VLocal = *VMap_.CurrentEntry();
                    const int r = VLocal.Width();
                    Dense<Scalar> VLocalSub;
                    for( int s=sStart,sOffset=0; s<sStop;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        VLocalSub.LockedView
                        ( VLocal, sOffset, 0, node.sourceSizes[s], r );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyHMatCompressImportV
                            ( newRank, VLocalSub );
                    }
                    newRank += r;
                    VMap_.EraseCurrentEntry();
                }
            }
            else
                VMap_.Clear();
        }
        else // teamsize >= 4
        {
            if( inTargetTeam_ )
            {
                const int numEntries = UMap_.Size();
                UMap_.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& U = *UMap_.CurrentEntry();
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyHMatCompressImportU
                            ( newRank, U );
                    newRank += U.Width();
                    UMap_.EraseCurrentEntry();
                }
            }
            else
                UMap_.Clear();

            if( inSourceTeam_ )
            {
                newRank = rank;
                const int numEntries = VMap_.Size();
                VMap_.ResetIterator();
                for( int i=0; i<numEntries; ++i )
                {
                    const Dense<Scalar>& V = *VMap_.CurrentEntry();
                    for( int s=0; s<4; ++s )
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyHMatCompressImportV
                            ( newRank, V );
                    newRank += V.Width();
                    VMap_.EraseCurrentEntry();
                }
            }
            else
                VMap_.Clear();
        }
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressLowRankImport( newRank );
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        int newRank = rank;
        if( inTargetTeam_ )
        {
            const int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& U = *UMap_.CurrentEntry();
                Dense<Scalar> ULocal;

                for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                {
                    ULocal.LockedView
                    ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).MultiplyHMatCompressImportU
                        ( newRank, ULocal );
                }
                newRank += U.Width();
                UMap_.EraseCurrentEntry();
            }
        }
        if( inSourceTeam_ )
        {
            newRank = rank;
            const int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i )
            {
                const Dense<Scalar>& V = *VMap_.CurrentEntry();
                Dense<Scalar> VLocal;

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    VLocal.LockedView
                    ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                    for( int t=0; t<4; ++t )
                        node.Child(t,s).MultiplyHMatCompressImportV
                        ( newRank, VLocal );
                }
                newRank += V.Width();
                VMap_.EraseCurrentEntry();
            }
        }
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressLowRankImport( newRank );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        int newRank = rank;
        if( inTargetTeam_ )
        {
            Dense<Scalar>* mainU;
            if( block_.type == DIST_LOW_RANK )
                mainU = &block_.data.DF->ULocal;
            else if( block_.type == SPLIT_LOW_RANK )
                mainU = &block_.data.SF->D;
            else
                mainU = &block_.data.F->U;

            int numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& X = *colXMap_.CurrentEntry();
                const int m = X.Height();
                const int r = X.Width();
                for( int j=0; j<r; ++j )
                    MemCopy
                    ( mainU->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      m );
                colXMap_.EraseCurrentEntry();
                newRank += r;
            }

            numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& U = *UMap_.CurrentEntry();
                const int m = U.Height();
                const int r = U.Width();
                for( int j=0; j<r; ++j )
                    MemCopy
                    ( mainU->Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m );
                UMap_.EraseCurrentEntry();
                newRank += r;
            }
        }
        else
        {
            UMap_.Clear();
            colXMap_.Clear();
        }

        if( inSourceTeam_ )
        {
            newRank = rank;

            Dense<Scalar>* mainV;
            if( block_.type == DIST_LOW_RANK )
                mainV = &block_.data.DF->VLocal;
            else if( block_.type == SPLIT_LOW_RANK )
                mainV = &block_.data.SF->D;
            else
                mainV = &block_.data.F->V;

            int numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                const Dense<Scalar>& X = *rowXMap_.CurrentEntry();
                const int n = X.Height();
                const int r = X.Width();
                for( int j=0; j<r; ++j )
                    MemCopy
                    ( mainV->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      n );
                rowXMap_.EraseCurrentEntry();
                newRank += r;
            }

            numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int entry=0; entry<numEntries; ++entry )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                const int n = V.Height();
                const int r = V.Width();
                for( int j=0; j<r; ++j )
                    MemCopy
                    ( mainV->Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n );
                VMap_.EraseCurrentEntry();
                newRank += r;
            }
        }
        else
        {
            VMap_.Clear();
            rowXMap_.Clear();
        }
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
        break;
    default:
        colXMap_.Clear();
        rowXMap_.Clear();
        UMap_.Clear();
        VMap_.Clear();
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressImportU
( int rank, const Dense<Scalar>& U )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressImportU");
#endif
    if( !inTargetTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        mpi::Comm team = teams_->Team( level_ );
        const int teamsize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamsize == 2 )
        {
            const int tStart = (teamRank==0 ? 0 : 2);
            const int tStop = (teamRank==0 ? 2 : 4);
            Dense<Scalar> USub;
            for( int t=tStart,tOffset=0; t<tStop;
                 tOffset+=node.targetSizes[t],++t )
            {
                USub.LockedView
                ( U, tOffset, 0, node.targetSizes[t], U.Width() );
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressImportU( rank, USub );
            }
        }
        else  // teamsize >= 4
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressImportU( rank, U );
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        Dense<Scalar> USub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            USub.LockedView( U, tOffset, 0, node.targetSizes[t], U.Width() );
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressImportU( rank, USub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *block_.data.DF;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            MemCopy( DF.ULocal.Buffer(0,rank+j), U.LockedBuffer(0,j), m );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            MemCopy( SF.D.Buffer(0,rank+j), U.LockedBuffer(0,j), m );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            MemCopy( F.U.Buffer(0,rank+j), U.LockedBuffer(0,j), m );
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
    {
        UMap_.ResetIterator();
        Dense<Scalar>& mainU = *UMap_.CurrentEntry();
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            MemCopy( mainU.Buffer(0,rank+j), U.LockedBuffer(0,j), m );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressImportV
( int rank, const Dense<Scalar>& V )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressImportV");
#endif
    if( !inSourceTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        Node& node = *block_.data.N;
        mpi::Comm team = teams_->Team( level_ );
        const int teamsize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamsize == 2 )
        {
            const int sStart = (teamRank==0 ? 0 : 2);
            const int sStop = (teamRank==0 ? 2 : 4);
            Dense<Scalar> VSub;
            for( int s=sStart,sOffset=0; s<sStop;
                 sOffset+=node.sourceSizes[s],++s )
            {
                VSub.LockedView
                ( V, sOffset, 0, node.sourceSizes[s], V.Width() );
                for( int t=0; t<4; ++t )
                    node.Child(t,s).MultiplyHMatCompressImportV( rank, VSub );
            }
        }
        else  // teamsize >= 4
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressImportV( rank, V );
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        Dense<Scalar> VSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            VSub.LockedView( V, sOffset, 0, node.sourceSizes[s], V.Width() );
            for( int t=0; t<4; ++t )
                node.Child(t,s).MultiplyHMatCompressImportV( rank, VSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *block_.data.DF;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            MemCopy( DF.VLocal.Buffer(0,rank+j), V.LockedBuffer(0,j), n );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            MemCopy( SF.D.Buffer(0,rank+j), V.LockedBuffer(0,j), n );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            MemCopy( F.V.Buffer(0,rank+j), V.LockedBuffer(0,j), n );
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
    {
        VMap_.ResetIterator();
        Dense<Scalar>& mainV = *VMap_.CurrentEntry();
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            MemCopy( mainV.Buffer(0,rank+j), V.LockedBuffer(0,j), n );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPrecompute()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPrecompute");
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
                    node.Child(t,s).MultiplyHMatCompressFPrecompute();
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;

        if( inTargetTeam_ && totalrank > MaxRank() )
        {
            USqrEig_.Resize( totalrank );
            Dense<Scalar>& U = DF.ULocal;

            hmat_tools::AdjointMultiply( Scalar(1), U, U, USqr_ );
        }

        totalrank=DF.rank;
        if( inSourceTeam_ && totalrank > MaxRank() )
        {
            VSqrEig_.Resize( totalrank );
            Dense<Scalar>& V = DF.VLocal;

            hmat_tools::AdjointMultiply( Scalar(1), V, V, VSqr_ );
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

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
                USqrEig_.Resize( totalrank );
                Dense<Scalar>& U = SF.D;

                hmat_tools::AdjointMultiply( Scalar(1), U, U, USqr_ );
            }

            if( inSourceTeam_ )
            {
                VSqrEig_.Resize( totalrank );
                Dense<Scalar>& V = SF.D;

                hmat_tools::AdjointMultiply( Scalar(1), V, V, VSqr_ );
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

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            USqrEig_.Resize( totalrank );

            hmat_tools::AdjointMultiply( Scalar(1), F.U, F.U, USqr_ );

            VSqrEig_.Resize( totalrank );

            hmat_tools::AdjointMultiply( Scalar(1), F.V, F.V, VSqr_ );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReduces()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFReduces");
#endif

    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    Vector<int> sizes( numReduces, 0 );
    MultiplyHMatCompressFReducesCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numReduces; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFReducesPack
    ( buffer, offsetscopy );

    MultiplyHMatCompressFTreeReduces( buffer, sizes );

    MultiplyHMatCompressFReducesUnpack
    ( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReducesCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFReduceCount");
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
                node.Child(t,s).MultiplyHMatCompressFReducesCount
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
            sizes[level_] += USqr_.Height()*USqr_.Width();
        if( inSourceTeam_ )
            sizes[level_] += VSqr_.Height()*VSqr_.Width();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReducesPack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFReducePack");
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
                node.Child(t,s).MultiplyHMatCompressFReducesPack
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
            int size=USqr_.Height()*USqr_.Width();
            MemCopy( &buffer[offsets[level_]], USqr_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                USqr_.Clear();
        }

        if( inSourceTeam_ )
        {
            int size=VSqr_.Height()*VSqr_.Width();
            MemCopy( &buffer[offsets[level_]], VSqr_.LockedBuffer(), size );
            offsets[level_] += size;
            if( teamRank != 0 )
                VSqr_.Clear();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFTreeReduces
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeReduces");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReducesUnpack
( const Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFReducesUnpack");
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
                node.Child(t,s).MultiplyHMatCompressFReducesUnpack
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
                int size=USqr_.Height()*USqr_.Width();
                MemCopy( USqr_.Buffer(), &buffer[offsets[level_]], size );
                offsets[level_] += size;
            }

            if( inSourceTeam_ )
            {
                int size=VSqr_.Height()*VSqr_.Width();
                MemCopy( VSqr_.Buffer(), &buffer[offsets[level_]], size );
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
DistHMat2d<Scalar>::MultiplyHMatCompressFEigenDecomp()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFEigenDecomp");
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
                node.Child(t,s).MultiplyHMatCompressFEigenDecomp();
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
            // Calculate Eigenvalues of Squared Matrix
            if( inTargetTeam_ )
            {
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 1 );
#endif
                lapack::EVD
                ( 'V', 'U', USqr_.Height(),
                  USqr_.Buffer(), USqr_.LDim(), &USqrEig_[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 1 );
#endif
            }

            if( inSourceTeam_ )
            {
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 1 );
#endif
                lapack::EVD
                ( 'V', 'U', VSqr_.Height(),
                  VSqr_.Buffer(), VSqr_.LDim(), &VSqrEig_[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 1 );
#endif
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 1 );
#endif
                lapack::EVD
                ( 'V', 'U', USqr_.Height(),
                  USqr_.Buffer(), USqr_.LDim(), &USqrEig_[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 1 );
#endif
            }

            if( inSourceTeam_ )
            {
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 1 );
#endif
                lapack::EVD
                ( 'V', 'U', VSqr_.Height(),
                  VSqr_.Buffer(), VSqr_.LDim(), &VSqrEig_[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 1 );
#endif
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

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
#ifdef TIME_MULTIPLY
            timerGlobal.Start( 1 );
#endif
            lapack::EVD
            ( 'V', 'U', USqr_.Height(),
              USqr_.Buffer(), USqr_.LDim(), &USqrEig_[0] );

            lapack::EVD
            ( 'V', 'U', VSqr_.Height(),
              VSqr_.Buffer(), VSqr_.LDim(), &VSqrEig_[0] );
#ifdef TIME_MULTIPLY
            timerGlobal.Stop( 1 );
#endif
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrix()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrix");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendsizes, recvsizes;
    MultiplyHMatCompressFPassMatrixCount
    ( sendsizes, recvsizes );

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
    MultiplyHMatCompressFPassMatrixPack
    ( sendBuffer, offsets );

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
    MultiplyHMatCompressFPassMatrixUnpack
    ( recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrixCount
( std::map<int,int>& sendsizes, std::map<int,int>& recvsizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrixCount");
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
                node.Child(t,s).MultiplyHMatCompressFPassMatrixCount
                ( sendsizes, recvsizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inSourceTeam_ )
                AddToMap( sendsizes, targetRoot_, VSqr_.Height()*VSqr_.Width() );
            if( inTargetTeam_ )
                AddToMap( recvsizes, sourceRoot_, USqr_.Height()*USqr_.Width() );
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inSourceTeam_ )
                AddToMap( sendsizes, targetRoot_, VSqr_.Height()*VSqr_.Width() );
            else
                AddToMap( recvsizes, sourceRoot_, USqr_.Height()*USqr_.Width() );
        }
        else
        {
            if( inSourceTeam_ )
            {
                AddToMap( sendsizes, targetRoot_, SF.D.Height()*SF.D.Width() );
                AddToMap( recvsizes, targetRoot_, LH*SF.D.Width() );
            }
            else
            {
                AddToMap( sendsizes, sourceRoot_, SF.D.Height()*SF.D.Width() );
                AddToMap( recvsizes, sourceRoot_, LW*SF.D.Width() );
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrixPack
( Vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrixPack");
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
                node.Child(t,s).MultiplyHMatCompressFPassMatrixPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            MemCopy
            ( &buffer[offsets[targetRoot_]], VSqr_.LockedBuffer(),
              VSqr_.Height()*VSqr_.Width() );
            offsets[targetRoot_] += VSqr_.Height()*VSqr_.Width();
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
                break;
            MemCopy
            ( &buffer[offsets[targetRoot_]], VSqr_.LockedBuffer(),
              VSqr_.Height()*VSqr_.Width() );
            offsets[targetRoot_] += VSqr_.Height()*VSqr_.Width();
        }
        else
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
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrixUnpack
( const Vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrixUnpack");
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
                node.Child(t,s).MultiplyHMatCompressFPassMatrixUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            VSqr_.Resize( USqr_.Height(), USqr_.Width() );
            VSqr_.Init();
            MemCopy
            ( VSqr_.Buffer(), &buffer[offsets[sourceRoot_]],
              VSqr_.Height()*VSqr_.Width() );
            offsets[sourceRoot_] += VSqr_.Height()*VSqr_.Width();
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inSourceTeam_ )
                break;
            VSqr_.Resize( USqr_.Height(), USqr_.Width() );
            VSqr_.Init();
            MemCopy
            ( VSqr_.Buffer(), &buffer[offsets[sourceRoot_]],
              VSqr_.Height()*VSqr_.Width() );
            offsets[sourceRoot_] += VSqr_.Height()*VSqr_.Width();
        }
        else
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
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVector()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassVector");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendsizes, recvsizes;
    MultiplyHMatCompressFPassVectorCount
    ( sendsizes, recvsizes );

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
    Vector<Real> sendBuffer(totalSendsize);
    std::map<int,int> offsets = sendOffsets;
    MultiplyHMatCompressFPassVectorPack
    ( sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvsizes.size();
    Vector<mpi::Request> recvRequests( numRecvs );
    Vector<Real> recvBuffer( totalRecvsize );
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
    MultiplyHMatCompressFPassVectorUnpack
    ( recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVectorCount
( std::map<int,int>& sendsizes, std::map<int,int>& recvsizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassVectorCount");
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
                node.Child(t,s).MultiplyHMatCompressFPassVectorCount
                ( sendsizes, recvsizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inSourceTeam_ )
                AddToMap( sendsizes, targetRoot_, VSqrEig_.Size() );
            else
                AddToMap( recvsizes, sourceRoot_, USqrEig_.Size() );
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inSourceTeam_ )
                AddToMap( sendsizes, targetRoot_, VSqrEig_.Size() );
            else
                AddToMap( recvsizes, sourceRoot_, USqrEig_.Size() );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVectorPack
( Vector<Real>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassVectorPack");
#endif
    if( !inSourceTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressFPassVectorPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            MemCopy
            ( &buffer[offsets[targetRoot_]], &VSqrEig_[0], VSqrEig_.Size() );
            offsets[targetRoot_] += VSqrEig_.Size();
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            MemCopy
            ( &buffer[offsets[targetRoot_]], &VSqrEig_[0], VSqrEig_.Size() );
            offsets[targetRoot_] += VSqrEig_.Size();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVectorUnpack
( const Vector<Real>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassVectorUnpack");
#endif
    if( !inTargetTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressFPassVectorUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            VSqrEig_.Resize( USqrEig_.Size() );
            MemCopy
            ( &VSqrEig_[0], &buffer[offsets[sourceRoot_]], VSqrEig_.Size() );
            offsets[sourceRoot_] += VSqrEig_.Size();
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            VSqrEig_.Resize( USqrEig_.Size() );
            MemCopy
            ( &VSqrEig_[0], &buffer[offsets[sourceRoot_]], VSqrEig_.Size() );
            offsets[sourceRoot_] += VSqrEig_.Size();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFMidcompute
( Real relTol, Real twoNorm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFMidcompute");
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
                node.Child(t,s).MultiplyHMatCompressFMidcompute
                ( relTol, twoNorm );
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
        if( teamRank == 0 && inTargetTeam_ )
        {
            const int k = USqr_.Width();
            Dense<Scalar> BSqr;
            Vector<Real> USqrSigma(k), VSqrSigma(k);
            BSqrU_.Resize( k, k );
            BSqrU_.Init();
            BSqrVH_.Resize( k, k );
            BSqrVH_.Init();
            BSigma_.Resize( k );

            if( k==0 )
                break;

            hmat_tools::TransposeMultiply( Scalar(1), USqr_, VSqr_, BSqr );

            hmat_tools::Conjugate( BSqr );
            for( int i=0; i<k; ++i)
                USqrSigma[i] = sqrt( std::max( USqrEig_[i], Real(0) ) );
            for( int i=0; i<k; ++i)
                VSqrSigma[i] = sqrt( std::max( VSqrEig_[i], Real(0) ) );

            for( int j=0; j<k; ++j)
                for( int i=0; i<k; ++i)
                    BSqr.Set(i,j, BSqr.Get(i,j)*USqrSigma[i]*VSqrSigma[j]);


            const int lwork = lapack::SVDWorkSize(k,k);
            const int lrwork = lapack::SVDRealWorkSize(k,k);
            Vector<Scalar> work(lwork);
            Vector<Real> rwork(lrwork);
#ifdef TIME_MULTIPLY
            timerGlobal.Start( 0 );
#endif
            lapack::SVD
            ('S', 'S' ,k ,k,
             BSqr.Buffer(), BSqr.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );
#ifdef TIME_MULTIPLY
            timerGlobal.Stop( 0 );
#endif

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol, twoNorm );
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
        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
            {
                const int k = USqr_.Width();
                Dense<Scalar> BSqr;
                Vector<Real> USqrSigma(k), VSqrSigma(k);
                BSqrU_.Resize( k, k );
                BSqrU_.Init();
                BSqrVH_.Resize( k, k );
                BSqrVH_.Init();
                BSigma_.Resize( k );

                if( k==0 )
                    break;

                hmat_tools::TransposeMultiply( Scalar(1), USqr_, VSqr_, BSqr );

                hmat_tools::Conjugate( BSqr );
                for( int i=0; i<k; ++i)
                    USqrSigma[i] = sqrt( std::max( USqrEig_[i], Real(0) ) );
                for( int i=0; i<k; ++i)
                    VSqrSigma[i] = sqrt( std::max( VSqrEig_[i], Real(0) ) );

                for( int j=0; j<k; ++j)
                    for( int i=0; i<k; ++i)
                        BSqr.Set(i,j, BSqr.Get(i,j)*USqrSigma[i]*VSqrSigma[j]);


                const int lwork = lapack::SVDWorkSize(k,k);
                const int lrwork = lapack::SVDRealWorkSize(k,k);
                Vector<Scalar> work(lwork);
                Vector<Real> rwork(lrwork);
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 0 );
#endif
                lapack::SVD
                ('S', 'S' ,k ,k,
                 BSqr.Buffer(), BSqr.LDim(), &BSigma_[0],
                 BSqrU_.Buffer(), BSqrU_.LDim(),
                 BSqrVH_.Buffer(), BSqrVH_.LDim(),
                 &work[0], lwork, &rwork[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 0 );
#endif

                SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol, twoNorm );
            }
        }
        else
        {
            Dense<Scalar> B;
            if( inSourceTeam_ )
                hmat_tools::MultiplyTranspose( Scalar(1), SFD_, SF.D, B );
            else
                hmat_tools::MultiplyTranspose( Scalar(1), SF.D, SFD_, B );

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
#ifdef TIME_MULTIPLY
            timerGlobal.Start( 0 );
#endif
            lapack::SVD
            ('S', 'S' , LH, LW,
             B.Buffer(), B.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );
#ifdef TIME_MULTIPLY
            timerGlobal.Stop( 0 );
#endif

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol, twoNorm );
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
        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            const int k = USqr_.Width();
            Dense<Scalar> BSqr;
            Vector<Real> USqrSigma(k), VSqrSigma(k);
            BSqrU_.Resize( k, k );
            BSqrU_.Init();
            BSqrVH_.Resize( k, k );
            BSqrVH_.Init();
            BSigma_.Resize( k );

            if( k==0 )
                break;

            hmat_tools::TransposeMultiply( Scalar(1), USqr_, VSqr_, BSqr );

            hmat_tools::Conjugate( BSqr );
            for( int i=0; i<k; ++i)
                USqrSigma[i] = sqrt( std::max( USqrEig_[i], Real(0) ) );
            for( int i=0; i<k; ++i)
                VSqrSigma[i] = sqrt( std::max( VSqrEig_[i], Real(0) ) );

            for( int j=0; j<k; ++j)
                for( int i=0; i<k; ++i)
                    BSqr.Set(i,j, BSqr.Get(i,j)*USqrSigma[i]*VSqrSigma[j]);


            const int lwork = lapack::SVDWorkSize(k,k);
            const int lrwork = lapack::SVDRealWorkSize(k,k);
            Vector<Scalar> work(lwork);
            Vector<Real> rwork(lrwork);
#ifdef TIME_MULTIPLY
            timerGlobal.Start( 0 );
#endif
            lapack::SVD
            ('S', 'S' ,k ,k,
             BSqr.Buffer(), BSqr.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );
#ifdef TIME_MULTIPLY
            timerGlobal.Stop( 0 );
#endif

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol, twoNorm );
        }
        else
        {
            Dense<Scalar> B;
            hmat_tools::MultiplyTranspose( Scalar(1), F.U, F.V, B );

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
#ifdef TIME_MULTIPLY
            timerGlobal.Start( 0 );
#endif
            lapack::SVD
            ('S', 'S' , LH, LW,
             B.Buffer(), B.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );
#ifdef TIME_MULTIPLY
            timerGlobal.Stop( 0 );
#endif

            SVDTrunc( BSqrU_, BSigma_, BSqrVH_, relTol, twoNorm );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNum()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackNum");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendsizes, recvsizes;
    MultiplyHMatCompressFPassbackNumCount
    ( sendsizes, recvsizes );

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
    Vector<int> sendBuffer(totalSendsize);
    std::map<int,int> offsets = sendOffsets;
    MultiplyHMatCompressFPassbackNumPack
    ( sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvsizes.size();
    Vector<mpi::Request> recvRequests( numRecvs );
    Vector<int> recvBuffer( totalRecvsize );
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
    MultiplyHMatCompressFPassbackNumUnpack
    ( recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNumCount
( std::map<int,int>& sendsizes, std::map<int,int>& recvsizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackNumCount");
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
                node.Child(t,s).MultiplyHMatCompressFPassbackNumCount
                ( sendsizes, recvsizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
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
                AddToMap( sendsizes, sourceRoot_, 1 );
            else
                AddToMap( recvsizes, targetRoot_, 1 );
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
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            if( inTargetTeam_ )
                AddToMap( sendsizes, sourceRoot_, 1 );
            else
                AddToMap( recvsizes, targetRoot_, 1 );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNumPack
( Vector<int>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackNumPack");
#endif
    if( !inTargetTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressFPassbackNumPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            buffer[offsets[sourceRoot_]]=BSqrVH_.Height();
            offsets[sourceRoot_] ++;
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            buffer[offsets[sourceRoot_]]=BSqrVH_.Height();
            offsets[sourceRoot_] ++;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNumUnpack
( const Vector<int>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackNumUnpack");
#endif
    if( !inSourceTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressFPassbackNumUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            BSqrVH_.Resize
            ( buffer[offsets[targetRoot_]], VSqr_.Height() );
            BSqrVH_.Init();
            offsets[targetRoot_] ++;
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        SplitLowRank& SF = *block_.data.SF;
        const int LH = Height();
        const int LW = Width();
        // Here we guarantee that all the data are in SF.D
        const int totalrank = SF.rank;

        if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
        {
            BSqrVH_.Resize
            ( buffer[offsets[targetRoot_]], VSqr_.Height() );
            BSqrVH_.Init();
            offsets[targetRoot_] ++;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackData()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackData");
#endif

    // Compute send and recv sizes
    std::map<int,int> sendsizes, recvsizes;
    MultiplyHMatCompressFPassbackDataCount
    ( sendsizes, recvsizes );

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
    MultiplyHMatCompressFPassbackDataPack
    ( sendBuffer, offsets );

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
    MultiplyHMatCompressFPassbackDataUnpack
    ( recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackDataCount
( std::map<int,int>& sendsizes, std::map<int,int>& recvsizes )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackDataCount");
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
                node.Child(t,s).MultiplyHMatCompressFPassbackDataCount
                ( sendsizes, recvsizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
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
                AddToMap( sendsizes, sourceRoot_, BSqrVH_.Height()*BSqrVH_.Width() );
            else
                AddToMap( recvsizes, targetRoot_, BSqrVH_.Height()*BSqrVH_.Width() );
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

        if( !haveDenseUpdate_ )
        {
            if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
            {
                if( inTargetTeam_ )
                    AddToMap( sendsizes, sourceRoot_,
                              BSqrVH_.Height()*BSqrVH_.Width() );
                else
                    AddToMap( recvsizes, targetRoot_,
                              BSqrVH_.Height()*BSqrVH_.Width() );
            }
        }
        else
        {
            if( inTargetTeam_ )
            {
                AddToMap( sendsizes, sourceRoot_, Height()*SF.D.Width() );
                AddToMap( recvsizes, sourceRoot_,
                          Width()*SF.D.Width()+Width()*Height() );
            }
            else
            {
                AddToMap( sendsizes, targetRoot_,
                          Width()*SF.D.Width()+Width()*Height() );
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackDataPack
( Vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackDataPack");
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
                node.Child(t,s).MultiplyHMatCompressFPassbackDataPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ || !inTargetTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int size=BSqrVH_.Height()*BSqrVH_.Width();
            MemCopy
            ( &buffer[offsets[sourceRoot_]], BSqrVH_.LockedBuffer(), size );
            offsets[sourceRoot_] += size;
            BSqrVH_.Clear();
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

        if( !haveDenseUpdate_ )
        {
            if( inSourceTeam_ )
                break;
            if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
            {
                int size=BSqrVH_.Height()*BSqrVH_.Width();
                MemCopy
                ( &buffer[offsets[sourceRoot_]], BSqrVH_.LockedBuffer(), size );
                offsets[sourceRoot_] += size;
                BSqrVH_.Clear();
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackDataUnpack
( const Vector<Scalar>& buffer, std::map<int,int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackDataUnpack");
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
                node.Child(t,s).MultiplyHMatCompressFPassbackDataUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ || !inSourceTeam_ )
            break;
        DistLowRank &DF = *block_.data.DF;
        int totalrank = DF.rank;
        if( totalrank <= MaxRank() )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int size=BSqrVH_.Height()*BSqrVH_.Width();
            MemCopy
            ( BSqrVH_.Buffer(), &buffer[offsets[targetRoot_]], size );
            offsets[targetRoot_] += size;
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
        if( !haveDenseUpdate_ )
        {
            if( inTargetTeam_ )
                break;
            if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
            {
                int size=BSqrVH_.Height()*BSqrVH_.Width();
                MemCopy
                ( BSqrVH_.Buffer(), &buffer[offsets[targetRoot_]], size );
                offsets[targetRoot_] += size;
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPostcompute
( Real epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPostcompute");
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
                node.Child(t,s).MultiplyHMatCompressFPostcompute
                ( epsilon );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( block_.type == LOW_RANK )
            {
                LowRank<Scalar> &F = *block_.data.F;
                const int LH = Height();
                const int LW = Width();
                const int totalrank = F.U.Width();
                if( LH <= totalrank || LW <= totalrank || totalrank <= MaxRank() )
                    break;
            }
            if( block_.type == SPLIT_LOW_RANK )
            {
                SplitLowRank& SF = *block_.data.SF;
                const int LH = Height();
                const int LW = Width();
                const int totalrank = SF.D.Width();
                if( LH <= totalrank || LW <= totalrank || totalrank <= MaxRank() )
                    break;
            }
            if( block_.type == DIST_LOW_RANK )
            {
                DistLowRank &DF = *block_.data.DF;
                int totalrank = DF.rank;
                if( totalrank <= MaxRank() )
                    break;
            }
            const int kU = USqrEig_.Size();
            if( inTargetTeam_ )
            {
                const Real maxEig = std::max( USqrEig_[kU-1], Real(0) );
                const Real tolerance = sqrt(epsilon*maxEig*kU);
                for( int j=0; j<kU; ++j )
                {
                    const Real omega = std::max( USqrEig_[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<kU; ++i )
                            USqr_.Set(i,j,USqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( USqr_.Buffer(0,j), kU );
                }

                const int m = BSqrU_.Height();
                const int n = BSqrU_.Width();
                for( int j=0; j<n; ++j )
                    for( int i=0; i<m; ++i )
                        BSqrU_.Set(i,j,BSqrU_.Get(i,j)*BSigma_[j]);

                hmat_tools::Multiply( Scalar(1), USqr_, BSqrU_, BL_ );
                USqr_.Clear();
                BSqrU_.Clear();
            }

            const int kV = VSqrEig_.Size();
            if( inSourceTeam_ && kV > 0 )
            {
                const Real maxEig = std::max( VSqrEig_[kV-1], Real(0) );
                const Real tolerance = sqrt(epsilon*maxEig*kV);
                for( int j=0; j<kV; ++j )
                {
                    const Real omega = std::max( VSqrEig_[j], Real(0) );
                    const Real sqrtOmega = sqrt( omega );
                    if( sqrtOmega > tolerance )
                        for( int i=0; i<kV; ++i )
                            VSqr_.Set(i,j,VSqr_.Get(i,j)/sqrtOmega);
                    else
                        MemZero( VSqr_.Buffer(0,j), kV );
                }

                hmat_tools::MultiplyTranspose( Scalar(1), VSqr_, BSqrVH_, BR_ );
                VSqr_.Clear();
                BSqrVH_.Clear();
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNum()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsNum");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatCompressFBroadcastsNumCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<int> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFBroadcastsNumPack
    ( buffer, offsetscopy );

    MultiplyHMatCompressFTreeBroadcastsNum( buffer, sizes );

    MultiplyHMatCompressFBroadcastsNumUnpack
    ( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNumCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsNumCount");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsNumCount
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNumPack
( Vector<int>& buffer, Vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsNumPack");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsNumPack
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
DistHMat2d<Scalar>::MultiplyHMatCompressFTreeBroadcastsNum
( Vector<int>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeBroadcastsNum");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNumUnpack
( Vector<int>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsNumUnpack");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsNumUnpack
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
                (totalrank, buffer[offsets[level_]]);
                BL_.Init();
                offsets[level_]++;
            }
            if( inSourceTeam_ )
            {
                BR_.Resize
                (totalrank, buffer[offsets[level_]]);
                BR_.Init();
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcasts()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    Vector<int> sizes( numBroadcasts, 0 );
    MultiplyHMatCompressFBroadcastsCount( sizes );

    int totalsize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalsize += sizes[i];
    Vector<Scalar> buffer( totalsize );
    Vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    Vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFBroadcastsPack
    ( buffer, offsetscopy );

    MultiplyHMatCompressFTreeBroadcasts( buffer, sizes );

    MultiplyHMatCompressFBroadcastsUnpack
    ( buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsCount
( Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsCount");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsCount
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsPack
( Vector<Scalar>& buffer, Vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsPack");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsPack
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
DistHMat2d<Scalar>::MultiplyHMatCompressFTreeBroadcasts
( Vector<Scalar>& buffer, Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeBroadcasts");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsUnpack
( Vector<Scalar>& buffer, Vector<int>& offsets )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsUnpack");
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
                node.Child(t,s).MultiplyHMatCompressFBroadcastsUnpack
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
DistHMat2d<Scalar>::MultiplyHMatCompressFFinalcompute
( Real relTol, Real twoNorm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFFinalcompute");
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
                node.Child(t,s).MultiplyHMatCompressFFinalcompute
                ( relTol, twoNorm );
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
            Dense<Scalar> Utmp;
            hmat_tools::Copy(U, Utmp);
            DF.rank = BL_.Width();
            hmat_tools::Multiply( Scalar(1), Utmp, BL_, U );
            BL_.Clear();
        }
        if( inSourceTeam_ )
        {
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &V = DF.VLocal;
            Dense<Scalar> Vtmp;
            hmat_tools::Copy(V, Vtmp);
            DF.rank = BR_.Width();
            hmat_tools::Multiply( Scalar(1), Vtmp, BR_, V );
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
            if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
            {
                if( inTargetTeam_ )
                {
                    Dense<Scalar> &U = SF.D;
                    Dense<Scalar> Utmp;
                    hmat_tools::Copy(U, Utmp);
                    SF.rank = BL_.Width();
                    hmat_tools::Multiply( Scalar(1), Utmp, BL_, U );
                    BL_.Clear();
                }
                if( inSourceTeam_ )
                {
                    Dense<Scalar> &V = SF.D;
                    Dense<Scalar> Vtmp;
                    hmat_tools::Copy(V, Vtmp);
                    SF.rank = BR_.Width();
                    hmat_tools::Multiply( Scalar(1), Vtmp, BR_, V );
                    BR_.Clear();
                }
            }
            else
            {
                SF.rank = BSqrU_.Width();
                if( inTargetTeam_ )
                {
                    hmat_tools::Copy( BSqrU_, SF.D );
                }
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
            {
                SF.rank = maxRank;
                Dense<Scalar> VH( std::min(m,n), n );
                Dense<Scalar> U( m, std::min(m,n) );
                Vector<Real> sigma( minDim );
                Vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                Vector<Real> realWork( lapack::SVDRealWorkSize(m,n) );
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 0 );
#endif
                lapack::SVD
                ( 'S', 'S', m, n,
                  D_.Buffer(), D_.LDim(), &sigma[0],
                  U.Buffer(), U.LDim(), VH.Buffer(), VH.LDim(),
                  &work[0], work.Size(), &realWork[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 0 );
#endif
                SVDTrunc( U, sigma, VH, relTol, twoNorm );

                if( inTargetTeam_ )
                {

                    SF.D.Resize( m, U.Width() );
                    SF.D.Init();
                    for( int j=0; j<U.Width(); j++ )
                        for( int i=0; i<m; i++)
                            SF.D.Set(i,j,U.Get(i,j)*sigma[j]);
                }
                else
                {

                    SF.D.Resize( n, VH.Height() );
                    SF.D.Init();
                    for( int j=0; j<maxRank; j++ )
                        for( int i=0; i<n; i++)
                            SF.D.Set(i,j,VH.Get(j,i));
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
            if( LH > totalrank && LW > totalrank && totalrank > MaxRank() )
            {
                LowRank<Scalar> &F = *block_.data.F;
                Dense<Scalar> &U = F.U;
                Dense<Scalar> Utmp;
                hmat_tools::Copy(U, Utmp);
                hmat_tools::Multiply( Scalar(1), Utmp, BL_, U );

                Dense<Scalar> &V = F.V;
                Dense<Scalar> Vtmp;
                hmat_tools::Copy(V, Vtmp);
                hmat_tools::Multiply( Scalar(1), Vtmp, BR_, V );
            }
            else
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
            hmat_tools::MultiplyTranspose( Scalar(1), F.U, F.V, Scalar(1), D_ );

            {
                // Perform an SVD on the dense matrix, overwriting it with
                // the left singular vectors and VH with the adjoint of the
                // right singular vecs
                Dense<Scalar> VH( std::min(m,n), n );
                Dense<Scalar> U( m, std::min(m,n) );
                Vector<Real> sigma( minDim );
                Vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                Vector<Real> realWork( lapack::SVDRealWorkSize(m,n) );
#ifdef TIME_MULTIPLY
                timerGlobal.Start( 0 );
#endif
                lapack::SVD
                ( 'S', 'S', m, n,
                  D_.Buffer(), D_.LDim(), &sigma[0],
                  U.Buffer(), U.LDim(), VH.Buffer(), VH.LDim(),
                  &work[0], work.Size(), &realWork[0] );
#ifdef TIME_MULTIPLY
                timerGlobal.Stop( 0 );
#endif
                SVDTrunc( U, sigma, VH, relTol, twoNorm );

                // Form U with the truncated left singular vectors scaled
                // by the corresponding singular values
                F.U.Resize( m, U.Width() );
                F.U.Init();
                for( int j=0; j<U.Width(); ++j )
                    for( int i=0; i<m; ++i )
                        F.U.Set(i,j,sigma[j]*U.Get(i,j));

                // Form V with the truncated right singular vectors
                F.V.Resize( n, VH.Height() );
                F.V.Init();
                for( int j=0; j<VH.Height(); ++j )
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

            hmat_tools::MultiplyTranspose( Scalar(1), U, V, Scalar(1), SD.D );

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

        hmat_tools::MultiplyTranspose( Scalar(1), U, V, Scalar(1), D );

        UMap_.Clear();
        VMap_.Clear();
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFCleanup()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFCleanup");
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
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFCleanup();
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
        USqr_.Clear();
        VSqr_.Clear();
        USqrEig_.Clear();
        VSqrEig_.Clear();
        BSqrU_.Clear();
        BSqrVH_.Clear();
        BSigma_.Clear();
        BL_.Clear();
        BR_.Clear();
        UMap_.Clear();
        VMap_.Clear();
        ZMap_.Clear();
        colXMap_.Clear();
        rowXMap_.Clear();
        D_.Clear();
        SFD_.Clear();
        mainContextMap_.Clear();
        colFHHContextMap_.Clear();
        rowFHHContextMap_.Clear();
        OmegaTU_.Clear();
        OmegaTV_.Clear();
        colTSqr_.Clear();
        rowTSqr_.Clear();
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
