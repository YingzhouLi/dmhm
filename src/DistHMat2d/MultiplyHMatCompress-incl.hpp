/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

#include "./Truncation-incl.hpp"

int EVD_Count;

namespace dmhm {

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompress( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompress");
#endif
    Real error = lapack::MachineEpsilon<Real>();
    //Written By Ryan Li
    // Compress low-rank F matrix into much lower form.
    // Our low-rank matrix is UV', we want to compute eigenvalues and 
    // eigenvectors of U'U and V'V.
    // U'U first stored in USqr_, then we use USqr_ to store the orthognal
    // vectors of U'U, and USqrEig_ to store the eigenvalues of U'U.
    // Everything about V are same in VSqr_ and VSqrEig_.
    
//    MultiplyHMatCompressFCompressless( startLevel, endLevel );
    mpi::Comm team = teams_->Team( level_ );
    const int teamRank = mpi::CommRank( team );
    MultiplyHMatCompressLowRankCountAndResize(0);
    MultiplyHMatCompressLowRankImport(0);
    MultiplyHMatCompressFPrecompute( startLevel, endLevel);
    MultiplyHMatCompressFReduces( startLevel, endLevel );

    EVD_Count=0;
    MultiplyHMatCompressFEigenDecomp( startLevel, endLevel );
    MultiplyHMatCompressFPassMatrix( startLevel, endLevel );
    MultiplyHMatCompressFPassVector( startLevel, endLevel );

    // Compute sigma_1 V1' V2 sigma_2, the middle part of UV'
    // We use B to state the mid part of UV' that is 
    // B = sigma_1 V1' V2 sigma_2.
    // BSqr_ = sqrt(USqrEig_) USqr_' VSqr_ sqrt(VSqrEig_)
    // Then BSqr_ also will be used to store the eigenvectors
    // of B. BSqrEig_ stores eigenvalues of B.
    MultiplyHMatCompressFMidcompute( error, startLevel, endLevel );
    MultiplyHMatCompressFPassbackNum( startLevel, endLevel );
    MultiplyHMatCompressFPassbackData( startLevel, endLevel );

    // Compute USqr*sqrt(USqrEig)^-1 BSqrU BSigma = BL
    // We overwrite the USqr = USqr*sqrt(USqrEig)^-1
    // Also overwrite the BSqrU = BSqrU BSigma
    // Compute VSqr*sqrt(VSqrEig)^-1 BSqrV = BR
    // We overwrite the VSqr = VSqr*sqrt(VSqrEig)^-1
    MultiplyHMatCompressFPostcompute( error, startLevel, endLevel );

//    Real zeroerror = (Real) 0.1;
//    MultiplyHMatCompressFEigenTrunc( zeroerror );
    MultiplyHMatCompressFBroadcastsNum( startLevel, endLevel );
    MultiplyHMatCompressFBroadcasts( startLevel, endLevel );
    // Compute the final U and V store in the usual space.
    MultiplyHMatCompressFFinalcompute( startLevel, endLevel );
    
    // Clean up all the space used in this file
    // Also, clean up the colXMap_, rowXMap_, UMap_, VMap_, ZMap_
    //MultiplyHMatCompressFCleanup( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressLowRankCountAndResize( int rank )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressLowRankCountAndResize");
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

        // Store the rank and create the space
        const unsigned teamLevel = teams_->TeamLevel( level_ );
        if( inTargetTeam_ )
        {
            const int oldRank = DF.ULocal.Width();
            const int localHeight = DF.ULocal.Height();

            Dense<Scalar> ULocalCopy;
            hmat_tools::Copy( DF.ULocal, ULocalCopy );
            DF.ULocal.Resize( localHeight, rank, localHeight );
            std::memcpy
            ( DF.ULocal.Buffer(0,rank-oldRank), ULocalCopy.LockedBuffer(), 
              localHeight*oldRank*sizeof(Scalar) );

        }
        if( inSourceTeam_ )
        {
            const int oldRank = DF.VLocal.Width();
            const int localWidth = DF.VLocal.Height();

            Dense<Scalar> VLocalCopy;
            hmat_tools::Copy( DF.VLocal, VLocalCopy );
            DF.VLocal.Resize( localWidth, rank, localWidth );
            std::memcpy
            ( DF.VLocal.Buffer(0,rank-oldRank), VLocalCopy.LockedBuffer(),
              localWidth*oldRank*sizeof(Scalar) );

        }
        DF.rank = rank;
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const unsigned teamLevel = teams_->TeamLevel( level_ );

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

        // Create the space and store the rank if we'll need to do a QR
        if( inTargetTeam_ )
        {
            const int oldRank = SF.D.Width();
            const int height = SF.D.Height();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( SF.D, UCopy );
            SF.D.Resize( height, rank, height );
            SF.rank = rank;
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              height*oldRank*sizeof(Scalar) );

        }
        else
        {
            const int oldRank = SF.D.Width();
            const int width = SF.D.Height();

            Dense<Scalar> VCopy;
            hmat_tools::Copy( SF.D, VCopy );
            SF.D.Resize( width, rank, width );
            SF.rank = rank;
            std::memcpy
            ( SF.D.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              width*oldRank*sizeof(Scalar) );

        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const unsigned teamLevel = teams_->TeamLevel( level_ );
        
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

        // Create the space and store the updates. If there are no dense 
        // updates, then mark two more matrices for QR factorization.
        {
            const int oldRank = F.Rank();
            const int height = F.Height();
            const int width = F.Width();

            Dense<Scalar> UCopy;
            hmat_tools::Copy( F.U, UCopy );
            F.U.Resize( height, rank, height );
            std::memcpy
            ( F.U.Buffer(0,rank-oldRank), UCopy.LockedBuffer(), 
              height*oldRank*sizeof(Scalar) );

            Dense<Scalar> VCopy;
            hmat_tools::Copy( F.V, VCopy );
            F.V.Resize( width, rank, width );
            std::memcpy
            ( F.V.Buffer(0,rank-oldRank), VCopy.LockedBuffer(),
              width*oldRank*sizeof(Scalar) );

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
                UMap_.Set( 0, new Dense<Scalar>( m, rank ) );
            else
            {
                UMap_.ResetIterator();
                Dense<Scalar>& firstU = *UMap_.CurrentEntry();
                UMap_.Increment();

                Dense<Scalar> firstUCopy;
                hmat_tools::Copy( firstU, firstUCopy );

                firstU.Resize( m, rank, m );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstUCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), 
                          firstUCopy.LockedBuffer(0,j), m*sizeof(Scalar) );
                }
                // Push the rest of the updates in and then erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& U = *UMap_.CurrentEntry();
                    const int r = U.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstU.Buffer(0,rOffset+j), U.LockedBuffer(0,j),
                          m*sizeof(Scalar) );
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
                VMap_.Set( 0, new Dense<Scalar>( n, rank ) );
            else
            {
                VMap_.ResetIterator();
                Dense<Scalar>& firstV = *VMap_.CurrentEntry();
                VMap_.Increment();

                Dense<Scalar> firstVCopy;
                hmat_tools::Copy( firstV, firstVCopy );

                firstV.Resize( n, rank, n );
                // Push the original first update into the back
                int rOffset = rank;
                {
                    const int r = firstVCopy.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), 
                          firstVCopy.LockedBuffer(0,j), n*sizeof(Scalar) );
                }
                // Push the rest of the updates in and erase them
                for( int update=1; update<numLowRankUpdates; ++update )
                {
                    Dense<Scalar>& V = *VMap_.CurrentEntry();
                    const int r = V.Width();
                    rOffset -= r;
                    for( int j=0; j<r; ++j )
                        std::memcpy
                        ( firstV.Buffer(0,rOffset+j), V.LockedBuffer(0,j),
                          n*sizeof(Scalar) );
                    VMap_.EraseCurrentEntry();
                }
            }
        }
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
            const int r = U.Width();
            const char option = 'T';
            blas::Gemm
            ( 'N', option, m, n, r,
              Scalar(1), U.LockedBuffer(), U.LDim(),
                         V.LockedBuffer(), V.LDim(),
              Scalar(1), D.Buffer(),       D.LDim() );
            UMap_.EraseCurrentEntry();
            VMap_.EraseCurrentEntry();
        }

        // Create space for storing the parent updates
        UMap_.Set( 0, new Dense<Scalar>(m,rank) );
        VMap_.Set( 0, new Dense<Scalar>(n,rank) );
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
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        int newRank = rank;
        if( teamSize == 2 )
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
        else // teamSize >= 4
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
                    std::memcpy
                    ( mainU->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
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
                    std::memcpy
                    ( mainU->Buffer(0,newRank+j), U.LockedBuffer(0,j),
                      m*sizeof(Scalar) );
                UMap_.EraseCurrentEntry();
                newRank += r;
            }
        }
        else
            UMap_.Clear();

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
                    std::memcpy
                    ( mainV->Buffer(0,newRank+j), X.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
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
                    std::memcpy
                    ( mainV->Buffer(0,newRank+j), V.LockedBuffer(0,j),
                      n*sizeof(Scalar) );
                VMap_.EraseCurrentEntry();
                newRank += r;
            }
        }
        else
            VMap_.Clear();
        break;
    }
    case SPLIT_DENSE:
    case DENSE:
        break;
    default:
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
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamSize == 2 )
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
        else  // teamSize >= 4
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
            std::memcpy
            ( DF.ULocal.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( SF.D.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const int m = U.Height();
        const int r = U.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( F.U.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
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
            std::memcpy
            ( mainU.Buffer(0,rank+j), U.LockedBuffer(0,j),
              m*sizeof(Scalar) );
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
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        if( teamSize == 2 )
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
        else  // teamSize >= 4
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
            std::memcpy
            ( DF.VLocal.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( SF.D.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const int n = V.Height();
        const int r = V.Width();
        for( int j=0; j<r; ++j )
            std::memcpy
            ( F.V.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
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
            std::memcpy
            ( mainV.Buffer(0,rank+j), V.LockedBuffer(0,j),
              n*sizeof(Scalar) );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPrecompute
( int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatCompressFPrecompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        DistLowRank &DF = *block_.data.DF;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = 'T';
        int totalrank=colXMap_.TotalWidth() + UMap_.TotalWidth() + DF.ULocal.Width();
        
        if( inTargetTeam_ && totalrank > 0 && LH > 0 )
        {
            USqr_.Resize( totalrank, totalrank, totalrank );
            USqrEig_.resize( totalrank );
            Utmp_.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( Utmp_.Buffer(0,offset), DF.ULocal.LockedBuffer(),
              LH*DF.ULocal.Width()*sizeof(Scalar) );
            offset += DF.ULocal.Width();

            int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
            {
                Dense<Scalar>& U = *UMap_.CurrentEntry();
                std::memcpy
                ( Utmp_.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                std::memcpy
                ( Utmp_.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }

            blas::Gemm
            ( 'C', 'N', totalrank, totalrank, LH,
             Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                        Utmp_.LockedBuffer(), Utmp_.LDim(),
             Scalar(0), USqr_.Buffer(),       USqr_.LDim() );
        }

        totalrank=rowXMap_.TotalWidth() + VMap_.TotalWidth() + DF.VLocal.Width();
        offset = 0;
        if( inSourceTeam_ && totalrank > 0 && LW > 0 )
        {
            VSqr_.Resize( totalrank, totalrank, totalrank );
            VSqrEig_.resize( totalrank );
            Vtmp_.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( Vtmp_.Buffer(0,offset), DF.VLocal.LockedBuffer(),
              LW*DF.VLocal.Width()*sizeof(Scalar) );
            offset += DF.VLocal.Width();

            int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                std::memcpy
                ( Vtmp_.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& V = *rowXMap_.CurrentEntry();
                std::memcpy
                ( Vtmp_.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }

            blas::Gemm
            ( 'C', 'N', totalrank, totalrank, LW,
             Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                        Vtmp_.LockedBuffer(), Vtmp_.LDim(),
             Scalar(0), VSqr_.Buffer(),       VSqr_.LDim() );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( haveDenseUpdate_ )
            break;
        SplitLowRank &SF = *block_.data.SF;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = 'T';
        int totalrank = colXMap_.TotalWidth() + 
                        UMap_.TotalWidth() + SF.D.Width();
        
        if( inTargetTeam_ && totalrank > 0 && LH > 0 )
        {
            USqr_.Resize( totalrank, totalrank, totalrank );
            USqrEig_.resize( totalrank );
            Utmp_.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( Utmp_.Buffer(0,offset), SF.D.LockedBuffer(),
              LH*SF.D.Width()*sizeof(Scalar) );
            offset += SF.D.Width();

            int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
            {
                Dense<Scalar>& U = *UMap_.CurrentEntry();
                std::memcpy
                ( Utmp_.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                std::memcpy
                ( Utmp_.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LH,
             Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                        Utmp_.LockedBuffer(), Utmp_.LDim(),
             Scalar(0), USqr_.Buffer(),       USqr_.LDim() );
        }

        totalrank=rowXMap_.TotalWidth() + VMap_.TotalWidth() + SF.D.Width();
        offset = 0;
        if( inSourceTeam_ && totalrank > 0 && LW > 0 )
        {
            VSqr_.Resize( totalrank, totalrank, totalrank );
            VSqrEig_.resize( totalrank );
            Vtmp_.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( Vtmp_.Buffer(0,offset), SF.D.LockedBuffer(),
              LW*SF.D.Width()*sizeof(Scalar) );
            offset += SF.D.Width();

            int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                std::memcpy
                ( Vtmp_.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& V = *rowXMap_.CurrentEntry();
                std::memcpy
                ( Vtmp_.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LW,
             Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                        Vtmp_.LockedBuffer(), Vtmp_.LDim(),
             Scalar(0), VSqr_.Buffer(),       VSqr_.LDim() );
        }
        break;
    }
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( haveDenseUpdate_ )
            break;
        LowRank<Scalar> &F = *block_.data.F;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = 'T';
        int totalrank = F.U.Width();
        
        if( totalrank > MaxRank() && LH > 0 )
        {
            USqr_.Resize( totalrank, totalrank, totalrank );
            USqrEig_.resize( totalrank );
            Utmp_.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( Utmp_.Buffer(0,offset), F.U.LockedBuffer(),
              LH*F.U.Width()*sizeof(Scalar) );
            offset=F.U.Width();
//Print
//_Utmp.Print("_Utmp");
            blas::Gemm
            ( 'C', 'N', totalrank, totalrank, LH,
             Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                        Utmp_.LockedBuffer(), Utmp_.LDim(),
             Scalar(0), USqr_.Buffer(),       USqr_.LDim() );
        }

        totalrank = F.V.Width();
        offset = 0;
        if( totalrank > MaxRank() && LW > 0 )
        {
            VSqr_.Resize( totalrank, totalrank, totalrank );
            VSqrEig_.resize( totalrank );
            Vtmp_.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( Vtmp_.Buffer(0,offset), F.V.LockedBuffer(),
              LW*F.V.Width()*sizeof(Scalar) );
            offset=F.V.Width();

//Print
//_Vtmp.Print("_Vtmp");
            blas::Gemm
            ( 'C', 'N', totalrank, totalrank, LW,
             Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                        Vtmp_.LockedBuffer(), Vtmp_.LDim(),
             Scalar(0), VSqr_.Buffer(),       VSqr_.LDim() );
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
DistHMat2d<Scalar>::MultiplyHMatCompressFReduces
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFReduces");
#endif

    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyHMatCompressFReducesCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numReduces; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFReducesPack
    ( buffer, offsetscopy, startLevel, endLevel );
    
    MultiplyHMatCompressFTreeReduces( buffer, sizes );
    
    MultiplyHMatCompressFReducesUnpack
    ( buffer, offsets, startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReducesCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        if( level_ >= startLevel && level_ < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatCompressFReducesCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        sizes[level_] += USqr_.Height()*USqr_.Width();
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
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatCompressFReducesPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int Size=USqr_.Height()*USqr_.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[level_]], USqr_.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[level_] += Size;
        }

        Size=VSqr_.Height()*VSqr_.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[level_]], VSqr_.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[level_] += Size;
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
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeReduces");
#endif
    teams_-> TreeSumToRoots( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFReducesUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatCompressFReducesUnpack
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
            int Size=USqr_.Height()*USqr_.Width();                
            if( Size > 0 )
            {
                std::memcpy
                ( USqr_.Buffer(), &buffer[offsets[level_]],
                  Size*sizeof(Scalar) );
                offsets[level_] += Size;
            }

            Size=VSqr_.Height()*VSqr_.Width();
            if( Size > 0 )
            {
                std::memcpy
                ( VSqr_.Buffer(), &buffer[offsets[level_]],
                  Size*sizeof(Scalar) );
                offsets[level_] += Size;
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
DistHMat2d<Scalar>::MultiplyHMatCompressFEigenDecomp
( int startLevel, int endLevel )
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
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            // Calculate Eigenvalues of Squared Matrix               
            int Sizemax = std::max(USqr_.Height(), VSqr_.Height());
             
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( Sizemax );
            lrwork=lapack::EVDRealWorkSize( Sizemax );
            liwork=lapack::EVDIntWorkSize( Sizemax );
                    
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_VSqr.Print("_VSqr Before svd");*/
                                                                     
            if( !USqr_.IsEmpty() )
            {
//Print
//_USqr.Print("USqr_Before_EVD");
                lapack::EVD
                ('V', 'U', USqr_.Height(), 
                           USqr_.Buffer(), USqr_.LDim(),
                           &USqrEig_[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );
//Print
//std::cout << "Size: " << USqr_.Height() << std::endl;
EVD_Count++;
            }
                                                                     
            if( !VSqr_.IsEmpty() )
            {
                lapack::EVD
                ('V', 'U', VSqr_.Height(), 
                           VSqr_.Buffer(), VSqr_.LDim(),
                           &VSqrEig_[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );
//Print
//std::cout << "Size: " << VSqr_.Height() << std::endl;
EVD_Count++;
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrix
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrix");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatCompressFPassMatrixCount
    ( sendSizes, recvSizes, startLevel, endLevel );

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
    MultiplyHMatCompressFPassMatrixPack
    ( sendBuffer, offsets, startLevel, endLevel );

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
    MultiplyHMatCompressFPassMatrixUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::MultiplyHMatCompressFPassMatrixCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassMatrixCount
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inSourceTeam_ )
                AddToMap( sendSizes, targetRoot_, VSqr_.Height()*VSqr_.Width() );
            else
                AddToMap( recvSizes, sourceRoot_, USqr_.Height()*USqr_.Width() );
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
( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrixPack");
#endif
    if( !inSourceTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassMatrixPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && VSqr_.Height() > 0 )
        {
            std::memcpy
            ( &buffer[offsets[targetRoot_]], VSqr_.LockedBuffer(),
              VSqr_.Height()*VSqr_.Width()*sizeof(Scalar) );
            offsets[targetRoot_] += VSqr_.Height()*VSqr_.Width();
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
( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassMatrixUnpack");
#endif
    if( !inTargetTeam_ || Height() == 0 || Width() == 0 )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassMatrixUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( haveDenseUpdate_ )
            break;
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && USqr_.Height() > 0 )
        {
            VSqr_.Resize( USqr_.Height(), USqr_.Width(), USqr_.LDim() );
            std::memcpy
            ( VSqr_.Buffer(), &buffer[offsets[sourceRoot_]],
              VSqr_.Height()*VSqr_.Width()*sizeof(Scalar) );
            offsets[sourceRoot_] += VSqr_.Height()*VSqr_.Width();

        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVector
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassVector");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatCompressFPassVectorCount
    ( sendSizes, recvSizes, startLevel, endLevel );

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
    std::vector<Real> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    MultiplyHMatCompressFPassVectorPack
    ( sendBuffer, offsets, startLevel, endLevel );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<Real> recvBuffer( totalRecvSize );
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
    MultiplyHMatCompressFPassVectorUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::MultiplyHMatCompressFPassVectorCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassVectorCount
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inSourceTeam_ )
                AddToMap( sendSizes, targetRoot_, VSqrEig_.size() );
            else
                AddToMap( recvSizes, sourceRoot_, USqrEig_.size() );
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
( std::vector<Real>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassVectorPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && VSqrEig_.size() > 0 )
        {
            std::memcpy
            ( &buffer[offsets[targetRoot_]], &VSqrEig_[0],
              VSqrEig_.size()*sizeof(Real) );
            offsets[targetRoot_] += VSqrEig_.size();
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
( const std::vector<Real>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassVectorUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && USqrEig_.size() > 0 )
        {
            VSqrEig_.resize( USqrEig_.size() );

            std::memcpy
            ( &VSqrEig_[0], &buffer[offsets[sourceRoot_]],
              VSqrEig_.size()*sizeof(Real) );
            offsets[sourceRoot_] += VSqrEig_.size();
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
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFMidcompute");
#endif
    if( !inTargetTeam_ || Height() == 0 || Width() == 0 )
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
                    node.Child(t,s).MultiplyHMatCompressFMidcompute
                    ( error, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
//Print
//std::cout << USqr_.Height() << " " << USqr_.Width() << " " << USqr_.IsEmpty() << std::endl;
        if( teamRank == 0 && !USqr_.IsEmpty() && inTargetTeam_ )
        {
//Print
//_USqr.Print("USqr");
//_VSqr.Print("VSqr");
            if( USqr_.Height() != VSqr_.Height() ||
                USqrEig_.size() != VSqrEig_.size() )
            {
#ifndef RELEASE
                throw std::logic_error("Dimension error during calculation");
#endif
            }
            const char option = 'T';

          //  EVDTrunc(USqr_, USqrEig_, error);
          //  EVDTrunc(VSqr_, VSqrEig_, error);

            BSqr_.Resize(USqr_.Width(), VSqr_.Width(), USqr_.Width());

            blas::Gemm
            ( option, 'N', USqr_.Width(), VSqr_.Width(), USqr_.LDim(),
              Scalar(1), USqr_.LockedBuffer(), USqr_.LDim(),
                         VSqr_.LockedBuffer(), VSqr_.LDim(),
              Scalar(0), BSqr_.Buffer(),       BSqr_.LDim() );

//Print
//_BSqr.Print("BSqr");
            hmat_tools::Conjugate(BSqr_);
//Print
//_BSqr.Print("BSqr");
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_VSqr.Print("_VSqr Before svd");*/
//Print
//_VSqr.Print("_VSqr");
//std::cout << "_VSqrEig" << std::endl;
//for( int i=0; i<VSqrEig_.size(); ++i)
//    std::cout << VSqrEig_[i] << "  ";
//std::cout << std::endl;
            std::vector<Real> USqrSigma(USqrEig_.size());
            std::vector<Real> VSqrSigma(VSqrEig_.size());
            for( int i=0; i<USqrEig_.size(); ++i)
                if( USqrEig_[i] > (Real)0 )
                    USqrSigma[i] = sqrt(USqrEig_[i]);
                else
                    USqrSigma[i] = (Real)0;
            for( int i=0; i<VSqrEig_.size(); ++i)
                if( VSqrEig_[i] > (Real)0 )
                    VSqrSigma[i] = sqrt(VSqrEig_[i]);
                else
                    VSqrSigma[i] = (Real)0;

            for( int j=0; j<BSqr_.Width(); ++j)
                for( int i=0; i<BSqr_.Height(); ++i)
                    BSqr_.Set(i,j, BSqr_.Get(i,j)*USqrSigma[i]*VSqrSigma[j]);
//Print
//_BSqr.Print("BSqr");

            int m = BSqr_.Height();
            int n = BSqr_.Width();
            int k = std::min(m, n);

            BSqrU_.Resize(m,k,m);
            BSqrVH_.Resize(k,n,k);
            BSigma_.resize(k);

            int lwork = lapack::SVDWorkSize(m,n);
            int lrwork = lapack::SVDRealWorkSize(m,n);

            std::vector<Scalar> work(lwork);
            std::vector<Real> rwork(lrwork);
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_BSqr.Print("_BSqr Before svd");*/
            lapack::SVD
            ('S', 'S' ,m ,n, 
             BSqr_.Buffer(), BSqr_.LDim(), &BSigma_[0],
             BSqrU_.Buffer(), BSqrU_.LDim(),
             BSqrVH_.Buffer(), BSqrVH_.LDim(),
             &work[0], lwork, &rwork[0] );
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_BSqrVH.Print("_BSqrVH After svd");*/

            SVDTrunc(BSqrU_, BSigma_, BSqrVH_, error);
//Print
//_BSqrU.Print("BSqrU");
//Print
//_BSqrVH.Print("BSqrVH");
//Print
//mpi::Comm teamp = teams_->Team( 0 );
//const int teamRankp = mpi::CommRank( teamp );
//if( level_ == 3 && teamRankp == 0 && block_.type==LOW_RANK && Vtmp_.Height() == 2 && Utmp_.Height() == 2)
//std::cout << "Height: " << Utmp_.Height() << " Width: " << Vtmp_.Height() << " B(1,1): " << WrapScalar(BSqrVH_.Get(10,10)) << std::endl;
//_BSqrVH.Print("_SqrVH_Mid");

        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackNum");
#endif

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatCompressFPassbackNumCount
    ( sendSizes, recvSizes, startLevel, endLevel );

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
    std::vector<int> sendBuffer(totalSendSize);
    std::map<int,int> offsets = sendOffsets;
    MultiplyHMatCompressFPassbackNumPack
    ( sendBuffer, offsets, startLevel, endLevel );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<int> recvBuffer( totalRecvSize );
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
    MultiplyHMatCompressFPassbackNumUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNumCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackNumCount
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( inTargetTeam_ )
                AddToMap( sendSizes, sourceRoot_, 1 );
            else
                AddToMap( recvSizes, targetRoot_, 1 );
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
( std::vector<int>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackNumPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ )
            break;
        if( haveDenseUpdate_ )
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
    default:
        break;
    }
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackNumUnpack
( const std::vector<int>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackNumUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            BSqrVH_.Resize
            ( buffer[offsets[targetRoot_]], VSqr_.Height(), 
              buffer[offsets[targetRoot_]] );
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
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackData
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFPassbackData");
#endif

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatCompressFPassbackDataCount
    ( sendSizes, recvSizes, startLevel, endLevel );

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
    MultiplyHMatCompressFPassbackDataPack
    ( sendBuffer, offsets, startLevel, endLevel );

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
    MultiplyHMatCompressFPassbackDataUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void 
DistHMat2d<Scalar>::MultiplyHMatCompressFPassbackDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackDataCount
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( !haveDenseUpdate_ )
        {
            if( inSourceTeam_ && inTargetTeam_ )
                break; 
            mpi::Comm team = teams_->Team( level_ );
            const int teamRank = mpi::CommRank( team );                                   
            if( teamRank ==0 )
            {
                if( inTargetTeam_ )
                    AddToMap( sendSizes, sourceRoot_, BSqrVH_.Height()*BSqrVH_.Width() );
                else
                    AddToMap( recvSizes, targetRoot_, BSqrVH_.Height()*BSqrVH_.Width() );
            }
        }
        else if( block_.type == SPLIT_LOW_RANK )
        {
            const SplitLowRank& SF = *block_.data.SF;
            if( inTargetTeam_ )
            {
                AddToMap( sendSizes, sourceRoot_, Height()*SF.rank );
                AddToMap( recvSizes, sourceRoot_, Width()*SF.rank+Width()*Height() );
            }
            else
            {
                AddToMap( sendSizes, targetRoot_, Width()*SF.rank+Width()*Height() );
                AddToMap( recvSizes, targetRoot_, Height()*SF.rank );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )
        {
            UMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            AddToMap( sendSizes, sourceRoot_, Height()*U.Width() );
        }
        else
        {
            VMap_.ResetIterator();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();
            AddToMap( recvSizes, targetRoot_, Height()*V.Width() );
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
( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackDataPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( !haveDenseUpdate_ )
        {
            if( inSourceTeam_ || !inTargetTeam_ )
                break;
            mpi::Comm team = teams_->Team( level_ );                      
            const int teamRank = mpi::CommRank( team );
            if( teamRank ==0 )
            {
                int size=BSqrVH_.Height()*BSqrVH_.Width();
                std::memcpy
                ( &buffer[offsets[sourceRoot_]], BSqrVH_.LockedBuffer(),
                  size*sizeof(Scalar) );
                offsets[sourceRoot_] += size;
            }
        }
        else if( block_.type == SPLIT_LOW_RANK )
        {
            const SplitLowRank& SF = *block_.data.SF;
            const int m = Height();
            const int n = Width();
            if( inTargetTeam_ )
            {
                int size = m*SF.rank;
                std::memcpy
                ( &buffer[offsets[sourceRoot_]], SF.D.LockedBuffer(),
                  size*sizeof(Scalar) );
                offsets[sourceRoot_] += size;
            
            }
            else
            {
                int size = n*SF.rank;
                std::memcpy
                ( &buffer[offsets[targetRoot_]], SF.D.LockedBuffer(),
                  size*sizeof(Scalar) );
                offsets[targetRoot_] += size;
                
                size = n*m;
                std::memcpy
                ( &buffer[offsets[targetRoot_]], D_.LockedBuffer(),
                  size*sizeof(Scalar) );
                offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )
        {
            UMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            if( Height() != U.Height() )
                throw std::logic_error("Packing SPLIT_DENSE, the height does not fit");
            int size=U.Height()*U.Width();
            std::memcpy
            ( &buffer[offsets[sourceRoot_]], U.LockedBuffer(),
              size*sizeof(Scalar) );
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
( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassbackDataUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( !haveDenseUpdate_ )
        {
            if( inTargetTeam_ || !inSourceTeam_ )
                break;
            mpi::Comm team = teams_->Team( level_ );                
            const int teamRank = mpi::CommRank( team );
            if( teamRank ==0 )
            {
                int size=BSqrVH_.Height()*BSqrVH_.Width();
                std::memcpy
                ( BSqrVH_.Buffer(), &buffer[offsets[targetRoot_]],
                  size*sizeof(Scalar) );
                offsets[targetRoot_] += size;
            }
        }
        else if( block_.type == SPLIT_LOW_RANK )
        {
            const SplitLowRank& SF = *block_.data.SF;
            const int m = Height();
            const int n = Width();
            if( inTargetTeam_ )
            {
                int size = n*SF.rank;
                SFD_.Resize(n, SF.rank, n);
                std::memcpy
                ( SFD_.Buffer(), &buffer[offsets[sourceRoot_]],
                  size*sizeof(Scalar) );
                offsets[sourceRoot_] += size;
            
                size = n*m;
                D_.Resize(m, n, m);
                std::memcpy
                ( D_.Buffer(), &buffer[offsets[sourceRoot_]],
                  size*sizeof(Scalar) );
                offsets[sourceRoot_] += size;
            }
            else
            {
                int size = m*SF.rank;
                SFD_.Resize(m, SF.rank, m);
                std::memcpy
                ( SFD_.Buffer(), &buffer[offsets[targetRoot_]],
                  size*sizeof(Scalar) );
                offsets[targetRoot_] += size;
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        if( level_ < startLevel )
            break;
        if( inSourceTeam_ )
        {
            UMap_.Set( 0, new Dense<Scalar>);
            Dense<Scalar>& U = UMap_.Get(0);
            VMap_.ResetIterator();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();
            U.Resize(Height(), V.Width(), Height());
            int size=U.Height()*U.Width();
            std::memcpy
            ( U.Buffer(), &buffer[offsets[targetRoot_]], 
              size*sizeof(Scalar) );
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
( Real error, int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatCompressFPostcompute
                    ( error, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( haveDenseUpdate_ )
            break;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
// RYAN: Please do not check in code like this...
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
int print;
if(teamRankp==1)
print=0;
else
print=0;
        if( teamRank == 0 )
        {
            const char option = 'T';
//print
if(print)
std::cout << "Run until here 1" << std::endl;
            if( inTargetTeam_ && USqrEig_.size() > 0)
            {
//Print
if(print)
std::cout << USqrEig_.size() << " " << VSqrEig_.size() << " " << block_.type << std::endl;
                Real Eigmax;
                if( USqrEig_[USqrEig_.size()-1] > (Real)0 )
                    Eigmax=sqrt( USqrEig_[USqrEig_.size()-1] );
                else
                    Eigmax=0;
//Print
//if(level_==3 && block_.type==LOW_RANK)
//for(int i=0; i<USqrEig_.size(); i++)
//    std::cout << USqrEig_[i] << " ";
//std::cout << std::endl;
                for(int j=0; j<USqr_.Width(); ++j)
                    if(sqrt(std::abs(USqrEig_[j])) > error*error )
                    {
                        Real sqrteig=sqrt(std::abs(USqrEig_[j]));
                        for(int i=0; i<USqr_.Height(); ++i)
                            USqr_.Set(i,j,USqr_.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<USqr_.Height(); ++i)
                            USqr_.Set(i,j, Scalar(0));
                    }

//print
if(print)
std::cout << "Run until here 1.5" << std::endl;
                for(int j=0; j<BSqrU_.Width(); ++j)
                    for(int i=0; i<BSqrU_.Height(); ++i)
                        BSqrU_.Set(i,j,BSqrU_.Get(i,j)*BSigma_[j]);

                BL_.Resize(USqr_.Height(), BSqrU_.Width(), USqr_.Height());

//print
//_BSqrU.Print("BU*BS");
//_USqr.Print("USqr/Ueig");
if(print)
std::cout << "Run until here 1.8" << std::endl;
                blas::Gemm
                ( 'N', 'N', USqr_.Height(), BSqrU_.Width(), USqr_.Width(), 
                  Scalar(1), USqr_.LockedBuffer(),  USqr_.LDim(),
                             BSqrU_.LockedBuffer(), BSqrU_.LDim(),
                  Scalar(0), BL_.Buffer(), BL_.LDim() );
/*
                BL_.Resize(USqr_.Height(), BSqrU_.Width(), USqr_.Height());
                blas::Gemm
                ( 'N', option, USqr_.Height(), BSqrU_.Width(), USqr_.Width(), 
                  Scalar(1), USqr_.LockedBuffer(),  USqr_.LDim(),
                             USqr_.LockedBuffer(), USqr_.LDim(),
                  Scalar(0), BL_.Buffer(), BL_.LDim() );
            */
            }

//print
if(print)
std::cout << "Run until here 2, " << BSqrVH_.LDim() << " " << BSqrVH_.Height() << " " << BSqrVH_.Width() << std::endl;
            if(inSourceTeam_ && VSqrEig_.size() > 0)
            {
/*//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type == LOW_RANK)
_BSqrVH.Print("_BSqrVH_Post");*/
                Real Eigmax;
                if( VSqrEig_[VSqrEig_.size()-1] > (Real)0 )
                    Eigmax=sqrt(VSqrEig_[VSqrEig_.size()-1]);
                else
                    Eigmax=0;
                for(int j=0; j<VSqr_.Width(); ++j)
                    if(sqrt(std::abs(VSqrEig_[j])) > error*error )
                    {
                        Real sqrteig=sqrt(std::abs(VSqrEig_[j]));
                        for(int i=0; i<VSqr_.Height(); ++i)
                            VSqr_.Set(i,j,VSqr_.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<VSqr_.Height(); ++i)
                            VSqr_.Set(i,j, Scalar(0));
                    }

/*mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type==LOW_RANK && Vtmp_.Height() == 2 && Utmp_.Height() == 2)
{
    //_BL.Print("_BL");
    //_BR.Print("_BR");
    Dense<Scalar> BLRT;
    BLRT.Resize(VSqr_.Height(), VSqr_.Height(), VSqr_.Height());
    blas::Gemm
    ( option, 'N', VSqr_.Height(), VSqr_.Height(), VSqr_.Width(),
      Scalar(1), VSqr_.LockedBuffer(), VSqr_.LDim(),
                 VSqr_.LockedBuffer(), VSqr_.LDim(),
      Scalar(0), BLRT.Buffer(), BLRT.LDim());
    BLRT.Print("BLRT********************************************************");
    VSqr_.Print("_VSqr*****************************************");
}*/

                BR_.Resize(VSqr_.Height(), BSqrVH_.Height(), VSqr_.Height());

                blas::Gemm
                ( 'N', option, VSqr_.Height(), BSqrVH_.Height(), VSqr_.Width(),
                  Scalar(1), VSqr_.LockedBuffer(),  VSqr_.LDim(),
                             BSqrVH_.LockedBuffer(), BSqrVH_.LDim(),
                  Scalar(0), BR_.Buffer(), BR_.LDim() );
/*
                BR_.Resize(VSqr_.Height(), BSqrVH_.Height(), VSqr_.Height());
                blas::Gemm
                ( 'N', option, VSqr_.Height(), BSqrVH_.Height(), VSqr_.Width(),
                  Scalar(1), VSqr_.LockedBuffer(),  VSqr_.LDim(),
                             VSqr_.LockedBuffer(), VSqr_.LDim(),
                  Scalar(0), BR_.Buffer(), BR_.LDim() );
//                BR_.Print("_BR");*/
//Print
mpi::Comm teamp = teams_->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( level_ == 3 && teamRankp == 0 && block_.type==LOW_RANK )
{
    std::ofstream myfile;
    myfile.open("Check.txt", std::ios::app);
    //_BL.Print(myfile,"_BL");
    //_BR.Print(myfile,"_BR");
    Dense<Scalar> BLRT;
    BLRT.Resize(BL_.Height(), BR_.Height(), BL_.Height());
    blas::Gemm
    ( 'N', option, BL_.Height(), BR_.Height(), BL_.Width(),
      Scalar(1), BL_.LockedBuffer(), BL_.LDim(),
                 BR_.LockedBuffer(), BR_.LDim(),
      Scalar(0), BLRT.Buffer(), BLRT.LDim());
    BLRT.Print(myfile,"BLRT********************************************************");
    myfile.close();
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcastsNum");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyHMatCompressFBroadcastsNumCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalSize += sizes[i];
    std::vector<int> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFBroadcastsNumPack
    ( buffer, offsetscopy, startLevel, endLevel );

    MultiplyHMatCompressFTreeBroadcastsNum( buffer, sizes );
    
    MultiplyHMatCompressFBroadcastsNumUnpack
    ( buffer, offsets, startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNumCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
//Print
//if( level_ <2)
//_BL.Print("_BL");
    {
        if( level_ < startLevel )
            break;
        if( Utmp_.Height()>0 )
            sizes[level_]++;
        if( Vtmp_.Height()>0 )
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
( std::vector<int>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumPack
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
            if( BL_.Height()>0 )
            {
                buffer[offsets[level_]] = BL_.Width();
                offsets[level_]++;
            }
            if( BR_.Height()>0 )
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
( std::vector<int>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeBroadcastsNum");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsNumUnpack
( std::vector<int>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumUnpack
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
        if( teamRank != 0 )
        {
            if( Utmp_.Height()>0 )
            {
                BL_.Resize
                (Utmp_.Width(), buffer[offsets[level_]],
                 Utmp_.Width());
                offsets[level_]++;
            }
            if( Vtmp_.Height()>0 )
            {
                BR_.Resize
                (Vtmp_.Width(), buffer[offsets[level_]],
                 Vtmp_.Width());
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcasts
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFBroadcasts");
#endif
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyHMatCompressFBroadcastsCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatCompressFBroadcastsPack
    ( buffer, offsetscopy, startLevel, endLevel );

    MultiplyHMatCompressFTreeBroadcasts( buffer, sizes );
    
    MultiplyHMatCompressFBroadcastsUnpack
    ( buffer, offsets, startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsCount
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
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsPack
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
                int Size = BL_.Height()*BL_.Width();
                std::memcpy
                ( &buffer[offsets[level_]], BL_.LockedBuffer(),
                  Size*sizeof(Scalar) );
                offsets[level_] += Size;
            }
            if( BR_.Height() > 0 )
            {
                int Size = BR_.Height()*BR_.Width();
                std::memcpy
                ( &buffer[offsets[level_]], BR_.LockedBuffer(),
                  Size*sizeof(Scalar) );
                offsets[level_] += Size;
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
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFTreeBroadcasts");
#endif
    teams_-> TreeBroadcasts( buffer, sizes );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFBroadcastsUnpack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
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
            int Size = BL_.Height()*BL_.Width();
            std::memcpy
            ( BL_.Buffer(), &buffer[offsets[level_]],
              Size*sizeof(Scalar) );
            offsets[level_] += Size;
        }
        if( BR_.Height() > 0 )
        {                                                      
            int Size = BR_.Height()*BR_.Width();
            std::memcpy
            ( BR_.Buffer(), &buffer[offsets[level_]],
              Size*sizeof(Scalar) );
            offsets[level_] += Size;
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
( int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFFinalcompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( inTargetTeam_ )                                  
        { 
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &U = DF.ULocal;
            DF.rank = BL_.Width();
            U.Resize(Utmp_.Height(), BL_.Width(), Utmp_.Height());
            blas::Gemm
            ('N', 'N', Utmp_.Height(), BL_.Width(), Utmp_.Width(), 
             Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                        BL_.LockedBuffer(), BL_.LDim(),
             Scalar(0), U.Buffer(),         U.LDim() );
        }
        if( inSourceTeam_ )
        {                                                      
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &V = DF.VLocal;
            DF.rank = BR_.Width();
            V.Resize(Vtmp_.Height(), BR_.Width(), Vtmp_.Height());
            
            blas::Gemm
            ('N', 'N', Vtmp_.Height(), BR_.Width(), Vtmp_.Width(),
             Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                        BR_.LockedBuffer(), BR_.LDim(),
             Scalar(0), V.Buffer(),         V.LDim() );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        if( !haveDenseUpdate_ )
        {
            if( inTargetTeam_ )                                        
            { 
                SplitLowRank &SF = *block_.data.SF;
                Dense<Scalar> &U = SF.D;
                SF.rank = BL_.Width();
                U.Resize(Utmp_.Height(), BL_.Width(), Utmp_.Height());
                
                blas::Gemm
                ('N', 'N', Utmp_.Height(), BL_.Width(), Utmp_.Width(),
                 Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                            BL_.LockedBuffer(), BL_.LDim(),
                 Scalar(0), U.Buffer(),         U.LDim() );
            }
            if( inSourceTeam_ )
            {                                                      
                SplitLowRank &SF = *block_.data.SF;
                Dense<Scalar> &V = SF.D;
                SF.rank = BR_.Width();
                V.Resize(Vtmp_.Height(), BR_.Width(), Vtmp_.Height());
                
                blas::Gemm
                ('N', 'N', Vtmp_.Height(), BR_.Width(), Vtmp_.Width(),
                 Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                            BR_.LockedBuffer(), BR_.LDim(),
                 Scalar(0), V.Buffer(),         V.LDim() );
            }
        }
        else
        {
            SplitLowRank &SF = *block_.data.SF;
            const int m = Height();
            const int n = Width();
            const char option = 'T';
            if( inTargetTeam_ )
            {
                Dense<Scalar>& SFU = SF.D;
                Dense<Scalar>& SFV = SFD_;
                blas::Gemm                                   
                ('N', option, m, n, SF.rank,
                 Scalar(1), SFU.LockedBuffer(), SFU.LDim(),
                            SFV.LockedBuffer(), SFV.LDim(),
                 Scalar(1), D_.Buffer(),        D_.LDim() );
            }
            else
            {
                Dense<Scalar>& SFU = SFD_;
                Dense<Scalar>& SFV = SF.D;
                blas::Gemm                                   
                ('N', option, m, n, SF.rank,
                 Scalar(1), SFU.LockedBuffer(), SFU.LDim(),
                            SFV.LockedBuffer(), SFV.LDim(),
                 Scalar(1), D_.Buffer(),        D_.LDim() );
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
                        hmat_tools::Scale( Scalar(0), SF.D );
                        for( int i=0; i<m; i++)
                            SF.D.Set(i,i,Scalar(1));
                    }
                    else
                    {
                        hmat_tools::Copy( D_, SF.D );
                    }
                }
                else
                {
                    SF.rank = minDim;
                    if( m == minDim )
                    {
                        hmat_tools::Transpose( D_, SF.D );
                    }
                    else
                    {
                        SF.D.Resize( n, n, n);
                        hmat_tools::Scale( Scalar(0), SF.D );
                        for( int i=0; i<n; i++)
                            SF.D.Set(i,i,Scalar(1));
                    }
                }
            }
            else
            {
                SF.rank = maxRank; 
                std::vector<Real> sigma( minDim );
                std::vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                std::vector<Real> realwork( lapack::SVDRealWorkSize(m,n) );
                if( inTargetTeam_ )
                {
                    lapack::SVD
                    ( 'O', 'N', m, n, D_.Buffer(), D_.LDim(),
                      &sigma[0], 0, 1, 0, 1,
                      &work[0], work.size(), &realwork[0] );

                    SF.D.Resize( m, maxRank );
                    for( int j=0; j<maxRank; j++ )
                        for( int i=0; i<m; i++)
                            SF.D.Set(i,j,D_.Get(i,j)*sigma[j]);
                }
                else
                {
                    lapack::SVD
                    ( 'N', 'O', m, n, D_.Buffer(), D_.LDim(),
                      &sigma[0], 0, 1, 0, 1,
                      &work[0], work.size(), &realwork[0] );

                    SF.D.Resize( n, maxRank );
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
        if( level_ < startLevel )
            break;
//Print
//mpi::Comm teamp = teams_->Team( 0 );
//const int teamRankp = mpi::CommRank( teamp );
//if( level_ == 3 && teamRankp == 0 && block_.type==LOW_RANK && Vtmp_.Height() == 2 && Utmp_.Height() == 2)
//{
//    BL_.Print("_BL");
//    BR_.Print("_BR");
//}
        if( !haveDenseUpdate_ )
        {
            if( !BL_.IsEmpty() )                                       
            { 
//Print                                                                
//_BL.Print("_BL");
                LowRank<Scalar> &F = *block_.data.F;
                Dense<Scalar> &U = F.U;
                U.Resize(Utmp_.Height(), BL_.Width(), Utmp_.Height());
                
                blas::Gemm
                ('N', 'N', Utmp_.Height(), BL_.Width(), Utmp_.Width(),
                 Scalar(1), Utmp_.LockedBuffer(), Utmp_.LDim(),
                            BL_.LockedBuffer(), BL_.LDim(),
                 Scalar(0), U.Buffer(),         U.LDim() );
            }
            if( !BR_.IsEmpty() )
            {                                                      
//Print                                                                
//_BR.Print("_BR");
                LowRank<Scalar> &F = *block_.data.F;
                Dense<Scalar> &V = F.V;
                V.Resize(Vtmp_.Height(), BR_.Width(), Vtmp_.Height());
                
                blas::Gemm
                ('N', 'N', Vtmp_.Height(), BR_.Width(), Vtmp_.Width(),
                 Scalar(1), Vtmp_.LockedBuffer(), Vtmp_.LDim(),
                            BR_.LockedBuffer(), BR_.LDim(),
                 Scalar(0), V.Buffer(),         V.LDim() );
            }
        }
        else
        {
            LowRank<Scalar> &F = *block_.data.F;
            const int m = F.Height();
            const int n = F.Width();
            const int minDim = std::min( m, n );
            const int maxRank = MaxRank();
            const int r = F.Rank();

            // Add U V^[T/H] onto the dense update
            const char option = 'T';
            blas::Gemm
            ( 'N', option, m, n, r, 
              Scalar(1), F.U.LockedBuffer(), F.U.LDim(),
                         F.V.LockedBuffer(), F.V.LDim(),
              Scalar(1), D_.Buffer(),        D_.LDim() );

            if( minDim <= maxRank )
            {
                if( m == minDim )
                {
                    // Make U := I and V := D_^[T/H]
                    F.U.Resize( minDim, minDim );
                    hmat_tools::Scale( Scalar(0), F.U );
                    for( int j=0; j<minDim; ++j )
                        F.U.Set(j,j,Scalar(1));
                    hmat_tools::Transpose( D_, F.V );
                }
                else
                {
                    // Make U := D_ and V := I
                    hmat_tools::Copy( D_, F.U );
                    F.V.Resize( minDim, minDim );
                    hmat_tools::Scale( Scalar(0), F.V );
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
                std::vector<Real> sigma( minDim );
                std::vector<Scalar> work( lapack::SVDWorkSize(m,n) );
                std::vector<Real> realWork( lapack::SVDRealWorkSize(m,n) );
                lapack::SVD
                ( 'O', 'S', m, n, D_.Buffer(), D_.LDim(), 
                  &sigma[0], 0, 1, VH.Buffer(), VH.LDim(), 
                  &work[0], work.size(), &realWork[0] );

                // Form U with the truncated left singular vectors scaled
                // by the corresponding singular values
                F.U.Resize( m, maxRank );
                for( int j=0; j<maxRank; ++j )
                    for( int i=0; i<m; ++i )
                        F.U.Set(i,j,sigma[j]*D_.Get(i,j));

                // Form V with the truncated right singular vectors
                F.V.Resize( n, maxRank );
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
        if(level_ < startLevel )
            break;
        if( inSourceTeam_ )
        {
            SplitDense& SD = *block_.data.SD;
            const int m = SD.D.Height();
            const int n = SD.D.Width();

            UMap_.ResetIterator();
            VMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();

            const char option = 'T';

            blas::Gemm
            ('N', option, m, n, U.Width(),
             Scalar(1), U.LockedBuffer(), U.LDim(),
                        V.LockedBuffer(), V.LDim(),
             Scalar(1), SD.D.Buffer(), SD.D.LDim() );

            VMap_.Clear();
        }
        break;
    }
    case DENSE:
    {
        if( level_ < startLevel )
            break;
            Dense<Scalar>& D = *block_.data.D;
            const int m = D.Height();
            const int n = D.Width();

            UMap_.ResetIterator();
            VMap_.ResetIterator();
            const Dense<Scalar>& U = *UMap_.CurrentEntry();
            const Dense<Scalar>& V = *VMap_.CurrentEntry();

            const char option = 'T';

            blas::Gemm
            ('N', option, m, n, U.Width(),
             Scalar(1), U.LockedBuffer(), U.LDim(),
                        V.LockedBuffer(), V.LDim(),
             Scalar(1), D.Buffer(), D.LDim() );

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
DistHMat2d<Scalar>::MultiplyHMatCompressFCleanup
( int startLevel, int endLevel )
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
        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFCleanup
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
        Utmp_.Clear();
        Vtmp_.Clear();
        USqr_.Clear();
        VSqr_.Clear();
        USqrEig_.resize(0);
        VSqrEig_.resize(0);
        BSqr_.Clear();
        BSqrEig_.resize(0);
        BSqrU_.Clear();
        BSqrVH_.Clear();
        BSigma_.resize(0);
        BL_.Clear();
        BR_.Clear();
        UMap_.Clear();
        VMap_.Clear();
        ZMap_.Clear();
        colXMap_.Clear();
        rowXMap_.Clear();
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatCompressFCompressless
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatCompressFCompressless");
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
                    node.Child(t,s).MultiplyHMatCompressFCompressless
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( inTargetTeam_ )                                  
        {
            int offset=0;
            int totalrank=colXMap_.TotalWidth() + UMap_.TotalWidth();
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &Utmp = DF.ULocal;
            DF.rank = totalrank;
//Print
//std::cout << Utmp.Height()*Utmp.Width() << std::endl;
            if( Utmp.Height()*Utmp.Width() != 0)
            {
                offset=Utmp.Width();
                totalrank+=Utmp.Width();
            }
            Utmp.Resize( LH, totalrank, LH );
            
            int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
            {
                Dense<Scalar>& U = *UMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( inSourceTeam_ )
        {                                                      
            int offset=0;
            int totalrank=rowXMap_.TotalWidth() + VMap_.TotalWidth();
            DistLowRank &DF = *block_.data.DF;
            Dense<Scalar> &Vtmp = DF.VLocal;
            DF.rank = totalrank;
//Print
//std::cout << Vtmp.Height()*Vtmp.Width() << std::endl;
            if( Vtmp.Height()*Vtmp.Width() != 0)
            {
                offset=Vtmp.Width();
                totalrank+=Vtmp.Width();
            }
            Vtmp.Resize( LW, totalrank, LW );
            
            int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& V = *rowXMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( inTargetTeam_ )                                  
        {
            int offset=0;
            int totalrank=colXMap_.TotalWidth() + UMap_.TotalWidth();
            SplitLowRank &SF = *block_.data.SF;
            Dense<Scalar> &Utmp = SF.D;
            SF.rank = totalrank;
//Print
//std::cout << Utmp.Height()*Utmp.Width() << std::endl;
            if( Utmp.Height()*Utmp.Width() != 0)
            {
                offset=Utmp.Width();
                totalrank+=Utmp.Width();
            }
            Utmp.Resize( LH, totalrank, LH );
            
            int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
            {
                Dense<Scalar>& U = *UMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( inSourceTeam_ )
        {                                                      
            int offset=0;
            int totalrank=rowXMap_.TotalWidth() + VMap_.TotalWidth();
            SplitLowRank &SF = *block_.data.SF;
            Dense<Scalar> &Vtmp = SF.D;
            SF.rank = totalrank;
//Print
//std::cout << Vtmp.Height()*Vtmp.Width() << std::endl;
            if( Vtmp.Height()*Vtmp.Width() != 0)
            {
                offset=Vtmp.Width();
                totalrank+=Vtmp.Width();
            }
            Vtmp.Resize( LW, totalrank, LW );
            
            int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& V = *rowXMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
        }
        break;
    }
    case LOW_RANK:
    {
        if( level_ < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( inTargetTeam_ )                                  
        {
            int offset=0;
            int totalrank=colXMap_.TotalWidth() + UMap_.TotalWidth();
            LowRank<Scalar> &F = *block_.data.F;
            Dense<Scalar> &Utmp = F.U;
//Print
//std::cout << Utmp.Height()*Utmp.Width() << std::endl;
            if( Utmp.Height()*Utmp.Width() != 0)
            {
                offset=Utmp.Width();
                totalrank+=Utmp.Width();
            }
            Utmp.Resize( LH, totalrank, LH );
            
            int numEntries = UMap_.Size();
            UMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,UMap_.Increment() )
            {
                Dense<Scalar>& U = *UMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = colXMap_.Size();
            colXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,colXMap_.Increment() )
            {
                Dense<Scalar>& U = *colXMap_.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( inSourceTeam_ )
        {                                                      
            int offset=0;
            int totalrank=rowXMap_.TotalWidth() + VMap_.TotalWidth();
            LowRank<Scalar> &F = *block_.data.F;
            Dense<Scalar> &Vtmp = F.V;
//Print
//std::cout << Vtmp.Height()*Vtmp.Width() << std::endl;
            if( Vtmp.Height()*Vtmp.Width() != 0)
            {
                offset=Vtmp.Width();
                totalrank+=Vtmp.Width();
            }
            Vtmp.Resize( LW, totalrank, LW );
            
            int numEntries = VMap_.Size();
            VMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,VMap_.Increment() )
            {
                Dense<Scalar>& V = *VMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = rowXMap_.Size();
            rowXMap_.ResetIterator();
            for( int i=0; i<numEntries; ++i,rowXMap_.Increment() )
            {
                Dense<Scalar>& V = *rowXMap_.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
        }
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
