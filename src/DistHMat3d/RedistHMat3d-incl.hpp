/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//

template<typename Scalar>
std::size_t
DistHMat3d<Scalar>::PackedSizes
( Vector<std::size_t>& packedSizes,
  const HMat3d<Scalar>& H, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::PackedSizes");
#endif
    mpi::Comm comm = teams.Team(0);
    const unsigned p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( p > (1u<<(3*(H.numLevels_-1))) )
        throw std::logic_error("More than 8^(numLevels-1) processes.");

    // Initialize for the recursion
    packedSizes.Resize( p );
    MemZero( &packedSizes[0], p );
    Vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );

    // Count the top-level header data
    const std::size_t headerSize = 17*sizeof(int) + sizeof(bool);
    for( unsigned i=0; i<p; ++i )
        packedSizes[i] += headerSize;

    // Recurse on this block to compute the packed sizes
    PackedSizesRecursion( packedSizes, localSizes, 0, 0, p, H );
    std::size_t totalSize = 0;
    for( unsigned i=0; i<p; ++i )
        totalSize += packedSizes[i];
    return totalSize;
}

template<typename Scalar>
std::size_t
DistHMat3d<Scalar>::Pack
( Vector<byte*>& packedSubs,
  const HMat3d<Scalar>& H, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::Pack");
#endif
    mpi::Comm comm = teams.Team(0);
    const int p = mpi::CommSize( comm );
    Vector<byte*> heads = packedSubs;
    Vector<byte**> headPointers(p);
    for( int i=0; i<p; ++i )
        headPointers[i] = &heads[i];

    // Write the top-level header data
    for( int i=0; i<p; ++i )
    {
        byte** h = headPointers[i];

        Write( h, H.numLevels_ );
        Write( h, H.maxRank_ );
        Write( h, H.sourceOffset_ );
        Write( h, H.targetOffset_ );
        // Write( h, H.type_ );
        Write( h, H.stronglyAdmissible_ );
        Write( h, H.xSizeSource_ );
        Write( h, H.xSizeTarget_ );
        Write( h, H.ySizeSource_ );
        Write( h, H.ySizeTarget_ );
        Write( h, H.zSizeSource_ );
        Write( h, H.zSizeTarget_ );
        Write( h, H.xSource_ );
        Write( h, H.xTarget_ );
        Write( h, H.ySource_ );
        Write( h, H.yTarget_ );
        Write( h, H.zSource_ );
        Write( h, H.zTarget_ );
    }

    Vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );
    PackRecursion( headPointers, localSizes, 0, 0, p, H );

    std::size_t totalSize = 0;
    for( int i=0; i<p; ++i )
        totalSize += (*headPointers[i]-packedSubs[i]);
    return totalSize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::ComputeLocalHeight
( int p, int rank, const HMat3d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ComputeLocalHeight");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localHeight;
    int xSize = H.xSizeTarget_;
    int ySize = H.ySizeTarget_;
    int zSize = H.zSizeTarget_;
    ComputeLocalDimensionRecursion( localHeight, xSize, ySize, zSize, p, rank );
    return localHeight;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::ComputeLocalWidth
( int p, int rank, const HMat3d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ComputeLocalWidth");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localWidth;
    int xSize = H.xSizeSource_;
    int ySize = H.ySizeSource_;
    int zSize = H.zSizeSource_;
    ComputeLocalDimensionRecursion( localWidth, xSize, ySize, zSize, p, rank );
    return localWidth;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::ComputeFirstLocalRow
( int p, int rank, const HMat3d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ComputeFirstLocalRow");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, H.xSizeTarget_, H.ySizeTarget_, H.zSizeTarget_, p, rank );
    return firstLocalRow;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::ComputeFirstLocalCol
( int p, int rank, const HMat3d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ComputeFirstLocalCol");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, H.xSizeSource_, H.ySizeSource_, H.zSizeSource_, p, rank );
    return firstLocalCol;
}

template<typename Scalar>
void
DistHMat3d<Scalar>::ComputeLocalSizes
( Vector<int>& localSizes, const HMat3d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ComputeLocalSizes");
    const int p = localSizes.Size();
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( (H.xSizeSource_ != H.xSizeTarget_) ||
        (H.ySizeSource_ != H.ySizeTarget_) ||
        (H.zSizeSource_ != H.zSizeTarget_) )
        throw std::logic_error("Routine meant for square nodes");
#endif
    ComputeLocalSizesRecursion
    ( &localSizes[0], localSizes.Size(), H.xSizeSource_, H.ySizeSource_, H.zSizeSource_ );
}

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar>
void
DistHMat3d<Scalar>::PackedSizesRecursion
( Vector<std::size_t>& packedSizes,
  const Vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const HMat3d<Scalar>& H )
{
    typedef HMat3d<Scalar> HMat;

    for( int i=0; i<teamSize; ++i )
        packedSizes[sourceRankOffset+i] += sizeof(BlockType);
    if( sourceRankOffset != targetRankOffset )
        for( int i=0; i<teamSize; ++i )
            packedSizes[targetRankOffset+i] += sizeof(BlockType);

    const typename HMat::Block& block = H.block_;
    const int m = H.Height();
    const int n = H.Width();
    switch( block.type )
    {
    case HMat::NODE:
    {
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes, sourceRank, targetRank, 1,
                      block.data.N->Child(t,s) );
        }
        else if( teamSize == 2 )
        {
            // Give the upper-left 2x2 to the first halves of the teams
            // and the lower-right 2x2 to the second halves.
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+s/4, targetRankOffset+t/4, 1,
                      block.data.N->Child(t,s) );
        }
        else if( teamSize == 4 )
        {
            // Give the upper-left 4x4 to the first halves of the teams
            // and the lower-right 4x4 to the second halves.
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+s/2, targetRankOffset+t/2, 1,
                      block.data.N->Child(t,s) );
        }
        else // team Size >= 4
        {
            // Give each diagonal block of the 4x4 partition to a different
            // quarter of the teams
            const int newTeamSize = teamSize/8;
            for( int t=0; t<8; ++t )
                for( int s=0; s<8; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+newTeamSize*s,
                      targetRankOffset+newTeamSize*t, newTeamSize,
                      block.data.N->Child(t,s) );
        }
        break;
    }
    case HMat::NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    }
    case HMat::LOW_RANK:
    {
        const int r = block.data.F->Rank();
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store a serial low-rank matrix
                packedSizes[sourceRank] += sizeof(int) + (m+n)*r*sizeof(Scalar);
            }
            else
            {
                // Store a split low-rank matrix

                // The source and target processes store the matrix rank and
                // their factor's entries.
                packedSizes[sourceRank] += sizeof(int) + n*r*sizeof(Scalar);
                packedSizes[targetRank] += sizeof(int) + m*r*sizeof(Scalar);
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // Every process owns a piece of both U and V. Store those
                // pieces along with the matrix rank.
                std::cerr << "WARNING: Unlikely admissible case." << std::endl;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    const int localSize = localSizes[sourceRank];
                    packedSizes[sourceRank] +=
                        sizeof(int) + 2*localSize*r*sizeof(Scalar);
                }
            }
            else
            {
                // Each process either owns a piece of U or V. Store it along
                // with the matrix rank.

                // Write out the source information
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    packedSizes[sourceRank] +=
                        sizeof(int) + localSizes[sourceRank]*r*sizeof(Scalar);
                }

                // Write out the target information
                for( int i=0; i<teamSize; ++i )
                {
                    const int targetRank = targetRankOffset + i;
                    packedSizes[targetRank] +=
                        sizeof(int) + localSizes[targetRank]*r*sizeof(Scalar);
                }
            }
        }
        break;
    }
    case HMat::DENSE:
    {
        const Dense<Scalar>& D = *block.data.D;
        const MatrixType type = D.Type();

        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store the type and entries
                packedSizes[sourceRank] += sizeof(MatrixType);
                if( type == GENERAL )
                    packedSizes[sourceRank] += m*n*sizeof(Scalar);
                else
                    packedSizes[sourceRank] += ((m*m+m)/2)*sizeof(Scalar);
            }
            else
            {
                // The source side stores the matrix type and entries
                packedSizes[sourceRank] += sizeof(MatrixType);
                if( type == GENERAL )
                    packedSizes[sourceRank] += m*n*sizeof(Scalar);
                else
                    packedSizes[sourceRank] += ((m*m+m)/2)*sizeof(Scalar);
            }
        }
        else
        {
#ifndef RELEASE
            throw std::logic_error("Too many processes");
#endif
        }
        break;
    }
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::PackRecursion
( Vector<byte**>& headPointers,
  const Vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const HMat3d<Scalar>& H )
{
    typedef HMat3d<Scalar> HMat;

    const typename HMat::Block& block = H.block_;
    const int m = H.Height();
    const int n = H.Width();
    switch( block.type )
    {
    case HMat::NODE:
    {
        const typename HMat::Node& node = *block.data.N;
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                Write( headPointers[sourceRank], NODE );
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        PackRecursion
                        ( headPointers, localSizes, sourceRank, targetRank, 1,
                          node.Child(t,s) );
            }
            else
            {
                Write( headPointers[sourceRank], SPLIT_NODE );
                Write( headPointers[targetRank], SPLIT_NODE );
                for( int t=0; t<8; ++t )
                    for( int s=0; s<8; ++s )
                        PackRecursion
                        ( headPointers, localSizes, sourceRank, targetRank, 1,
                          node.Child(t,s) );
            }
        }
        else if( teamSize == 2 )
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            for( int i=0; i<teamSize; ++i )
                Write( headPointers[sourceRankOffset+i], DIST_NODE );
            if( sourceRankOffset != targetRankOffset )
                for( int i=0; i<teamSize; ++i )
                    Write( headPointers[targetRankOffset+i], DIST_NODE );
            const int newTeamSize = teamSize/2;
            // Top-left block
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset, newTeamSize,
                      node.Child(t,s) );
            // Top-right block
            for( int t=0; t<4; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, targetRankOffset,
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=4; t<8; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset+newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=4; t<8; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize,
                      targetRankOffset+newTeamSize,
                      newTeamSize, node.Child(t,s) );
        }
        else if( teamSize == 4 )
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            for( int i=0; i<teamSize; ++i )
                Write( headPointers[sourceRankOffset+i], DIST_NODE );
            if( sourceRankOffset != targetRankOffset )
                for( int i=0; i<teamSize; ++i )
                    Write( headPointers[targetRankOffset+i], DIST_NODE );
            const int newTeamSize = teamSize/4;
            // Top-left block
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+(s/2)*newTeamSize,
                      targetRankOffset+(t/2)*newTeamSize,
                      newTeamSize,
                      node.Child(t,s) );
            // Top-right block
            for( int t=0; t<4; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+(s/2)*newTeamSize,
                      targetRankOffset+(t/2)*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=4; t<8; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+(s/2)*newTeamSize,
                      targetRankOffset+(t/2)*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=4; t<8; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+(s/2)*newTeamSize,
                      targetRankOffset+(t/2)*newTeamSize,
                      newTeamSize, node.Child(t,s) );
        }
        else
        {
            // Recurse in 2x2 blocks:
            // top-left, top-right, bottom-left, bottom-right
            for( int i=0; i<teamSize; ++i )
                Write( headPointers[sourceRankOffset+i], DIST_NODE );
            if( sourceRankOffset != targetRankOffset )
                for( int i=0; i<teamSize; ++i )
                    Write( headPointers[targetRankOffset+i], DIST_NODE );
            const int newTeamSize = teamSize/8;
            // Top-left block
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize,
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Top-right block
            for( int t=0; t<4; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize,
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=4; t<8; ++t )
                for( int s=0; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize,
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=4; t<8; ++t )
                for( int s=4; s<8; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize,
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
        }
        break;
    }
    case HMat::NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    }
    case HMat::LOW_RANK:
    {
        const Dense<Scalar>& U = block.data.F->U;
        const Dense<Scalar>& V = block.data.F->V;
        const int r = U.Width();
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store a serial low-rank representation
                byte** h = headPointers[sourceRank];
                Write( h, LOW_RANK );

                // Store the rank and matrix entries
                Write( h, r );
                for( int j=0; j<r; ++j )
                    Write( h, U.LockedBuffer(0,j), m );
                for( int j=0; j<r; ++j )
                    Write( h, V.LockedBuffer(0,j), n );
            }
            else
            {
                // Store a split low-rank representation
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];
                Write( hSource, SPLIT_LOW_RANK );
                Write( hTarget, SPLIT_LOW_RANK );

                // Store the rank and entries of V on the source side
                Write( hSource, r );
                for( int j=0; j<r; ++j )
                    Write( hSource, V.LockedBuffer(0,j), n );

                // Store the rank and entries of U on the target side
                Write( hTarget, r );
                for( int j=0; j<r; ++j )
                    Write( hTarget, U.LockedBuffer(0,j), m );
            }
        }
        else
        {
            if( sourceRankOffset == targetRankOffset )
            {
                // NOTE: This should only happen when there is a weird
                //       admissibility condition that allows diagonal blocks
                //       to be low-rank.
#ifndef RELEASE
                std::cerr << "WARNING: Unlikely admissible case." << std::endl;
#endif

                // Store a distributed low-rank representation
                int offset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    // Store the header information
                    const int rank = sourceRankOffset + i;
                    byte** h = headPointers[rank];
                    Write( h, DIST_LOW_RANK );
                    Write( h, r );

                    // Store our local U and V
                    const int localSize = localSizes[rank];
                    for( int j=0; j<r; ++j )
                        Write( h, U.LockedBuffer(offset,j), localSize );
                    for( int j=0; j<r; ++j )
                        Write( h, V.LockedBuffer(offset,j), localSize );
                    offset += localSize;
                }
            }
            else
            {
                // Store a distributed split low-rank representation

                // Store the source data
                int offset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int sourceRank = sourceRankOffset + i;
                    byte** hSource = headPointers[sourceRank];

                    Write( hSource, DIST_LOW_RANK );
                    Write( hSource, r );

                    const int localWidth = localSizes[sourceRank];
                    for( int j=0; j<r; ++j )
                        Write( hSource, V.LockedBuffer(offset,j), localWidth );
                    offset += localWidth;
                }

                // Store the target data
                offset = 0;
                for( int i=0; i<teamSize; ++i )
                {
                    const int targetRank = targetRankOffset + i;
                    byte** hTarget = headPointers[targetRank];

                    Write( hTarget, DIST_LOW_RANK );
                    Write( hTarget, r );

                    const int localHeight = localSizes[targetRank];
                    for( int j=0; j<r; ++j )
                        Write( hTarget, U.LockedBuffer(offset,j), localHeight );
                    offset += localHeight;
                }
            }
        }
        break;
    }
    case HMat::DENSE:
    {
        const Dense<Scalar>& D = *block.data.D;
        const MatrixType type = D.Type();
        if( teamSize == 1 )
        {
            const int sourceRank = sourceRankOffset;
            const int targetRank = targetRankOffset;
            if( sourceRank == targetRank )
            {
                // Store a serial dense matrix
                byte** h = headPointers[sourceRank];
                Write( h, DENSE );
                Write( h, type );
                if( type == GENERAL )
                    for( int j=0; j<n; ++j )
                        Write( h, D.LockedBuffer(0,j), m );
                else
                    for( int j=0; j<n; ++j )
                        Write( h, D.LockedBuffer(j,j), m-j );
            }
            else
            {
                // Store a split dense matrix
                byte** hSource = headPointers[sourceRank];
                byte** hTarget = headPointers[targetRank];

                // Store the source data
                Write( hSource, SPLIT_DENSE );
                Write( hSource, type );
                if( type == GENERAL )
                    for( int j=0; j<n; ++j )
                        Write( hSource, D.LockedBuffer(0,j), m );
                else
                    for( int j=0; j<n; ++j )
                        Write( hSource, D.LockedBuffer(j,j), m-j );

                // There is no target data to store
                Write( hTarget, SPLIT_DENSE );
            }
        }
#ifndef RELEASE
        else
            throw std::logic_error("Too many processes");
#endif
        break;
    }
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::ComputeLocalDimensionRecursion
( int& localDim, int& xSize, int& ySize, int& zSize, int p, int rank )
{
    if( p >= 8 )
    {
        const int subteam = rank/(p/8);
        const int subteamRank = rank-subteam*(p/8);
        const bool onRight = (subteam & 1);
        const bool onTop = ((subteam/2)&1);
        const bool onBack = (subteam/4);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        xSize = ( onRight ? xRightSize : xLeftSize   );
        ySize = ( onTop   ? yTopSize   : yBottomSize );
        zSize = ( onBack ? zBackSize : zFrontSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, zSize, p/8, subteamRank );
    }
    else if( p == 4 )
    {
        const int subteam = rank;
        const int subteamRank = rank-subteam;
        const bool onTop = (subteam&1);
        const bool onBack = (subteam/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        ySize = ( onTop   ? yTopSize   : yBottomSize );
        zSize = ( onBack ? zBackSize : zFrontSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, zSize, p/4, subteamRank );

    }
    else if( p == 2 )
    {
        const int subteam = rank;
        const int subteamRank = rank-subteam;

        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        zSize = ( subteam ? zBackSize : zFrontSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, zSize, p/2, subteamRank );
    }
    else // p == 1
        localDim = xSize*ySize*zSize;
}

template<typename Scalar>
void
DistHMat3d<Scalar>::ComputeFirstLocalIndexRecursion
( int& firstLocalIndex, int xSize, int ySize, int zSize, int p, int rank )
{
    if( p >= 8 )
    {
        const int subteam = rank/(p/8);
        const int subteamRank = rank-subteam*(p/8);
        const bool onRight = (subteam & 1);
        const bool onTop = ((subteam/2)&1);
        const bool onBack = (subteam/4);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        // Add on this level of offsets
        if( onRight && onTop && onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize
                                 + xSize*yBottomSize*zBackSize
                                 + xLeftSize*yTopSize*zBackSize );
        else if( onTop && onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize
                                 + xSize*yBottomSize*zBackSize );
        else if( onRight && onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize
                                 + xLeftSize*yBottomSize*zBackSize );
        else if( onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize );
        else if( onRight && onTop )
            firstLocalIndex += ( xSize*yBottomSize*zFrontSize
                                 + xLeftSize*yTopSize*zFrontSize );
        else if( onTop )
            firstLocalIndex += ( xSize*yBottomSize*zFrontSize );
        else if( onRight )
            firstLocalIndex += ( xLeftSize*yBottomSize*zFrontSize );


        const int xSizeNew = ( onRight ? xRightSize : xLeftSize   );
        const int ySizeNew = ( onTop   ? yTopSize   : yBottomSize );
        const int zSizeNew = ( onBack  ? zBackSize  : zFrontSize  );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSizeNew, ySizeNew, zSizeNew, p/8, subteamRank );
    }
    else if( p == 4 )
    {
        const int subteam = rank/(p/4);
        const int subteamRank = rank-subteam*(p/4);
        const bool onTop = (subteam&1);
        const bool onBack = (subteam/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        // Add on this level of offsets
        if( onTop && onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize
                                 + xSize*yBottomSize*zBackSize );
        else if( onBack )
            firstLocalIndex += ( xSize*ySize*zFrontSize );
        else if( onTop )
            firstLocalIndex += ( xSize*yBottomSize*zFrontSize );


        const int ySizeNew = ( onTop   ? yTopSize   : yBottomSize );
        const int zSizeNew = ( onBack  ? zBackSize  : zFrontSize  );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSize, ySizeNew, zSizeNew, p/4, subteamRank );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;

        // Add on this level of offsets
        if( subteam )
            firstLocalIndex += xSize*ySize*zFrontSize;

        const int zSizeNew = ( subteam ? zBackSize  : zFrontSize  );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSize, ySize, zSizeNew, p/2, subteamRank );
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::ComputeLocalSizesRecursion
( int* localSizes, int teamSize, int xSize, int ySize, int zSize )
{
    if( teamSize >=8 )
    {
        const int newTeamSize = teamSize/8;
        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;
        // Front-Bottom-left piece of 3d domain
        ComputeLocalSizesRecursion
        ( localSizes, newTeamSize, xLeftSize, yBottomSize, zFrontSize );
        // Front-Bottom-right piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[newTeamSize], newTeamSize,
          xRightSize, yBottomSize, zFrontSize );
        // Front-Top-left piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[2*newTeamSize], newTeamSize,
          xLeftSize, yTopSize, zFrontSize );
        // Front-Top-right piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[3*newTeamSize], newTeamSize,
          xRightSize, yTopSize, zFrontSize );
        // Back-Bottom-left piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[4*newTeamSize], newTeamSize,
          xLeftSize, yBottomSize, zBackSize );
        // Back-Bottom-right piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[5*newTeamSize], newTeamSize,
          xRightSize, yBottomSize, zBackSize );
        // Back-Top-left piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[6*newTeamSize], newTeamSize,
          xLeftSize, yTopSize, zBackSize );
        // Back-Top-right piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[7*newTeamSize], newTeamSize,
          xRightSize, yTopSize, zBackSize );
    }
    else if( teamSize == 4 )
    {
        const int newTeamSize = teamSize/4;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;
        // Front-Bottom piece of 3d domain
        ComputeLocalSizesRecursion
        ( localSizes, newTeamSize, xSize, yBottomSize, zFrontSize );
        // Front-Top piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[1], newTeamSize,
          xSize, yTopSize, zFrontSize );
        // Back-Bottom piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[2], newTeamSize,
          xSize, yBottomSize, zBackSize );
        // Back-Top piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[3], newTeamSize,
          xSize, yTopSize, zBackSize );
    }
    else if( teamSize == 2 )
    {
        const int newTeamSize = teamSize/2;
        const int zFrontSize = zSize/2;
        const int zBackSize = zSize - zFrontSize;
        // Front piece of 3d domain
        ComputeLocalSizesRecursion
        ( localSizes, newTeamSize, xSize, ySize, zFrontSize );
        // Back piece of 3d domain
        ComputeLocalSizesRecursion
        ( &localSizes[1], newTeamSize, xSize, ySize, zBackSize );
    }
    else
        localSizes[0] = xSize*ySize*zSize;
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
DistHMat3d<Scalar>::DistHMat3d
( const byte* packedSub, const Teams& teams )
: haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::DistHMat3d");
#endif
    Unpack( packedSub, teams );
}

template<typename Scalar>
std::size_t
DistHMat3d<Scalar>::Unpack
( const byte* packedDistHMat, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::Unpack");
#endif
    teams_ = &teams;
    level_ = 0;
    inSourceTeam_ = true;
    inTargetTeam_ = true;
    sourceRoot_ = 0;
    targetRoot_ = 0;

    const byte* head = packedDistHMat;

    // Read in the header information
    numLevels_          = Read<int>( head );
    maxRank_            = Read<int>( head );
    sourceOffset_       = Read<int>( head );
    targetOffset_       = Read<int>( head );
    //_type             = Read<MatrixType>( head );
    stronglyAdmissible_ = Read<bool>( head );
    xSizeSource_        = Read<int>( head );
    xSizeTarget_        = Read<int>( head );
    ySizeSource_        = Read<int>( head );
    ySizeTarget_        = Read<int>( head );
    zSizeSource_        = Read<int>( head );
    zSizeTarget_        = Read<int>( head );
    xSource_            = Read<int>( head );
    xTarget_            = Read<int>( head );
    ySource_            = Read<int>( head );
    yTarget_            = Read<int>( head );
    zSource_            = Read<int>( head );
    zTarget_            = Read<int>( head );

    UnpackRecursion( head );
    return (head-packedDistHMat);
}

template<typename Scalar>
void
DistHMat3d<Scalar>::UnpackRecursion( const byte*& head )
{
    mpi::Comm team = teams_->Team( level_ );
    if( !inSourceTeam_ && !inTargetTeam_ )
    {
        block_.type = EMPTY;
        return;
    }

    // Read in the information for the new block
    block_.Clear();
    block_.type = Read<BlockType>( head );
    const int m = Height();
    const int n = Width();
    switch( block_.type )
    {
    case DIST_NODE:
    {
        block_.data.N = NewNode();
        Node& node = *block_.data.N;

        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        if( teamSize >= 8 )
        {
            const int subteam = teamRank/(teamSize/8);
            // Top-left block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/8);
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/8);

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/8);
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/8);

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/8);
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/8);

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/8);
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/8);

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
        }
        else if( teamSize == 4 )
        {
            // Top-left block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/2) ),
                          inTargetTeam_ && ( teamRank == (t/2) ),
                          sourceRoot_+(s/2), targetRoot_+(t/2) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/2) ),
                          inTargetTeam_ && ( teamRank == (t/2) ),
                          sourceRoot_+(s/2), targetRoot_+(t/2) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/2) ),
                          inTargetTeam_ && ( teamRank == (t/2) ),
                          sourceRoot_+(s/2), targetRoot_+(t/2) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/2) ),
                          inTargetTeam_ && ( teamRank == (t/2) ),
                          sourceRoot_+(s/2), targetRoot_+(t/2) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
        }
        else  // teamSize == 2
        {
#ifndef RELEASE
            if( teamSize != 2 )
                throw std::logic_error("Team size was not 2 as expected");
#endif

            // Top-left block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/4) ),
                          inTargetTeam_ && ( teamRank == (t/4) ),
                          sourceRoot_+(s/4), targetRoot_+(t/4) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/4) ),
                          inTargetTeam_ && ( teamRank == (t/4) ),
                          sourceRoot_+(s/4), targetRoot_+(t/4) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/4) ),
                          inTargetTeam_ && ( teamRank == (t/4) ),
                          sourceRoot_+(s/4), targetRoot_+(t/4) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=4,tOffset=node.targetSizes[0]+node.targetSizes[1]
                                +node.targetSizes[2]+node.targetSizes[3];
                 t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=4,sOffset=node.sourceSizes[0]+node.sourceSizes[1]
                                    +node.sourceSizes[2]+node.sourceSizes[3];
                     s<8; sOffset+=node.sourceSizes[s],++s )
                {

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/4) ),
                          inTargetTeam_ && ( teamRank == (t/4) ),
                          sourceRoot_+(s/4), targetRoot_+(t/4) );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    case NODE:
    {
        block_.data.N = NewNode();
        Node& node = *block_.data.N;

        for( int t=0,tOffset=0; t<8; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0,sOffset=0; s<8; sOffset+=node.sourceSizes[s],++s )
            {
                node.children[s+8*t] =
                    new DistHMat3d<Scalar>
                    ( numLevels_-1, maxRank_, stronglyAdmissible_,
                      sourceOffset_+sOffset, targetOffset_+tOffset,
                      node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                      node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                      node.zSourceSizes[(s/4)], node.zTargetSizes[(t/4)],
                      2*xSource_+(s&1), 2*xTarget_+(t&1),
                      2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                      2*zSource_+(s/4), 2*zTarget_+(t/4),
                      *teams_, level_+1,
                      inSourceTeam_, inTargetTeam_,
                      sourceRoot_, targetRoot_ );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        block_.data.DF = new DistLowRank;
        DistLowRank& DF = *block_.data.DF;

        DF.rank = Read<int>( head );
        if( inSourceTeam_ )
        {
            // Read in V
            const int localWidth = LocalWidth();
            DF.VLocal.SetType( GENERAL );
            DF.VLocal.Resize( localWidth, DF.rank );
            DF.VLocal.Init();
            for( int j=0; j<DF.rank; ++j )
                Read( DF.VLocal.Buffer(0,j), head, localWidth );
        }
        if( inTargetTeam_ )
        {
            // Read in U
            const int localHeight = LocalHeight();
            DF.ULocal.SetType( GENERAL );
            DF.ULocal.Resize( localHeight, DF.rank );
            DF.ULocal.Init();
            for( int j=0; j<DF.rank; ++j )
                Read( DF.ULocal.Buffer(0,j), head, localHeight );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        block_.data.SF = new SplitLowRank;
        SplitLowRank& SF = *block_.data.SF;

        SF.rank = Read<int>( head );

        SF.D.SetType( GENERAL );
        if( inSourceTeam_ )
        {
            // Read in V
            SF.D.Resize( n, SF.rank );
            SF.D.Init();
            for( int j=0; j<SF.rank; ++j )
                Read( SF.D.Buffer(0,j), head, n );
        }
        else
        {
            // Read in U
            SF.D.Resize( m, SF.rank );
            SF.D.Init();
            for( int j=0; j<SF.rank; ++j )
                Read( SF.D.Buffer(0,j), head, m );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        block_.data.SD = new SplitDense;
        SplitDense& SD = *block_.data.SD;

        if( inSourceTeam_ )
        {
            const MatrixType type = Read<MatrixType>( head );
            SD.D.SetType( type );
            SD.D.Resize( m, n );
            SD.D.Init();
            if( type == GENERAL )
                for( int j=0; j<n; ++j )
                    Read( SD.D.Buffer(0,j), head, m );
            else
                for( int j=0; j<n; ++j )
                    Read( SD.D.Buffer(j,j), head, m-j );
        }
        break;
    }
    case LOW_RANK:
    {
        block_.data.F = new LowRank<Scalar>;
        LowRank<Scalar>& F = *block_.data.F;

        // Read in the rank
        const int r = Read<int>( head );

        // Read in U
        F.U.SetType( GENERAL );
        F.U.Resize( m, r );
        F.U.Init();
        for( int j=0; j<r; ++j )
            Read( F.U.Buffer(0,j), head, m );

        // Read in V
        F.V.SetType( GENERAL );
        F.V.Resize( n, r );
        F.V.Init();
        for( int j=0; j<r; ++j )
            Read( F.V.Buffer(0,j), head, n );
        break;
    }
    case DENSE:
    {
        block_.data.D = new Dense<Scalar>;
        Dense<Scalar>& D = *block_.data.D;

        const MatrixType type = Read<MatrixType>( head );

        D.SetType( type );
        D.Resize( m, n );
        D.Init();
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Read( D.Buffer(0,j), head, m );
        else
            for( int j=0; j<n; ++j )
                Read( D.Buffer(j,j), head, m-j );
        break;
    }
    default:
#ifndef RELEASE
        throw std::logic_error("Should not need to unpack empty submatrix");
#endif
        break;
    }
}

} // namespace dmhm
