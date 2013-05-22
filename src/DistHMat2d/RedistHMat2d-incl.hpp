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
DistHMat2d<Scalar>::PackedSizes
( std::vector<std::size_t>& packedSizes, 
  const HMat2d<Scalar>& H, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::PackedSizes");
#endif
    mpi::Comm comm = teams.Team(0);
    const unsigned p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( p > (1u<<(2*(H.numLevels_-1))) )
        throw std::logic_error("More than 4^(numLevels-1) processes.");

    // Initialize for the recursion
    packedSizes.resize( p );
    MemZero( &packedSizes[0], p );
    std::vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );

    // Count the top-level header data
    const std::size_t headerSize = 13*sizeof(int) + sizeof(bool);
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
DistHMat2d<Scalar>::Pack
( std::vector<byte*>& packedSubs, 
  const HMat2d<Scalar>& H, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Pack");
#endif
    mpi::Comm comm = teams.Team(0);
    const int p = mpi::CommSize( comm );
    std::vector<byte*> heads = packedSubs;
    std::vector<byte**> headPointers(p); 
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
        Write( h, H.xSource_ );
        Write( h, H.xTarget_ );
        Write( h, H.ySource_ );
        Write( h, H.yTarget_ );
    }

    std::vector<int> localSizes( p );
    ComputeLocalSizes( localSizes, H );
    PackRecursion( headPointers, localSizes, 0, 0, p, H );

    std::size_t totalSize = 0;
    for( int i=0; i<p; ++i )
        totalSize += (*headPointers[i]-packedSubs[i]);
    return totalSize;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::ComputeLocalHeight
( int p, int rank, const HMat2d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::ComputeLocalHeight");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localHeight;
    int xSize = H.xSizeTarget_;
    int ySize = H.ySizeTarget_;
    ComputeLocalDimensionRecursion( localHeight, xSize, ySize, p, rank );
    return localHeight;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::ComputeLocalWidth
( int p, int rank, const HMat2d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::ComputeLocalWidth");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int localWidth;
    int xSize = H.xSizeSource_;
    int ySize = H.ySizeSource_;
    ComputeLocalDimensionRecursion( localWidth, xSize, ySize, p, rank );
    return localWidth;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::ComputeFirstLocalRow
( int p, int rank, const HMat2d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::ComputeFirstLocalRow");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, H.xSizeTarget_, H.ySizeTarget_, p, rank );
    return firstLocalRow;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::ComputeFirstLocalCol
( int p, int rank, const HMat2d<Scalar>& H )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::ComputeFirstLocalCol");
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
#endif
    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, H.xSizeSource_, H.ySizeSource_, p, rank );
    return firstLocalCol;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::ComputeLocalSizes
( std::vector<int>& localSizes, const HMat2d<Scalar>& H )
{
#ifndef RELEASE    
    CallStackEntry entry("DistHMat2d::ComputeLocalSizes");
    const int p = localSizes.size();
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");
    if( (H.xSizeSource_ != H.xSizeTarget_) || 
        (H.ySizeSource_ != H.ySizeTarget_) )
        throw std::logic_error("Routine meant for square nodes");
#endif
    ComputeLocalSizesRecursion
    ( &localSizes[0], localSizes.size(), H.xSizeSource_, H.ySizeSource_ );
}

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar>
void
DistHMat2d<Scalar>::PackedSizesRecursion
( std::vector<std::size_t>& packedSizes, 
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const HMat2d<Scalar>& H )
{
    typedef HMat2d<Scalar> HMat;

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
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes, sourceRank, targetRank, 1,
                      block.data.N->Child(t,s) );
        }
        else if( teamSize == 2 )
        {
            // Give the upper-left 2x2 to the first halves of the teams 
            // and the lower-right 2x2 to the second halves.
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    PackedSizesRecursion
                    ( packedSizes, localSizes,
                      sourceRankOffset+s/2, targetRankOffset+t/2, 1,
                      block.data.N->Child(t,s) );
        }
        else // team Size >= 4
        {
            // Give each diagonal block of the 4x4 partition to a different
            // quarter of the teams
            const int newTeamSize = teamSize/4;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
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
DistHMat2d<Scalar>::PackRecursion
( std::vector<byte**>& headPointers,
  const std::vector<int>& localSizes,
  int sourceRankOffset, int targetRankOffset, int teamSize,
  const HMat2d<Scalar>& H )
{
    typedef HMat2d<Scalar> HMat;

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
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        PackRecursion
                        ( headPointers, localSizes, sourceRank, targetRank, 1,
                          node.Child(t,s) );
            }
            else
            {
                Write( headPointers[sourceRank], SPLIT_NODE );
                Write( headPointers[targetRank], SPLIT_NODE );
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
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
            for( int t=0; t<2; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset, newTeamSize,
                      node.Child(t,s) );
            // Top-right block
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, targetRankOffset, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset, targetRankOffset+newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=2; t<4; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+newTeamSize, 
                      targetRankOffset+newTeamSize,
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
            const int newTeamSize = teamSize/4;
            // Top-left block
            for( int t=0; t<2; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Top-right block
            for( int t=0; t<2; ++t )
                for( int s=2; s<4; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize,
                      newTeamSize, node.Child(t,s) );
            // Bottom-left block
            for( int t=2; t<4; ++t )
                for( int s=0; s<2; ++s )
                    PackRecursion
                    ( headPointers, localSizes,
                      sourceRankOffset+s*newTeamSize, 
                      targetRankOffset+t*newTeamSize, 
                      newTeamSize, node.Child(t,s) );
            // Bottom-right block
            for( int t=2; t<4; ++t )
                for( int s=2; s<4; ++s )
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
DistHMat2d<Scalar>::ComputeLocalDimensionRecursion
( int& localDim, int& xSize, int& ySize, int p, int rank )
{
    if( p >= 4 )
    {
        const int subteam = rank/(p/4);
        const int subteamRank = rank-subteam*(p/4);
        const bool onRight = (subteam & 1);
        const bool onTop = (subteam/2);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        xSize = ( onRight ? xRightSize : xLeftSize   );
        ySize = ( onTop   ? yTopSize   : yBottomSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, p/4, subteamRank );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        xSize = xSize;
        ySize = ( subteam ? yTopSize : yBottomSize );
        ComputeLocalDimensionRecursion
        ( localDim, xSize, ySize, p/2, subteamRank );
    }
    else // p == 1
        localDim = xSize*ySize;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::ComputeFirstLocalIndexRecursion
( int& firstLocalIndex, int xSize, int ySize, int p, int rank )
{
    if( p >= 4 )
    {
        const int subteam = rank/(p/4);
        const int subteamRank = rank-subteam*(p/4);
        const bool onRight = (subteam & 1);
        const bool onTop = (subteam/2);

        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        // Add on this level of offsets
        if( onRight && onTop )
            firstLocalIndex += (xSize*yBottomSize + xLeftSize*yTopSize);
        else if( onTop )
            firstLocalIndex += xSize*yBottomSize;
        else if( onRight )
            firstLocalIndex += xLeftSize*yBottomSize;

        const int xSizeNew = ( onRight ? xRightSize : xLeftSize   );
        const int ySizeNew = ( onTop   ? yTopSize   : yBottomSize );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSizeNew, ySizeNew, p/4, subteamRank );
    }
    else if( p == 2 )
    {
        const int subteam = rank/(p/2);
        const int subteamRank = rank-subteam*(p/2);

        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;

        // Add on this level of offsets
        if( subteam )
            firstLocalIndex += xSize*yBottomSize;

        const int ySizeNew = ( subteam ? yTopSize : yBottomSize );
        ComputeFirstLocalIndexRecursion
        ( firstLocalIndex, xSize, ySizeNew, p/2, subteamRank );
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::ComputeLocalSizesRecursion
( int* localSizes, int teamSize, int xSize, int ySize ) 
{
    if( teamSize >=4 )
    {
        const int newTeamSize = teamSize/4;
        const int xLeftSize = xSize/2;
        const int xRightSize = xSize - xLeftSize;
        const int yBottomSize = ySize/2;
        const int yTopSize = ySize - yBottomSize;
        // Bottom-left piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( localSizes, newTeamSize, xLeftSize, yBottomSize );
        // Bottom-right piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[newTeamSize], newTeamSize, xRightSize, yBottomSize );
        // Top-left piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[2*newTeamSize], newTeamSize, xLeftSize, yTopSize );
        // Top-right piece of quasi2d domain
        ComputeLocalSizesRecursion
        ( &localSizes[3*newTeamSize], newTeamSize, xRightSize, yTopSize );
    }
    else if( teamSize == 2 )
    {
        // Bottom piece of quasi2d domain
        ComputeLocalSizesRecursion( localSizes, 1, xSize, ySize/2 );
        // Top piece of quasi2d domain
        ComputeLocalSizesRecursion( &localSizes[1], 1, xSize, ySize-ySize/2 );
    }
    else
        localSizes[0] = xSize*ySize;
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d
( const byte* packedSub, const Teams& teams )
: haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::DistHMat2d");
#endif
    Unpack( packedSub, teams );
}

template<typename Scalar>
std::size_t
DistHMat2d<Scalar>::Unpack
( const byte* packedDistHMat, const Teams& teams )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Unpack");
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
    xSource_            = Read<int>( head );
    xTarget_            = Read<int>( head );
    ySource_            = Read<int>( head );
    yTarget_            = Read<int>( head );

    UnpackRecursion( head );
    return (head-packedDistHMat);
}

template<typename Scalar>
void
DistHMat2d<Scalar>::UnpackRecursion( const byte*& head )
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
        if( teamSize >= 4 )
        {
            const int subteam = teamRank/(teamSize/4);
            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_+1,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_+1,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
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
            const bool inUpperTeam = ( teamRank == 1 );
            const bool inLeftSourceTeam = ( !inUpperTeam && inSourceTeam_ );
            const bool inRightSourceTeam = ( inUpperTeam && inSourceTeam_ );
            const bool inTopTargetTeam = ( !inUpperTeam && inTargetTeam_ );
            const bool inBottomTargetTeam = ( inUpperTeam && inTargetTeam_ );

            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_,
                          *teams_, level_+1,
                          inLeftSourceTeam, inTopTargetTeam,
                          sourceRoot_, targetRoot_ );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_,
                          *teams_, level_+1,
                          inRightSourceTeam, inTopTargetTeam,
                          sourceRoot_+1, targetRoot_ );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_+1,
                          *teams_, level_+1,
                          inLeftSourceTeam, inBottomTargetTeam,
                          sourceRoot_, targetRoot_+1 );
                    node.Child(t,s).UnpackRecursion( head );
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1]; 
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1]; 
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_+1,
                          *teams_, level_+1,
                          inRightSourceTeam, inBottomTargetTeam,
                          sourceRoot_+1, targetRoot_+1 );
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

        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                node.children[s+4*t] = 
                    new DistHMat2d<Scalar>
                    ( numLevels_-1, maxRank_, stronglyAdmissible_,
                      sourceOffset_+sOffset, targetOffset_+tOffset,
                      node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                      node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                      2*xSource_+(s&1), 2*xTarget_+(t&1),
                      2*ySource_+(s/2), 2*yTarget_+(t/2),
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
            for( int j=0; j<DF.rank; ++j )
                Read( DF.VLocal.Buffer(0,j), head, localWidth );
        }
        if( inTargetTeam_ )
        {
            // Read in U 
            const int localHeight = LocalHeight();
            DF.ULocal.SetType( GENERAL );
            DF.ULocal.Resize( localHeight, DF.rank );
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
            for( int j=0; j<SF.rank; ++j )
                Read( SF.D.Buffer(0,j), head, n );
        }
        else
        {
            // Read in U
            SF.D.Resize( m, SF.rank );
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
        F.U.SetType( GENERAL ); F.U.Resize( m, r );
        for( int j=0; j<r; ++j )
            Read( F.U.Buffer(0,j), head, m );

        // Read in V
        F.V.SetType( GENERAL ); F.V.Resize( n, r );
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
