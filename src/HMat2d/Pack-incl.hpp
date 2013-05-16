/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
HMat2d<Scalar>::HMat2d( const std::vector<byte>& packedHMat )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::HMat2d");
#endif
    Unpack( packedHMat );
}

template<typename Scalar>
std::size_t
HMat2d<Scalar>::PackedSize() const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::PackedSize");
#endif
    std::size_t packedSize = 13*sizeof(int) + 2*sizeof(bool);
    PackedSizeRecursion( packedSize );
    return packedSize;
}

template<typename Scalar>
std::size_t
HMat2d<Scalar>::Pack( byte* packedHMat ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Pack");
#endif
    byte* head = packedHMat;
    
    // Write the header information
    Write( head, numLevels_ );
    Write( head, maxRank_ );
    Write( head, sourceOffset_ );
    Write( head, targetOffset_ );
    Write( head, symmetric_ );
    Write( head, stronglyAdmissible_ );
    Write( head, xSizeSource_ );
    Write( head, xSizeTarget_ );
    Write( head, ySizeSource_ );
    Write( head, ySizeTarget_ );
    Write( head, xSource_ );
    Write( head, xTarget_ );
    Write( head, ySource_ );
    Write( head, yTarget_ );

    PackRecursion( head );
    return head-packedHMat;
}

template<typename Scalar>
std::size_t
HMat2d<Scalar>::Pack( std::vector<byte>& packedHMat ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Pack");
#endif
    // Create the storage and extract the buffer
    const std::size_t packedSize = PackedSize();
    packedHMat.resize( packedSize );
    byte* head = &packedHMat[0];

    // Write the header information
    Write( head, numLevels_ );
    Write( head, maxRank_ );
    Write( head, sourceOffset_ );
    Write( head, targetOffset_ );
    Write( head, symmetric_ );
    Write( head, stronglyAdmissible_ );
    Write( head, xSizeSource_ );
    Write( head, xSizeTarget_ );
    Write( head, ySizeSource_ );
    Write( head, ySizeTarget_ );
    Write( head, xSource_ );
    Write( head, xTarget_ );
    Write( head, ySource_ );
    Write( head, yTarget_ );

    PackRecursion( head );
    return head-&packedHMat[0];
}

template<typename Scalar>
std::size_t
HMat2d<Scalar>::Unpack( const byte* packedHMat )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Unpack");
#endif
    const byte* head = packedHMat;
    
    // Unpack the top-level header information
    numLevels_          = Read<int>( head );
    maxRank_            = Read<int>( head );
    sourceOffset_       = Read<int>( head );
    targetOffset_       = Read<int>( head );
    symmetric_          = Read<bool>( head );
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
    return head-packedHMat;
}

template<typename Scalar>
std::size_t
HMat2d<Scalar>::Unpack( const std::vector<byte>& packedHMat )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Unpack");
#endif
    const byte* head = &packedHMat[0];

    // Unpack the top-level header information
    numLevels_          = Read<int>( head );
    maxRank_            = Read<int>( head );
    sourceOffset_       = Read<int>( head );
    targetOffset_       = Read<int>( head );
    symmetric_          = Read<bool>( head );
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
    return head-&packedHMat[0];
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar>
void
HMat2d<Scalar>::PackedSizeRecursion( std::size_t& packedSize ) const
{
    packedSize += sizeof(BlockType);
    switch( block_.type )
    {
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                block_.data.N->Child(t,s).PackedSizeRecursion( packedSize );
        break;
    case NODE_SYMMETRIC:
        for( int t=0; t<4; ++t )
            for( int s=0; s<=t; ++s )
                block_.data.NS->Child(t,s).PackedSizeRecursion( packedSize );
        break;
    case LOW_RANK:
    {
        const Dense<Scalar>& U = block_.data.F->U;
        const Dense<Scalar>& V = block_.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // The height and width are already known, we just need the rank
        packedSize += sizeof(int);

        // Make space for U and V
        packedSize += (m+n)*r*sizeof(Scalar);

        break;
    }
    case DENSE:
    {
        const Dense<Scalar>& D = *block_.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Make space for the matrix type and data
        packedSize += sizeof(MatrixType);
        if( type == GENERAL )
            packedSize += m*n*sizeof(Scalar);
        else
            packedSize += ((m*m+m)/2)*sizeof(Scalar);
        break;
    }
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::PackRecursion( byte*& head ) const
{
    Write( head, block_.type );
    switch( block_.type )
    {
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                block_.data.N->Child(t,s).PackRecursion( head );
        break;
    case NODE_SYMMETRIC:
        for( int t=0; t<4; ++t )
            for( int s=0; s<=t; ++s )
                block_.data.NS->Child(t,s).PackRecursion( head );
        break;
    case LOW_RANK:
    {
        const Dense<Scalar>& U = block_.data.F->U;
        const Dense<Scalar>& V = block_.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // Write out the rank
        Write( head, r );

        // Write out U
        for( int j=0; j<r; ++j )
            Write( head, U.LockedBuffer(0,j), m );

        // Write out V
        for( int j=0; j<r; ++j )
            Write( head, V.LockedBuffer(0,j), n );

        break;
    }
    case DENSE:
    {
        const Dense<Scalar>& D = *block_.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the matrix type and data
        Write( head, type );
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Write( head, D.LockedBuffer(0,j), m );
        else
            for( int j=0; j<n; ++j )
                Write( head, D.LockedBuffer(j,j), m-j );
        break;
    }
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::UnpackRecursion( const byte*& head )
{
    block_.Clear();
    block_.type = Read<BlockType>( head );
    switch( block_.type )
    {
    case NODE:
    {
        block_.data.N = NewNode();
        Node& node = *block_.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                node.children[s+4*t] = 
                    new HMat2d<Scalar>
                    ( numLevels_-1, maxRank_, symmetric_, stronglyAdmissible_,
                      node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                      node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                      2*xSource_+(s&1), 2*xTarget_+(t&1),
                      2*ySource_+(s/2), 2*yTarget_+(t/2),
                      sOffset+sourceOffset_, tOffset+targetOffset_ );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        block_.data.NS = NewNodeSymmetric();
        NodeSymmetric& node = *block_.data.NS;
        int child = 0;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
        {
            for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
            {
                node.children[child++] =  
                    new HMat2d<Scalar>
                    ( numLevels_-1, maxRank_, symmetric_, stronglyAdmissible_,
                      node.xSizes[s&1], node.xSizes[t&1],
                      node.ySizes[s/2], node.ySizes[t/2],
                      2*xSource_+(s&1), 2*xTarget_+(t&1),
                      2*ySource_+(s/2), 2*yTarget_+(t/2),
                      sOffset+targetOffset_, tOffset+targetOffset_ );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case LOW_RANK:
    {
        block_.data.F = new LowRank<Scalar>;
        Dense<Scalar>& U = block_.data.F->U;
        Dense<Scalar>& V = block_.data.F->V;
        const int m = Height();
        const int n = Width();

        // Read in the matrix rank
        const int r = Read<int>( head );
        U.SetType( GENERAL ); U.Resize( m, r );
        V.SetType( GENERAL ); V.Resize( n, r );

        // Read in U
        for( int j=0; j<r; ++j )
            Read( U.Buffer(0,j), head, m );

        // Read in V
        for( int j=0; j<r; ++j )
            Read( V.Buffer(0,j), head, n );

        break;
    }
    case DENSE:
        block_.data.D = new Dense<Scalar>;
        Dense<Scalar>& D = *block_.data.D;
        const int m = Height();
        const int n = Width();

        const MatrixType type = Read<MatrixType>( head );
        D.SetType( type ); 
        D.Resize( m, n );

        // Read in the matrix
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Read( D.Buffer(0,j), head, m );
        else
            for( int j=0; j<n; ++j )
                Read( D.Buffer(j,j), head, m-j );
        break;
    }
}

} // namespace dmhm
