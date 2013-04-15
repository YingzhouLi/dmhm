/*
   Distributed-Memory Hierarchical Matrices (DMHM): a prototype implementation
   of distributed-memory H-matrix arithmetic. 

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompress()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompress");
#endif

    throw std::logic_error("This routine is in a state of flux.");

    //Wrote By Ryan
    // Convert low-rank HH matrix into the UV' form.
    
    MultiplyHMatCompressHHSetup();

    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyHMatCompressHHReducesCount( sizes );

    int totalSize = 0;
    for(int i=0; i<numReduces; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatCompressHHReducesPack( buffer, offsetscopy );
    
    MultiplyHMatCompressHHTreeReduces( buffer, sizes );
    
    MultiplyHMatCompressHHReducesUnpack( buffer, offsets );

    MultiplyHMatCompressHHEigenDecomp();

    Real zeroerror = (Real) 0.1;
    MultiplyHMatCompressHHEigenTrunc( zeroerror );

    const int numBroadcasts = numLevels-1;
    sizes.resize( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyHMatCompressHHBroadcastsCount( sizes );

    totalSize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalSize += sizes[i];
    buffer.resize( totalSize );
    offsets.resize( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    offsetscopy = offsets;
    MultiplyHMatCompressHHBroadcastsPack( buffer, offsetscopy );

    MultiplyHMatCompressHHTreeBroadcasts( buffer, sizes );
    
    MultiplyHMatCompressHHBroadcastsUnpack( buffer, offsets );
    /*
    MultiplyHMatCompressHHPostcompute();

    // Compress low-rank matrix with only UV' matrix
    MultiplyHMatCompressFSetup();
    MultiplyHMatCompressFPrecompute();
    MultiplyHMatCompressFReducesCount();
    MultiplyHMatCompressFReducesPack();
    MultiplyHMatCompressFTreeReduces();
    MultiplyHMatCompressFReducesUnpack();

    MultiplyHMatCompressFEigenDecomp();

    MultiplyHMatCompressFBroadcastsCount();
    MultiplyHMatCompressFBroadcastsPack();
    MultiplyHMatCompressFBroadcasts();
    MultiplyHMatCompressFBroadcastsUnpack();
    MultiplyHMatCompressFPostcompute();
    */

    /*
    MultiplyHMatCompressRandomSetup( 0 );
    MultiplyHMatCompressRandomPrecompute( 0 );
    */
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHSetup()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHSetup");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHSetup();
        break;
    }
    case DIST_LOW_RANK:
    {
        // Add the F+=HH updates
        const char option = (Conjugated ? 'C' : 'T' );
        int numEntries = _rowXMap.Size();
        _rowXMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
        {
            int key = _rowXMap.CurrentKey();
            Dense<Scalar>& rowX = _rowXMap.Get( key );
            int Size= rowX.Width();
            _rowSqrMap.Set
            ( key, new Dense<Scalar>( Size, Size ) );
            Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );

            blas::Gemm
            (option, 'N', Size, rowX.Height(), Size, 
             (Scalar)1, rowX.LockedBuffer(), rowX.LDim(),
                        rowX.LockedBuffer(), rowX.LDim(),
             (Scalar)0, rowSqr.Buffer(),      rowSqr.LDim() );
        }

        numEntries = _colXMap.Size();
        _colXMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
        {
            int key = _colXMap.CurrentKey();
            Dense<Scalar>& colX = _colXMap.Get( key );
            int Size= colX.Width();
            _colSqrMap.Set
            ( key, new Dense<Scalar>( Size, Size ) );
            Dense<Scalar>& colSqr = _colSqrMap.Get( key );

            blas::Gemm
            (option, 'N', Size, colX.Height(), Size, 
             (Scalar)1, colX.LockedBuffer(), colX.LDim(),
                        colX.LockedBuffer(), colX.LDim(),
             (Scalar)0, colSqr.Buffer(),      colSqr.LDim() );
        }

        break;
    }
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( !_haveDenseUpdate )
        {   
            // Add the F+=HH updates
            const char option = (Conjugated ? 'C' : 'T' );
            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                int key = _rowXMap.CurrentKey();
                Dense<Scalar>& rowX = _rowXMap.Get( key );
                int Size= rowX.Width();
                _rowSqrMap.Set
                ( key, new Dense<Scalar>( Size, Size ) );
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
    
                blas::Gemm
                (option, 'N', Size, rowX.Height(), Size, 
                 (Scalar)1, rowX.LockedBuffer(), rowX.LDim(),
                            rowX.LockedBuffer(), rowX.LDim(),
                 (Scalar)0, rowSqr.Buffer(),      rowSqr.LDim() );
            }
    
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                int key = _colXMap.CurrentKey();
                Dense<Scalar>& colX = _colXMap.Get( key );
                int Size= colX.Width();
                _colSqrMap.Set
                ( key, new Dense<Scalar>( Size, Size ) );
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
    
                blas::Gemm
                (option, 'N', Size, colX.Height(), Size, 
                 (Scalar)1, colX.LockedBuffer(), colX.LDim(),
                            colX.LockedBuffer(), colX.LDim(),
                 (Scalar)0, colSqr.Buffer(),      colSqr.LDim() );
            }
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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHReducesCount
( std::vector<int>& sizes )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHReducesCount");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHReducesCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        // Count the size of square matrix
        int numEntries = _rowSqrMap.Size();
        _rowSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
        {
            Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();
            sizes[_level] += rowSqr.Height()*rowSqr.Width();
        }

        numEntries = _colSqrMap.Size();
        _colSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
        {
            Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();
            sizes[_level] += colSqr.Height()*colSqr.Width();
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


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHReducesPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHReducesPack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHReducesPack( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        // Copy the square matrix to sending buffer
        int numEntries = _rowSqrMap.Size();
        _rowSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
        {
            Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();
            int Size = rowSqr.Height()*rowSqr.Width();
            std::memcpy
            ( &buffer[offsets[_level]], rowSqr.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }

        numEntries = _colSqrMap.Size();
        _colSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
        {
            Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();
            int Size = colSqr.Height()*colSqr.Width();
            std::memcpy
            ( &buffer[offsets[_level]], colSqr.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
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

template<typename Scalar, bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar, Conjugated>::MultiplyHMatCompressHHTreeReduces
( std::vector<Scalar>& buffer, std::vector<int>& sizes )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHTreeReduces");
#endif

    _teams->TreeSumToRoots( buffer, sizes );

#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHReducesUnpack
( std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHReducesUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHReducesUnpack( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();
                int Size = rowSqr.Height()*rowSqr.Width();
                std::memcpy
                ( rowSqr.Buffer(), &buffer[offsets[_level]],
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();
                int Size = colSqr.Height()*colSqr.Width();
                std::memcpy
                ( colSqr.Buffer(), &buffer[offsets[_level]], 
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }
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


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHEigenDecomp()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHEigenDecomp");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHEigenDecomp();
        break;
    }
    case DIST_LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            // Calculate Eigenvalues of Squared Matrix               
            int Sizemax=0;
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                int Size= rowSqr.Width();
                _rowSqrEigMap.Set
                ( key, new std::vector<Real>( Size ) );
                if( Size > Sizemax )
                    Sizemax=Size;
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                int Size= colSqr.Width();
                _colSqrEigMap.Set
                ( key, new std::vector<Real>( Size ) );
                if( Size > Sizemax )
                    Sizemax=Size;
            }
                                                                     
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( Sizemax );
            lrwork=lapack::EVDRealWorkSize( Sizemax );
            liwork=lapack::EVDIntWorkSize( Sizemax );
                                                                     
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
                                                                     
            numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                int Size= rowSqr.Width();
                std::vector<Real>& rowSqrEig = _rowSqrEigMap.Get( key );
                
                lapack::EVD
                ('V', 'U', Size, rowSqr.Buffer(), rowSqr.LDim(),
                                 &rowSqrEig[0],
                                 &evdWork[0],     lwork,
                                 &evdIntWork[0],  liwork,
                                 &evdRealWork[0], lrwork );
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                int Size= colSqr.Width();
                std::vector<Real>& colSqrEig = _colSqrEigMap.Get( key );
                
                lapack::EVD
                ('V', 'U', Size, colSqr.Buffer(), colSqr.LDim(),
                                 &colSqrEig[0],
                                 &evdWork[0],     lwork,
                                 &evdIntWork[0],  liwork,
                                 &evdRealWork[0], lrwork );
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( !_haveDenseUpdate )
        {   
            // Calculate Eigenvalues of Squared Matrix               
            int Sizemax=0;
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                int Size= rowSqr.Width();
                _rowSqrEigMap.Set
                ( key, new std::vector<Real>( Size ) );
                if( Size > Sizemax )
                    Sizemax=Size;
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                int Size= colSqr.Width();
                _colSqrEigMap.Set
                ( key, new std::vector<Real>( Size ) );
                if( Size > Sizemax )
                    Sizemax=Size;
            }
                                                                     
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( Sizemax );
            lrwork=lapack::EVDRealWorkSize( Sizemax );
            liwork=lapack::EVDIntWorkSize( Sizemax );
                                                                     
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
                                                                     
            numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                int Size= rowSqr.Width();
                std::vector<Real>& rowSqrEig = _rowSqrEigMap.Get( key );
                
                lapack::EVD
                ('V', 'U', Size, rowSqr.Buffer(), rowSqr.LDim(),
                                 &rowSqrEig[0],
                                 &evdWork[0],     lwork,
                                 &evdIntWork[0],  liwork,
                                 &evdRealWork[0], lrwork );
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                int Size= colSqr.Width();
                std::vector<Real>& colSqrEig = _colSqrEigMap.Get( key );
                
                lapack::EVD
                ('V', 'U', Size, colSqr.Buffer(), colSqr.LDim(),
                                 &colSqrEig[0],
                                 &evdWork[0],     lwork,
                                 &evdIntWork[0],  liwork,
                                 &evdRealWork[0], lrwork );
            }
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

template<typename Scalar, bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar, Conjugated>::EVDTrunc
( Dense<Scalar>& Q, int ldq, std::vector<Real>& w, Real error )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::EVDTrunc");
    if( ldq ==0 )
        throw std::logic_error("ldq was 0");
#endif

    int L;
    for(L=0; L<ldq; ++L )
        if( w[L]>error)
            break;

    w.erase( w.begin(), w.begin()+L );
    Q.EraseCol(0, L-1);

#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHEigenTrunc
( Real error )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHEigenTrunc");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHEigenTrunc( error );
        break;
    }
    case DIST_LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                std::vector<Real>& rowSqrEig = _rowSqrEigMap.Get( key );
                
                EVDTrunc
                ( rowSqr, rowSqr.LDim(), rowSqrEig, error );
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                std::vector<Real>& colSqrEig = _colSqrEigMap.Get( key );
                
                EVDTrunc
                ( colSqr, colSqr.LDim(), colSqrEig, error );
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( !_haveDenseUpdate )
        {   
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
            {
                int key = _rowSqrMap.CurrentKey();
                Dense<Scalar>& rowSqr = _rowSqrMap.Get( key );
                std::vector<Real>& rowSqrEig = _rowSqrEigMap.Get( key );
                
                EVDTrunc
                ( rowSqr, rowSqr.LDim(), rowSqrEig, error );
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                std::vector<Real>& colSqrEig = _colSqrEigMap.Get( key );
                
                EVDTrunc
                ( colSqr, colSqr.LDim(), colSqrEig, error );
            }
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


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHBroadcastsCount
( std::vector<int>& sizes )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHBroadcastsCount");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        // Count the size of square matrix
        int numEntries = _rowSqrMap.Size();
        _rowSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )
        {
            Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();
            sizes[_level] += rowSqr.Height()*rowSqr.Width();
        }

        numEntries = _colSqrMap.Size();
        _colSqrMap.ResetIterator();
        for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
        {
            Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();
            sizes[_level] += colSqr.Height()*colSqr.Width();
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


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHBroadcastsPack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHBroadcastsPack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            // Copy the square matrix to sending buffer                                       
            int numEntries = _rowSqrMap.Size();
            _rowSqrMap.ResetIterator();
            _rowSqrEigMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,
                _rowSqrMap.Increment(),_rowSqrEigMap.Increment())
            {
                Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();
                std::vector<Real>& rowSqrEig = *_rowSqrEigMap.CurrentEntry();
                int Size = rowSqr.Height()*rowSqr.Width();
                for(int j=0; j<rowSqr.Width(); ++j)
                {
                    Real tmp=sqrt(rowSqrEig[j]);
                    for(int k=0; k<rowSqr.Height(); ++k)
                        buffer[offsets[_level]+k+rowSqr.LDim()*j] = rowSqr.Get(k,j)/tmp;
                }
                offsets[_level] += Size;
            }
                                                                                              
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            _colSqrEigMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,
                _colSqrMap.Increment(),_colSqrEigMap.Increment())
            {
                Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();
                std::vector<Real>& colSqrEig = *_colSqrEigMap.CurrentEntry();
                int Size = colSqr.Height()*colSqr.Width();
                for(int j=0; j<colSqr.Width(); ++j)
                {
                    Real tmp=sqrt(colSqrEig[j]);
                    for(int k=0; k<colSqr.Height(); ++k)
                        buffer[offsets[_level]+k+colSqr.LDim()*j] = colSqr.Get(k,j)/tmp;
                }
                offsets[_level] += Size;
            }
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


template<typename Scalar, bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar, Conjugated>::MultiplyHMatCompressHHTreeBroadcasts
( std::vector<Scalar>& buffer, std::vector<int>& sizes )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHTreeBreadcasts");
#endif

    _teams->TreeBroadcasts( buffer, sizes );

#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressHHBroadcastsUnpack
( std::vector<Scalar>& buffer, std::vector<int>& offsets )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressHHBroadcastsUnpack");
#endif
    if( Height() == 0 || Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    switch( _block.type )
    {
    case DIST_NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyHMatCompressHHBroadcastsUnpack
                ( buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        int numEntries = _rowSqrMap.Size();                       
        _rowSqrMap.ResetIterator();                                  
        for( int i=0; i<numEntries; ++i,_rowSqrMap.Increment() )     
        {                                                            
            Dense<Scalar>& rowSqr = *_rowSqrMap.CurrentEntry();      
            int Size = rowSqr.Height()*rowSqr.Width();               
            std::memcpy                                              
            ( rowSqr.Buffer(), &buffer[offsets[_level]],             
              Size*sizeof(Scalar) );                                 
            offsets[_level] += Size;                                 
        }                                                            
                                                                     
        numEntries = _colSqrMap.Size();                              
        _colSqrMap.ResetIterator();                                  
        for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )     
        {                                                            
            Dense<Scalar>& colSqr = *_colSqrMap.CurrentEntry();      
            int Size = colSqr.Height()*colSqr.Width();               
            std::memcpy                                              
            ( colSqr.Buffer(), &buffer[offsets[_level]],             
              Size*sizeof(Scalar) );                                 
            offsets[_level] += Size;                                 
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



