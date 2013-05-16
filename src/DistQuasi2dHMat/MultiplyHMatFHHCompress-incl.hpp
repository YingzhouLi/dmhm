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
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompress
( const DistQuasi2dHMat<Scalar>& B,
        DistQuasi2dHMat<Scalar>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompress");
#endif
    const DistQuasi2dHMat<Scalar>& A = *this;
    Real error = lapack::MachineEpsilon<Real>();
    
    // RYAN: Again...please properly format and don't check in debug code
MPI_Comm team = _teams->Team( _level );
const int teamRank = mpi::CommRank( team );
int print;
if(teamRank==0)
    print=0;
else
    print=0;

if(print)
    std::cout << teamRank << " Sum" << std::endl;
    C.MultiplyHMatFHHCompressSum
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Precompute" << std::endl;
    MultiplyHMatFHHCompressPrecompute
    ( B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );

if(print)
    std::cout << teamRank << " Reduces" << std::endl;
    C.MultiplyHMatFHHCompressReduces
    ( startLevel, endLevel );
    
if(print)
    std::cout << teamRank << " Midcompute" << std::endl;
    C.MultiplyHMatFHHCompressMidcompute
    ( error, startLevel, endLevel );

if(print)
    std::cout << teamRank << " Broadcasts" << std::endl;
    C.MultiplyHMatFHHCompressBroadcasts
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Postcompute" << std::endl;
    C.MultiplyHMatFHHCompressPostcompute
    ( startLevel, endLevel );

if(print)
    std::cout << teamRank << " Cleanup" << std::endl;
    C.MultiplyHMatFHHCompressCleanup
    ( startLevel, endLevel );

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressSum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressSum");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressSum
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( _level < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        const char option = 'T';
        int totalrank=_colXMap.FirstWidth();
        
        if( _inTargetTeam && totalrank > 0 && LH > 0 )
        {
            _colU.Resize( LH, totalrank, LH );
            hmat_tools::Scale( Scalar(0), _colU );

            int numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                hmat_tools::Add
                ( Scalar(1), _colU, Scalar(1), U, _colU );
            }
        }

        totalrank=_rowXMap.FirstWidth();
        if( _inSourceTeam && totalrank > 0 && LW > 0 )
        {
            _rowU.Resize( LW, totalrank, LW );
            hmat_tools::Scale( Scalar(0), _rowU );

            int numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& U = *_rowXMap.CurrentEntry();
                hmat_tools::Add
                ( Scalar(1), _rowU, Scalar(1), U, _rowU );
            }
        }
        break;
    }
    default:
        break;
    }
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
{
_USqr.Print("_USqr******************************************");
_VSqr.Print("_VSqr******************************************");
}*/

#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressPrecompute
( const DistQuasi2dHMat<Scalar>& B,
        DistQuasi2dHMat<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressPrecompute");
#endif
    const DistQuasi2dHMat<Scalar>& A = *this;
    if( (!A._inTargetTeam && !A._inSourceTeam && !B._inSourceTeam) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }
    const int rank = SampleRank( C.MaxRank() );
    switch( A._block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        switch( B._block.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            if( C.Admissible() )
            {
                if( C._level >= startLevel && C._level < endLevel &&
                    update >= startUpdate && update < endUpdate )
                {
                    const unsigned teamLevel = C._teams->TeamLevel(C._level);
                    if( C._inTargetTeam ) 
                    {
                        int Trank = C._colU.Width();
                        int Omegarank = A._rowOmega.Width();

                        //_colUSqr = ( A B Omega1)' ( A B Omega1 )
                        C._colUSqr.Resize( Trank, Trank, Trank );
                        C._colUSqrEig.resize( Trank );
                        C._BL.Resize(Trank, Trank, Trank);
                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, A.LocalHeight(),
                         Scalar(1), C._colU.LockedBuffer(), C._colU.LDim(),
                                    C._colU.LockedBuffer(), C._colU.LDim(),
                         Scalar(0), C._colUSqr.Buffer(), C._colUSqr.LDim() );

                        //_colPinv = Omega2' (A B Omega1)
                        C._colPinv.Resize( Omegarank, Trank, Omegarank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Omegarank, A.LocalHeight(),
                          Scalar(1), C._colU.LockedBuffer(), C._colU.LDim(),
                                     A._rowOmega.LockedBuffer(), A._rowOmega.LDim(),
                          Scalar(0), C._colPinv.Buffer(),      Omegarank );

                    }
                    if( C._inSourceTeam )
                    {
                        int Trank = C._rowU.Width();
                        int Omegarank = B._colOmega.Width();

                        //_rowUSqr = ( B' A' Omega2 )' ( B' A' Omega2 )
                        C._rowUSqr.Resize( Trank, Trank, Trank );
                        C._rowUSqrEig.resize( Trank );
                        C._BR.Resize(Trank, Omegarank, Trank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Trank, B.LocalWidth(),
                         Scalar(1), C._rowU.LockedBuffer(), C._rowU.LDim(),
                                    C._rowU.LockedBuffer(), C._rowU.LDim(),
                         Scalar(0), C._rowUSqr.Buffer(), C._rowUSqr.LDim() );

                        //_rowPinv = (B' A' Omega2)' Omega1
                        C._rowPinv.Resize( Trank, Omegarank, Trank );

                        blas::Gemm
                        ( 'C', 'N', Trank, Omegarank, B.LocalWidth(),
                          Scalar(1), C._rowU.LockedBuffer(), C._rowU.LDim(),
                                     B._colOmega.LockedBuffer(), B._colOmega.LDim(),
                          Scalar(0), C._rowPinv.Buffer(),      Trank );
                    }
                }
            }
            else if( C._level+1 < endLevel )
            {
                Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).
                            MultiplyHMatFHHCompressPrecompute
                            ( nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressReduces
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressReduces");
#endif

    const int numLevels = _teams->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces );
    std::memset( &sizes[0], 0, numReduces*sizeof(int) );
    MultiplyHMatFHHCompressReducesCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numReduces; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0, offset=0; i<numReduces; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressReducesPack
    ( buffer, offsetscopy, startLevel, endLevel );
    
    MultiplyHMatFHHCompressTreeReduces( buffer, sizes );
    
    MultiplyHMatFHHCompressReducesUnpack
    ( buffer, offsets, startLevel, endLevel );
    
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressReducesCount
( std::vector<int>& sizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressReduceCount");
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
        if( _level+1 < endLevel )
        if( _level >= startLevel && _level < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        sizes[_level] += _colUSqr.Height()*_colUSqr.Width();
        sizes[_level] += _colPinv.Height()*_colPinv.Width();
        sizes[_level] += _rowUSqr.Height()*_rowUSqr.Width();
        sizes[_level] += _rowPinv.Height()*_rowPinv.Width();
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
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressReducesPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressReducePack");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        int Size=_colUSqr.Height()*_colUSqr.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _colUSqr.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }

        Size=_colPinv.Height()*_colPinv.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _colPinv.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }

        Size=_rowUSqr.Height()*_rowUSqr.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _rowUSqr.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }

        Size=_rowPinv.Height()*_rowPinv.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _rowPinv.LockedBuffer(),
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

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressTreeReduces
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressTreeReduces");
#endif
    _teams-> TreeSumToRoots( buffer, sizes );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressReducesUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressReducesUnpack");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s)
                    node.Child(t,s).MultiplyHMatFHHCompressReducesUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int Size=_colUSqr.Height()*_colUSqr.Width();                
            if( Size > 0 )
            {
                std::memcpy
                ( _colUSqr.Buffer(), &buffer[offsets[_level]],
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }

            Size=_colPinv.Height()*_colPinv.Width();                
            if( Size > 0 )
            {
                std::memcpy
                ( _colPinv.Buffer(), &buffer[offsets[_level]],
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }

            Size=_rowUSqr.Height()*_rowUSqr.Width();
            if( Size > 0 )
            {
                std::memcpy
                ( _rowUSqr.Buffer(), &buffer[offsets[_level]],
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }

            Size=_rowPinv.Height()*_rowPinv.Width();                
            if( Size > 0 )
            {
                std::memcpy
                ( _rowPinv.Buffer(), &buffer[offsets[_level]],
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

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressMidcompute
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressMidcompute");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;                                 
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
        if( _level < startLevel )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            // Calculate Eigenvalues of Squared Matrix               
            int Sizemax = std::max(_colUSqr.Height(), _rowUSqr.Height());
             
            int lwork, lrwork, liwork;
            lwork=lapack::EVDWorkSize( Sizemax );
            lrwork=lapack::EVDRealWorkSize( Sizemax );
            liwork=lapack::EVDIntWorkSize( Sizemax );
                    
            std::vector<Scalar> evdWork(lwork);
            std::vector<Real> evdRealWork(lrwork);
            std::vector<int> evdIntWork(liwork);
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_VSqr.Print("_VSqr Before svd");*/
                                                                     
            if( _colUSqr.Height() > 0 )
            {
                lapack::EVD
                ('V', 'U', _colUSqr.Height(), 
                           _colUSqr.Buffer(), _colUSqr.LDim(),
                           &_colUSqrEig[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );

                //colOmegaT = T1' Omega2
                Dense<Scalar> colOmegaT;
                colOmegaT.Resize(_colPinv.Height(), _colPinv.Width(), _colPinv.LDim());
                std::memcpy
                ( colOmegaT.Buffer(), _colPinv.LockedBuffer(),
                  _colPinv.Height()*_colPinv.Width()*sizeof(Scalar) );

                Real Eigmax;
                if( _colUSqrEig[_colUSqrEig.size()-1]>0 )
                    Eigmax=sqrt( _colUSqrEig[_colUSqrEig.size()-1] );
                else
                    Eigmax=0;

                for(int j=0; j<_colUSqrEig.size(); j++)
                    if( _colUSqrEig[j] > error*Eigmax*_colUSqrEig.size() )
                    {
                        Real sqrteig=sqrt(_colUSqrEig[j]);
                        for(int i=0; i<_colUSqr.Height(); ++i)
                            _colUSqr.Set(i,j,_colUSqr.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<_colUSqr.Height(); ++i)
                            _colUSqr.Set(i,j,Scalar(0));
                    }
                
                blas::Gemm
                ( 'C', 'N', _colUSqr.Height(), _colPinv.Width(), _colUSqr.Width(),
                 Scalar(1), _colUSqr.LockedBuffer(), _colUSqr.LDim(),
                            _colPinv.LockedBuffer(), _colPinv.LDim(),
                 Scalar(0), _colPinv.Buffer(), _colPinv.LDim() );

                int rank=std::min(_colPinv.Height(), _colPinv.Width());
                std::vector<Real> singularValues(rank);
                std::vector<Scalar> U(rank*_colPinv.Height()), VH(rank*_colPinv.Width());
                int lsvdwork=lapack::SVDWorkSize(_colPinv.Height(), _colPinv.Width());
                int lrsvdwork=lapack::SVDRealWorkSize(_colPinv.Height(), _colPinv.Width());
                std::vector<Scalar> svdWork(lsvdwork);
                std::vector<Real> svdRealWork(lrsvdwork);
                lapack::AdjointPseudoInverse
                ( _colPinv.Height(), _colPinv.Width(), _colPinv.Buffer(), _colPinv.LDim(),
                  &singularValues[0], &U[0], _colPinv.Height(), &VH[0], rank,
                  &svdWork[0], lsvdwork, &svdRealWork[0]);

                blas::Gemm
                ( 'N', 'N', _colUSqr.Height(), _colPinv.Width(), _colUSqr.Width(),
                 Scalar(1), _colUSqr.LockedBuffer(), _colUSqr.LDim(),
                            _colPinv.LockedBuffer(), _colPinv.LDim(),
                 Scalar(0), _colPinv.Buffer(), _colPinv.LDim() );
                
                blas::Gemm
                ( 'N', 'C', _colPinv.Height(), colOmegaT.Height(), _colPinv.Width(),
                 Scalar(1), _colPinv.LockedBuffer(), _colPinv.LDim(),
                            colOmegaT.LockedBuffer(), colOmegaT.LDim(),
                 Scalar(0), _BL.Buffer(), _BL.LDim() );

            }
                                                                     
            if( _rowUSqr.Height() > 0 )
            {
                lapack::EVD
                ('V', 'U', _rowUSqr.Height(), 
                           _rowUSqr.Buffer(), _rowUSqr.LDim(),
                           &_rowUSqrEig[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );

                Real Eigmax;
                if( _rowUSqrEig[_rowUSqrEig.size()-1]>0 )
                    Eigmax=sqrt( _rowUSqrEig[_rowUSqrEig.size()-1] );
                else
                    Eigmax=0;

                for(int j=0; j<_rowUSqrEig.size(); j++)
                    if( _rowUSqrEig[j] > error*Eigmax*_rowUSqrEig.size() )
                    {
                        Real sqrteig=sqrt(_rowUSqrEig[j]);
                        for(int i=0; i<_rowUSqr.Height(); ++i)
                            _rowUSqr.Set(i,j,_rowUSqr.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<_rowUSqr.Height(); ++i)
                            _rowUSqr.Set(i,j,Scalar(0));
                    }
                
                blas::Gemm
                ( 'C', 'N', _rowUSqr.Height(), _rowPinv.Width(), _rowUSqr.Width(),
                 Scalar(1), _rowUSqr.LockedBuffer(), _rowUSqr.LDim(),
                            _rowPinv.LockedBuffer(), _rowPinv.LDim(),
                 Scalar(0), _rowPinv.Buffer(), _rowPinv.LDim() );

                int rank=std::min(_rowPinv.Height(), _rowPinv.Width());
                std::vector<Real> singularValues(rank);
                std::vector<Scalar> U(rank*_rowPinv.Height()), VH(rank*_rowPinv.Width());
                int lsvdwork=lapack::SVDWorkSize(_rowPinv.Height(), _rowPinv.Width());
                int lrsvdwork=lapack::SVDRealWorkSize(_rowPinv.Height(), _rowPinv.Width());
                std::vector<Scalar> svdWork(lsvdwork);
                std::vector<Real> svdRealWork(lrsvdwork);
                lapack::AdjointPseudoInverse
                ( _rowPinv.Height(), _rowPinv.Width(), _rowPinv.Buffer(), _rowPinv.LDim(),
                  &singularValues[0], &U[0], _rowPinv.Height(), &VH[0], rank,
                  &svdWork[0], lsvdwork, &svdRealWork[0]);

                blas::Gemm
                ( 'N', 'N', _rowUSqr.Height(), _rowPinv.Width(), _rowUSqr.Width(),
                 Scalar(1), _rowUSqr.LockedBuffer(), _rowUSqr.LDim(),
                            _rowPinv.LockedBuffer(), _rowPinv.LDim(),
                 Scalar(0), _BR.Buffer(), _BR.LDim() );
                
            }
/*//Print
if(_level==3 && _block.type==LOW_RANK)
std::cout << "MaxEig before pass: " << *std::max_element( _USqrEig.begin(), _USqrEig.end()) << " " << *std::max_element( _VSqrEig.begin(), _VSqrEig.end()) << std::endl;
*/
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
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressBroadcasts
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressBroadcasts");
#endif
    const int numLevels = _teams->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts );
    std::memset( &sizes[0], 0, numBroadcasts*sizeof(int) );
    MultiplyHMatFHHCompressBroadcastsCount( sizes, startLevel, endLevel );

    int totalSize = 0;
    for(int i=0; i<numBroadcasts; ++i)
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0, offset=0; i<numBroadcasts; offset+=sizes[i], ++i )
        offsets[i] = offset;
    std::vector<int> offsetscopy = offsets;
    MultiplyHMatFHHCompressBroadcastsPack
    ( buffer, offsetscopy, startLevel, endLevel );

    MultiplyHMatFHHCompressTreeBroadcasts( buffer, sizes );
    
    MultiplyHMatFHHCompressBroadcastsUnpack
    ( buffer, offsets, startLevel, endLevel );

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressBroadcastsCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressBroadcastsCount");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressBroadcastsCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _BL.Height()>0 )
            sizes[_level]+=_BL.Height()*_BL.Width();
        if( _BR.Height()>0 )
            sizes[_level]+=_BR.Height()*_BR.Width();
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
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressBroadcastsPack");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressBroadcastsPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( _BL.Height() > 0 )
            {
                int Size = _BL.Height()*_BL.Width();
                std::memcpy
                ( &buffer[offsets[_level]], _BL.LockedBuffer(),
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }
            if( _BR.Height() > 0 )
            {
                int Size = _BR.Height()*_BR.Width();
                std::memcpy
                ( &buffer[offsets[_level]], _BR.LockedBuffer(),
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

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressTreeBroadcasts
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressTreeBroadcasts");
#endif
    _teams-> TreeBroadcasts( buffer, sizes );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressBroadcastsUnpack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressBroadcastsUnpack");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _BL.Height() > 0 )                                  
        { 
            int Size = _BL.Height()*_BL.Width();
            std::memcpy
            ( _BL.Buffer(), &buffer[offsets[_level]],
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }
        if( _BR.Height() > 0 )
        {                                                      
            int Size = _BR.Height()*_BR.Width();
            std::memcpy
            ( _BR.Buffer(), &buffer[offsets[_level]],
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

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressPostcompute
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressPostcompute");
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
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressPostcompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inTargetTeam )                                  
        { 
            _colXMap.Clear();
            _colXMap.Set( 0, new Dense<Scalar>( LocalHeight(), _BL.Width()));
            Dense<Scalar> &U = _colXMap.Get( 0 );
            blas::Gemm
            ('N', 'N', _colU.Height(), _BL.Width(), _colU.Width(), 
             Scalar(1), _colU.LockedBuffer(), _colU.LDim(),
                        _BL.LockedBuffer(), _BL.LDim(),
             Scalar(0), U.Buffer(),         U.LDim() );
        }
        if( _inSourceTeam )
        {                                                      
            _rowXMap.Clear();
            _rowXMap.Set( 0, new Dense<Scalar>( LocalWidth(), _BR.Width()));
            Dense<Scalar> &U = _rowXMap.Get( 0 );
            blas::Gemm
            ('N', 'N', _rowU.Height(), _BR.Width(), _rowU.Width(), 
             Scalar(1), _rowU.LockedBuffer(), _rowU.LDim(),
                        _BR.LockedBuffer(), _BR.LDim(),
             Scalar(0), U.Buffer(),         U.LDim() );
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
DistQuasi2dHMat<Scalar>::MultiplyHMatFHHCompressCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFHHCompressCleanup");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatFHHCompressCleanup
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
        if( _level < startLevel )
            break;
        _colU.Resize(0,0,1);
        _rowU.Resize(0,0,1);
        _colPinv.Resize(0,0,1);
        _rowPinv.Resize(0,0,1);
        _colUSqr.Resize(0,0,1);
        _rowUSqr.Resize(0,0,1);
        _colUSqrEig.resize(0);
        _rowUSqrEig.resize(0);
        _BL.Resize(0,0,1);
        _BR.Resize(0,0,1);
    }
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
