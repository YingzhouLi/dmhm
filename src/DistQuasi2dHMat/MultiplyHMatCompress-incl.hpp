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

#include "./Truncation-incl.hpp"

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompress
(  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompress");
#endif


    Real error = (Real)0.000000001;
    //Wrote By Ryan
    // Compress low-rank F matrix into much lower form.
    // Our low-rank matrix is UV', we want to compute eigenvalues and 
    // eigenvectors of U'U and V'V.
    // U'U first stored in _USqr, then we use _USqr to store the orthognal
    // vectors of U'U, and _USqrEig to store the eigenvalues of U'U.
    // Everything about V are same in _VSqr and _VSqrEig.
    
//    MultiplyHMatCompressFCompressless( startLevel, endLevel );
    MPI_Comm team = _teams->Team( _level );
    const int teamRank = mpi::CommRank( team );
    int print;
    if(teamRank == 0)
    {
        print = 1;
        std::cout << startLevel << " " << endLevel << std::endl;
    }
    else
        print = 1;
    if( print )
        std::cout << teamRank << "Precompute" << std::endl;
    MultiplyHMatCompressFPrecompute( startLevel, endLevel);

    if( print )
        std::cout << teamRank << "Reduces" << std::endl;
    MultiplyHMatCompressFReduces( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "EigenDecomp" << std::endl;
    MultiplyHMatCompressFEigenDecomp( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "PassData" << std::endl;
    MultiplyHMatCompressFPassData( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "Midcompute" << std::endl;
    // Compute sigma_1 V1' V2 sigma_2, the middle part of UV'
    // We use B to state the mid part of UV' that is 
    // B = sigma_1 V1' V2 sigma_2.
    // _BSqr = sqrt(_USqrEig) _USqr' _VSqr sqrt(_VSqrEig)
    // Then _BSqr also will be used to store the eigenvectors
    // of B. _BSqrEig stores eigenvalues of B.
    MultiplyHMatCompressFMidcompute( error, startLevel, endLevel );

    if( print )
        std::cout << teamRank << "PassbackNum" << std::endl;
    MultiplyHMatCompressFPassbackNum( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "PassbackData" << std::endl;
    MultiplyHMatCompressFPassbackData( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "Postcompute" << std::endl;
    // Compute USqr*sqrt(USqrEig)^-1 BSqrU BSigma = BL
    // We overwrite the USqr = USqr*sqrt(USqrEig)^-1
    // Also overwrite the BSqrU = BSqrU BSigma
    // Compute VSqr*sqrt(VSqrEig)^-1 BSqrV = BR
    // We overwrite the VSqr = VSqr*sqrt(VSqrEig)^-1
    MultiplyHMatCompressFPostcompute( error, startLevel, endLevel );

//    Real zeroerror = (Real) 0.1;
//    MultiplyHMatCompressFEigenTrunc( zeroerror );

    if( print )
        std::cout << teamRank << "BroadcastsNum" << std::endl;
    MultiplyHMatCompressFBroadcastsNum( startLevel, endLevel );

    if( print )
        std::cout << teamRank << "Broadcasts" << std::endl;
    MultiplyHMatCompressFBroadcasts( startLevel, endLevel );
    
    if( print )
        std::cout << teamRank << "Finalcompute" << std::endl;
    // Compute the final U and V store in the usual space.
    MultiplyHMatCompressFFinalcompute( startLevel, endLevel );
    
    if( print )
        std::cout << teamRank << "Cleanup" << std::endl;
    // Clean up all the space used in this file
    // Also, clean up the _colXMap, _rowXMap, _UMap, _VMap, _ZMap
    //MultiplyHMatCompressFCleanup( startLevel, endLevel );

   // throw std::logic_error("This routine is in a state of flux.");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPrecompute
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPrecompute");
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
                    node.Child(t,s).MultiplyHMatCompressFPrecompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        DistLowRank &DF = *_block.data.DF;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = (Conjugated ? 'C' : 'T' );
        int totalrank=_colXMap.TotalWidth() + _UMap.TotalWidth() + DF.ULocal.Width();
        
        if( _inTargetTeam && totalrank > 0 && LH > 0 )
        {
            _USqr.Resize( totalrank, totalrank, totalrank );
            _USqrEig.resize( totalrank );
            _Utmp.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( _Utmp.Buffer(0,offset), DF.ULocal.LockedBuffer(),
              LH*DF.ULocal.Width()*sizeof(Scalar) );
            offset += DF.ULocal.Width();

            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LH,
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _Utmp.LockedBuffer(), _Utmp.LDim(),
             (Scalar)0, _USqr.Buffer(),       _USqr.LDim() );
        }

        totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth() + DF.VLocal.Width();
        offset = 0;
        if( _inSourceTeam && totalrank > 0 && LW > 0 )
        {
            _VSqr.Resize( totalrank, totalrank, totalrank );
            _VSqrEig.resize( totalrank );
            _Vtmp.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( _Vtmp.Buffer(0,offset), DF.VLocal.LockedBuffer(),
              LW*DF.VLocal.Width()*sizeof(Scalar) );
            offset += DF.VLocal.Width();

            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LW,
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _Vtmp.LockedBuffer(), _Vtmp.LDim(),
             (Scalar)0, _VSqr.Buffer(),       _VSqr.LDim() );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        SplitLowRank &SF = *_block.data.SF;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = (Conjugated ? 'C' : 'T' );
        int totalrank = _colXMap.TotalWidth() + _UMap.TotalWidth() + SF.D.Width();
        
        if( _inTargetTeam && totalrank > 0 && LH > 0 )
        {
            _USqr.Resize( totalrank, totalrank, totalrank );
            _USqrEig.resize( totalrank );
            _Utmp.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( _Utmp.Buffer(0,offset), SF.D.LockedBuffer(),
              LH*SF.D.Width()*sizeof(Scalar) );
            offset += SF.D.Width();

            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LH,
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _Utmp.LockedBuffer(), _Utmp.LDim(),
             (Scalar)0, _USqr.Buffer(),       _USqr.LDim() );
        }

        totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth() + SF.D.Width();
        offset = 0;
        if( _inSourceTeam && totalrank > 0 && LW > 0 )
        {
            _VSqr.Resize( totalrank, totalrank, totalrank );
            _VSqrEig.resize( totalrank );
            _Vtmp.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( _Vtmp.Buffer(0,offset), SF.D.LockedBuffer(),
              LW*SF.D.Width()*sizeof(Scalar) );
            offset += SF.D.Width();

            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LW,
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _Vtmp.LockedBuffer(), _Vtmp.LDim(),
             (Scalar)0, _VSqr.Buffer(),       _VSqr.LDim() );
        }
        break;
    }
    case LOW_RANK:
    {
        if( _level < startLevel )
            break;
        LowRank<Scalar,Conjugated> &F = *_block.data.F;
        int LH=LocalHeight();
        int LW=LocalWidth();
        int offset=0;
        const char option = (Conjugated ? 'C' : 'T' );
        int totalrank=_colXMap.TotalWidth() + _UMap.TotalWidth() + F.U.Width();
        
        if( totalrank > 0 && LH > 0 )
        {
            _USqr.Resize( totalrank, totalrank, totalrank );
            _USqrEig.resize( totalrank );
            _Utmp.Resize( LH, totalrank, LH );
            
            std::memcpy
            ( _Utmp.Buffer(0,offset), F.U.LockedBuffer(),
              LH*F.U.Width()*sizeof(Scalar) );
            offset=F.U.Width();

            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( _Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LH,
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _Utmp.LockedBuffer(), _Utmp.LDim(),
             (Scalar)0, _USqr.Buffer(),       _USqr.LDim() );
        }

        totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth() + F.V.Width();
        offset = 0;
        if( totalrank > 0 && LW > 0 )
        {
            _VSqr.Resize( totalrank, totalrank, totalrank );
            _VSqrEig.resize( totalrank );
            _Vtmp.Resize( LW, totalrank, LW );
            
            std::memcpy
            ( _Vtmp.Buffer(0,offset), F.V.LockedBuffer(),
              LW*F.V.Width()*sizeof(Scalar) );
            offset=F.V.Width();

            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
                std::memcpy
                ( _Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }

            blas::Gemm
            ('C', 'N', totalrank, totalrank, LW,
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _Vtmp.LockedBuffer(), _Vtmp.LDim(),
             (Scalar)0, _VSqr.Buffer(),       _VSqr.LDim() );
        }
        break;
    }
    default:
        break;
    }
//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_Vtmp.Print("_Vtmp******************************************");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFReduces
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFReduces");
#endif

    const int numLevels = _teams->NumLevels();
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
    
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFReducesCount
( std::vector<int>& sizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFReduceCount");
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
                    node.Child(t,s).MultiplyHMatCompressFReducesCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        sizes[_level] += _USqr.Height()*_USqr.Width();
        sizes[_level] += _USqr.Height()*_USqr.Width();
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFReducesPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFReducePack");
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
                    node.Child(t,s).MultiplyHMatCompressFReducesPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        int Size=_USqr.Height()*_USqr.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _USqr.LockedBuffer(),
              Size*sizeof(Scalar) );
            offsets[_level] += Size;
        }

        Size=_VSqr.Height()*_VSqr.Width();
        if( Size > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_level]], _VSqr.LockedBuffer(),
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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFTreeReduces
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFTreeReduces");
#endif

    _teams-> TreeSumToRoots( buffer, sizes );
    
#ifndef RELEASE
        PopCallStack();
#endif
}



template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFReducesUnpack
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFReducesUnpack");
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
                    node.Child(t,s).MultiplyHMatCompressFReducesUnpack
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
            int Size=_USqr.Height()*_USqr.Width();                
            if( Size > 0 )
            {
                std::memcpy
                ( _USqr.Buffer(), &buffer[offsets[_level]],
                  Size*sizeof(Scalar) );
                offsets[_level] += Size;
            }

            Size=_VSqr.Height()*_VSqr.Width();
            if( Size > 0 )
            {
                std::memcpy
                ( _VSqr.Buffer(), &buffer[offsets[_level]],
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFEigenDecomp
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFEigenDecomp");
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
            int Sizemax = std::max(_USqr.Height(), _VSqr.Height());
             
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
                                                                     
            if( _USqr.Height() > 0 )
            {
                lapack::EVD
                ('V', 'U', _USqr.Height(), 
                           _USqr.Buffer(), _USqr.LDim(),
                           &_USqrEig[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );
            }
                                                                     
            if( _VSqr.Height() > 0 )
            {
                lapack::EVD
                ('V', 'U', _VSqr.Height(), 
                           _VSqr.Buffer(), _VSqr.LDim(),
                           &_VSqrEig[0],
                           &evdWork[0],     lwork,
                           &evdIntWork[0],  liwork,
                           &evdRealWork[0], lrwork );
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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassData
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassData");
#endif
    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyHMatCompressFPassDataCount
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
    MultiplyHMatCompressFPassDataPack
    ( sendBuffer, offsets, startLevel, endLevel );

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
    MultiplyHMatCompressFPassDataUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void 
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassDataCount");
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassDataCount
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inSourceTeam && _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( _inSourceTeam )
                AddToMap( sendSizes, _targetRoot, _VSqr.Height()*(_VSqr.Height()+1) );
            else
                AddToMap( recvSizes, _sourceRoot, _USqr.Height()*(_USqr.Height()+1) );
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassDataPack
( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassDataPack");
#endif
    if( !_inSourceTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassDataPack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && _VSqr.Height() > 0 )
        {
            std::memcpy
            ( &buffer[offsets[_targetRoot]], _VSqr.LockedBuffer(),
              _VSqr.Height()*_VSqr.Width()*sizeof(Scalar) );
            offsets[_targetRoot] += _VSqr.Height()*_VSqr.Width();

            std::memcpy
            ( &buffer[offsets[_targetRoot]], &_VSqrEig[0],
              _VSqrEig.size()*sizeof(Scalar) );
            offsets[_targetRoot] += _VSqrEig.size();
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassDataUnpack
( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassDataUnpack");
#endif
    if( !_inTargetTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatCompressFPassDataUnpack
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inSourceTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 && _USqr.Height() > 0 )
        {
            _VSqr.Resize( _USqr.Height(), _USqr.Width(), _USqr.LDim() );
            _VSqrEig.resize( _USqrEig.size() );
            std::memcpy
            ( _VSqr.Buffer(), &buffer[offsets[_sourceRoot]],
              _VSqr.Height()*_VSqr.Width()*sizeof(Scalar) );
            offsets[_sourceRoot] += _VSqr.Height()*_VSqr.Width();

            std::memcpy
            ( &_VSqrEig[0], &buffer[offsets[_sourceRoot]],
              _VSqrEig.size()*sizeof(Real) );
            offsets[_sourceRoot] += _VSqrEig.size();
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFMidcompute
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFMidcompute");
#endif
    if( !_inTargetTeam || Height() == 0 || Width() == 0 )
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
                    node.Child(t,s).MultiplyHMatCompressFMidcompute
                    ( error, startLevel, endLevel );
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
        if( teamRank == 0 && _USqr.Height() > 0 && _inTargetTeam )
        {
            if( _USqr.Height() != _VSqr.Height() ||
                _USqrEig.size() != _VSqrEig.size() )
            {
#ifndef RELEASE
                throw std::logic_error("Dimension error during calculation");
#endif
            }
            const char option = ( Conjugated ? 'C' : 'T' );

          //  EVDTrunc(_USqr, _USqrEig, error);
          //  EVDTrunc(_VSqr, _VSqrEig, error);

            _BSqr.Resize(_USqr.Width(), _VSqr.Width(), _USqr.Width());

            blas::Gemm
            ( 'C', 'N', _USqr.Width(), _VSqr.Width(), _USqr.LDim(),
              (Scalar)1, _USqr.LockedBuffer(), _USqr.LDim(),
                         _VSqr.LockedBuffer(), _VSqr.LDim(),
              (Scalar)0, _BSqr.Buffer(),       _BSqr.LDim() );
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_VSqr.Print("_VSqr Before svd");*/
//Print
//_VSqr.Print("_VSqr");
//std::cout << "_VSqrEig" << std::endl;
//for( int i=0; i<_VSqrEig.size(); ++i)
//    std::cout << _VSqrEig[i] << "  ";
//std::cout << std::endl;
            std::vector<Real> USqrSigma(_USqrEig.size());
            std::vector<Real> VSqrSigma(_VSqrEig.size());
            for( int i=0; i<_USqrEig.size(); ++i)
                if( _USqrEig[i] > 0 )
                    USqrSigma[i] = sqrt(_USqrEig[i]);
                else
                    USqrSigma[i] = (Real)0;
            for( int i=0; i<_VSqrEig.size(); ++i)
                if( _VSqrEig[i] > 0 )
                    VSqrSigma[i] = sqrt(_VSqrEig[i]);
                else
                    VSqrSigma[i] = (Real)0;

            for( int j=0; j<_BSqr.Width(); ++j)
                for( int i=0; i<_BSqr.Height(); ++i)
                    _BSqr.Set(i,j, _BSqr.Get(i,j)*USqrSigma[i]*VSqrSigma[j]);
            
            int m = _BSqr.Height();
            int n = _BSqr.Width();
            int k = std::min(m, n);

            _BSqrU.Resize(m,k,m);
            _BSqrVH.Resize(k,n,k);
            _BSigma.resize(k);

            int lwork = lapack::SVDWorkSize(m,n);
            int lrwork = lapack::SVDRealWorkSize(m,n);

            std::vector<Scalar> work(lwork);
            std::vector<Real> rwork(lrwork);
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_BSqr.Print("_BSqr Before svd");*/
            lapack::SVD
            ('S', 'S' ,m ,n, 
             _BSqr.Buffer(), _BSqr.LDim(), &_BSigma[0],
             _BSqrU.Buffer(), _BSqrU.LDim(),
             _BSqrVH.Buffer(), _BSqrVH.LDim(),
             &work[0], lwork, &rwork[0] );
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_BSqrVH.Print("_BSqrVH After svd");*/

            //SVDTrunc(_BSqrU, _BSigma, _BSqrVH, error);
//Print
//MPI_Comm teamp = _teams->Team( 0 );
//const int teamRankp = mpi::CommRank( teamp );
//if( _level == 3 && teamRankp == 0 && _block.type==LOW_RANK && _Vtmp.Height() == 2 && _Utmp.Height() == 2)
//std::cout << "Height: " << _Utmp.Height() << " Width: " << _Vtmp.Height() << " B(1,1): " << WrapScalar(_BSqrVH.Get(10,10)) << std::endl;
//_BSqrVH.Print("_SqrVH_Mid");

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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFEigenTrunc
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFEigenTrunc");
#endif
    if( !_inSourceTeam || Height() == 0 || Width() == 0 )
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
                    node.Child(t,s).MultiplyHMatCompressFEigenTrunc
                    ( error, startLevel, endLevel );
        }
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
                ( rowSqr, rowSqrEig, error );
            }
                                                                     
            numEntries = _colSqrMap.Size();
            _colSqrMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colSqrMap.Increment() )
            {
                int key = _colSqrMap.CurrentKey();
                Dense<Scalar>& colSqr = _colSqrMap.Get( key );
                std::vector<Real>& colSqrEig = _colSqrEigMap.Get( key );
                
                EVDTrunc
                ( colSqr, colSqrEig, error );
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackNum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackNum");
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
    MPI_Comm comm = _teams->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<MPI_Request> recvRequests( numRecvs );
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
    MultiplyHMatCompressFPassbackNumUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void 
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackNumCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackNumCount");
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inSourceTeam && _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( _inTargetTeam )
                AddToMap( sendSizes, _sourceRoot, 1 );
            else
                AddToMap( recvSizes, _targetRoot, 1 );
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackNumPack
( std::vector<int>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackNumPack");
#endif
    if( !_inTargetTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inSourceTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            buffer[offsets[_sourceRoot]]=_BSqrVH.Height();
            offsets[_sourceRoot] ++;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackNumUnpack
( const std::vector<int>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackNumUnpack");
#endif
    if( !_inSourceTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            _BSqrVH.Resize
            ( buffer[offsets[_targetRoot]], _VSqr.Height(), 
              buffer[offsets[_targetRoot]] );
            offsets[_targetRoot] ++;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackData
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackData");
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
    MultiplyHMatCompressFPassbackDataUnpack
    ( recvBuffer, recvOffsets, startLevel, endLevel );

    // Don't exit until we know that the data was sent
    for( int i=0; i<numSends; ++i )
        mpi::Wait( sendRequests[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void 
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackDataCount");
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inSourceTeam && _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            if( _inTargetTeam )
                AddToMap( sendSizes, _sourceRoot, _BSqrVH.Height()*_BSqrVH.Width() );
            else
                AddToMap( recvSizes, _targetRoot, _BSqrVH.Height()*_BSqrVH.Width() );
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackDataPack
( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackDataPack");
#endif
    if( !_inTargetTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inSourceTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int size=_BSqrVH.Height()*_BSqrVH.Width();
            std::memcpy
            ( &buffer[offsets[_sourceRoot]], _BSqrVH.LockedBuffer(),
              size*sizeof(Scalar) );
            offsets[_sourceRoot] += size;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPassbackDataUnpack
( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPassbackDataUnpack");
#endif
    if( !_inSourceTeam || Height() == 0 || Width() == 0 )
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
    {
        if( _level+1 < endLevel )
        {
            Node& node = *_block.data.N;
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
        if( _level < startLevel )
            break;
        if( _inTargetTeam )
            break;
        MPI_Comm team = _teams->Team( _level );
        const int teamRank = mpi::CommRank( team );
        if( teamRank ==0 )
        {
            int size=_BSqrVH.Height()*_BSqrVH.Width();
            std::memcpy
            ( _BSqrVH.Buffer(), &buffer[offsets[_targetRoot]],
              size*sizeof(Scalar) );
            offsets[_targetRoot] += size;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFPostcompute
( Real error, int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFPostcompute");
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
                    node.Child(t,s).MultiplyHMatCompressFPostcompute
                    ( error, startLevel, endLevel );
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
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
int print;
if(teamRankp==1)
print=0;
else
print=0;
        if( teamRank == 0 )
        {
            const char option = ( Conjugated ? 'C' : 'T' );
//print
if(print)
std::cout << "Run until here 1" << std::endl;
            if( _inTargetTeam && _USqrEig.size() > 0)
            {
//Print
if(print)
std::cout << _USqrEig.size() << " " << _VSqrEig.size() << " " << _block.type << std::endl;
                Real Eigmax=_USqrEig[_USqrEig.size()-1];
//Print
//if(_level==3 && _block.type==LOW_RANK)
//std::cout << "Eigmax: " << Eigmax << " " << _USqrEig[_USqrEig.size()-2] << std::endl;
                for(int j=0; j<_USqr.Width(); ++j)
                    if(_USqrEig[j] > error*Eigmax && _USqrEig[j] > (Real)0.00000001)
                    {
                        Real sqrteig=sqrt(_USqrEig[j]);
                        for(int i=0; i<_USqr.Height(); ++i)
                            _USqr.Set(i,j,_USqr.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<_USqr.Height(); ++i)
                            _USqr.Set(i,j, (Scalar)0);
                    }

//print
if(print)
std::cout << "Run until here 1.5" << std::endl;
                for(int j=0; j<_BSqrU.Width(); ++j)
                    for(int i=0; i<_BSqrU.Height(); ++i)
                        _BSqrU.Set(i,j,_BSqrU.Get(i,j)*_BSigma[j]);

                _BL.Resize(_USqr.Height(), _BSqrU.Width(), _USqr.Height());

//print
if(print)
std::cout << "Run until here 1.8" << std::endl;
                blas::Gemm
                ( 'N', 'N', _USqr.Height(), _BSqrU.Width(), _USqr.Width(), 
                  (Scalar)1, _USqr.LockedBuffer(),  _USqr.LDim(),
                             _BSqrU.LockedBuffer(), _BSqrU.LDim(),
                  (Scalar)0, _BL.Buffer(), _BL.LDim() );
/*
                _BL.Resize(_USqr.Height(), _BSqrU.Width(), _USqr.Height());
                blas::Gemm
                ( 'N', 'C', _USqr.Height(), _BSqrU.Width(), _USqr.Width(), 
                  (Scalar)1, _USqr.LockedBuffer(),  _USqr.LDim(),
                             _USqr.LockedBuffer(), _USqr.LDim(),
                  (Scalar)0, _BL.Buffer(), _BL.LDim() );
            */
            }

//print
if(print)
std::cout << "Run until here 2, " << _BSqrVH.LDim() << " " << _BSqrVH.Height() << " " << _BSqrVH.Width() << std::endl;
            if(_inSourceTeam && _VSqrEig.size() > 0)
            {
/*//Print
MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type == LOW_RANK)
_BSqrVH.Print("_BSqrVH_Post");*/
                Real Eigmax=_VSqrEig[_VSqrEig.size()-1];
                for(int j=0; j<_VSqr.Width(); ++j)
                    if(_VSqrEig[j] > error*Eigmax && _VSqrEig[j] > (Real) 0.00000001 )
                    {
                        Real sqrteig=sqrt(_VSqrEig[j]);
                        for(int i=0; i<_VSqr.Height(); ++i)
                            _VSqr.Set(i,j,_VSqr.Get(i,j)/sqrteig);
                    }
                    else
                    {
                        for(int i=0; i<_VSqr.Height(); ++i)
                            _VSqr.Set(i,j, (Scalar)0);
                    }

/*MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type==LOW_RANK && _Vtmp.Height() == 2 && _Utmp.Height() == 2)
{
    //_BL.Print("_BL");
    //_BR.Print("_BR");
    Dense<Scalar> BLRT;
    BLRT.Resize(_VSqr.Height(), _VSqr.Height(), _VSqr.Height());
    blas::Gemm
    ( 'C', 'N', _VSqr.Height(), _VSqr.Height(), _VSqr.Width(),
      (Scalar)1, _VSqr.LockedBuffer(), _VSqr.LDim(),
                 _VSqr.LockedBuffer(), _VSqr.LDim(),
      (Scalar)0, BLRT.Buffer(), BLRT.LDim());
    BLRT.Print("BLRT********************************************************");
    _VSqr.Print("_VSqr*****************************************");
}*/

                _BR.Resize(_VSqr.Height(), _BSqrVH.Height(), _VSqr.Height());

                blas::Gemm
                ( 'N', 'C', _VSqr.Height(), _BSqrVH.Height(), _VSqr.Width(),
                  (Scalar)1, _VSqr.LockedBuffer(),  _VSqr.LDim(),
                             _BSqrVH.LockedBuffer(), _BSqrVH.LDim(),
                  (Scalar)0, _BR.Buffer(), _BR.LDim() );
/*
                _BR.Resize(_VSqr.Height(), _BSqrVH.Height(), _VSqr.Height());
                blas::Gemm
                ( 'N', 'C', _VSqr.Height(), _BSqrVH.Height(), _VSqr.Width(),
                  (Scalar)1, _VSqr.LockedBuffer(),  _VSqr.LDim(),
                             _VSqr.LockedBuffer(), _VSqr.LDim(),
                  (Scalar)0, _BR.Buffer(), _BR.LDim() );
//                _BR.Print("_BR");*/
//Print
/*MPI_Comm teamp = _teams->Team( 0 );
const int teamRankp = mpi::CommRank( teamp );
if( _level == 3 && teamRankp == 0 && _block.type==LOW_RANK && _Vtmp.Height() == 2 && _Utmp.Height() == 2)
{
    std::ofstream myfile;
    myfile.open("Check.txt", std::ios::app);
    _BL.Print(myfile,"_BL");
    _BR.Print(myfile,"_BR");
    Dense<Scalar> BLRT;
    BLRT.Resize(_BL.Height(), _BR.Height(), _BL.Height());
    blas::Gemm
    ( 'N', 'C', _BL.Height(), _BR.Height(), _BL.Width(),
      (Scalar)1, _BL.LockedBuffer(), _BL.LDim(),
                 _BR.LockedBuffer(), _BR.LDim(),
      (Scalar)0, BLRT.Buffer(), BLRT.LDim());
    BLRT.Print(myfile,"BLRT********************************************************");
    myfile.close();
}*/
                
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsNum
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsNum");
#endif
    const int numLevels = _teams->NumLevels();
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

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsNumCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsNumCount");
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
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumCount
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
//Print
//if( _level <2)
//_BL.Print("_BL");
    {
        if( _level < startLevel )
            break;
        if( _Utmp.Height()>0 )
            sizes[_level]++;
        if( _Vtmp.Height()>0 )
            sizes[_level]++;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsNumPack
( std::vector<int>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsNumPack");
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
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumPack
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
            if( _BL.Height()>0 )
            {
                buffer[offsets[_level]] = _BL.Width();
                offsets[_level]++;
            }
            if( _BR.Height()>0 )
            {
                buffer[offsets[_level]] = _BR.Width();
                offsets[_level]++;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFTreeBroadcastsNum
( std::vector<int>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFTreeBroadcastsNum");
#endif

    _teams-> TreeBroadcasts( buffer, sizes );
    
#ifndef RELEASE
        PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsNumUnpack
( std::vector<int>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsNumUnpack");
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
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsNumUnpack
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
        if( teamRank != 0 )
        {
            if( _Utmp.Height()>0 )
            {
                _BL.Resize
                (_Utmp.Width(), buffer[offsets[_level]],
                 _Utmp.Width());
                offsets[_level]++;
            }
            if( _Vtmp.Height()>0 )
            {
                _BR.Resize
                (_Vtmp.Width(), buffer[offsets[_level]],
                 _Vtmp.Width());
                offsets[_level]++;
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcasts
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcasts");
#endif
    const int numLevels = _teams->NumLevels();
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

#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsCount
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsCount");
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
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsCount
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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsPack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsPack");
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
                    node.Child(t,s).MultiplyHMatCompressFBroadcastsPack
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


template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFTreeBroadcasts
( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFTreeBroadcasts");
#endif

    _teams-> TreeBroadcasts( buffer, sizes );
    
#ifndef RELEASE
        PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFBroadcastsUnpack
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFBroadcastsUnpack");
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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFFinalcompute
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFFinalcompute");
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
                    node.Child(t,s).MultiplyHMatCompressFFinalcompute
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inTargetTeam )                                  
        { 
            DistLowRank &DF = *_block.data.DF;
            Dense<Scalar> &U = DF.ULocal;
            DF.rank = _BL.Width();
            U.Resize(_Utmp.Height(), _BL.Width(), _Utmp.Height());
            blas::Gemm
            ('N', 'N', _Utmp.Height(), _BL.Width(), _Utmp.Width(), 
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _BL.LockedBuffer(), _BL.LDim(),
             (Scalar)0, U.Buffer(),         U.LDim() );
        }
        if( _inSourceTeam )
        {                                                      
            DistLowRank &DF = *_block.data.DF;
            Dense<Scalar> &V = DF.VLocal;
            DF.rank = _BR.Width();
            V.Resize(_Vtmp.Height(), _BR.Width(), _Vtmp.Height());
            
            blas::Gemm
            ('N', 'N', _Vtmp.Height(), _BR.Width(), _Vtmp.Width(),
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _BR.LockedBuffer(), _BR.LDim(),
             (Scalar)0, V.Buffer(),         V.LDim() );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        if( _inTargetTeam )                                  
        { 
            SplitLowRank &SF = *_block.data.SF;
            Dense<Scalar> &U = SF.D;
            SF.rank = _BL.Width();
            U.Resize(_Utmp.Height(), _BL.Width(), _Utmp.Height());
            
            blas::Gemm
            ('N', 'N', _Utmp.Height(), _BL.Width(), _Utmp.Width(),
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _BL.LockedBuffer(), _BL.LDim(),
             (Scalar)0, U.Buffer(),         U.LDim() );
        }
        if( _inSourceTeam )
        {                                                      
            SplitLowRank &SF = *_block.data.SF;
            Dense<Scalar> &V = SF.D;
            SF.rank = _BR.Width();
            V.Resize(_Vtmp.Height(), _BR.Width(), _Vtmp.Height());
            
            blas::Gemm
            ('N', 'N', _Vtmp.Height(), _BR.Width(), _Vtmp.Width(),
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _BR.LockedBuffer(), _BR.LDim(),
             (Scalar)0, V.Buffer(),         V.LDim() );
        }
        break;
    }
    case LOW_RANK:
    {
        if( _level < startLevel )
            break;
//Print
//MPI_Comm teamp = _teams->Team( 0 );
//const int teamRankp = mpi::CommRank( teamp );
//if( _level == 3 && teamRankp == 0 && _block.type==LOW_RANK && _Vtmp.Height() == 2 && _Utmp.Height() == 2)
//{
//    _BL.Print("_BL");
//    _BR.Print("_BR");
//}

        if( 1 ) 
        { 
            LowRank<Scalar,Conjugated> &F = *_block.data.F;
            Dense<Scalar> &U = F.U;
            U.Resize(_Utmp.Height(), _BL.Width(), _Utmp.Height());
            
            blas::Gemm
            ('N', 'N', _Utmp.Height(), _BL.Width(), _Utmp.Width(),
             (Scalar)1, _Utmp.LockedBuffer(), _Utmp.LDim(),
                        _BL.LockedBuffer(), _BL.LDim(),
             (Scalar)0, U.Buffer(),         U.LDim() );
        }
        if( 1 )
        {                                                      
            LowRank<Scalar,Conjugated> &F = *_block.data.F;
            Dense<Scalar> &V = F.V;
            V.Resize(_Vtmp.Height(), _BR.Width(), _Vtmp.Height());
            
            blas::Gemm
            ('N', 'N', _Vtmp.Height(), _BR.Width(), _Vtmp.Width(),
             (Scalar)1, _Vtmp.LockedBuffer(), _Vtmp.LDim(),
                        _BR.LockedBuffer(), _BR.LDim(),
             (Scalar)0, V.Buffer(),         V.LDim() );
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFCleanup");
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
        if( _level < startLevel )
            break;
        _Utmp.Resize(0,0,1);
        _Vtmp.Resize(0,0,1);
        _USqr.Resize(0,0,1);
        _VSqr.Resize(0,0,1);
        _USqrEig.resize(0);
        _VSqrEig.resize(0);
        _BSqr.Resize(0,0,1);
        _BSqrEig.resize(0);
        _BSqrU.Resize(0,0,1);
        _BSqrVH.Resize(0,0,1);
        _BSigma.resize(0);
        _BL.Resize(0,0,1);
        _BR.Resize(0,0,1);
        _UMap.Clear();
        _VMap.Clear();
        _ZMap.Clear();
        _colXMap.Clear();
        _rowXMap.Clear();
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
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatCompressFCompressless
( int startLevel, int endLevel )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatCompressFCompressless");
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
                    node.Child(t,s).MultiplyHMatCompressFCompressless
                    ( startLevel, endLevel );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( _level < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( _inTargetTeam )                                  
        {
            int offset=0;
            int totalrank=_colXMap.TotalWidth() + _UMap.TotalWidth();
            DistLowRank &DF = *_block.data.DF;
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
            
            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( _inSourceTeam )
        {                                                      
            int offset=0;
            int totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth();
            DistLowRank &DF = *_block.data.DF;
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
            
            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
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
        if( _level < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( _inTargetTeam )                                  
        {
            int offset=0;
            int totalrank=_colXMap.TotalWidth() + _UMap.TotalWidth();
            SplitLowRank &SF = *_block.data.SF;
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
            
            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( _inSourceTeam )
        {                                                      
            int offset=0;
            int totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth();
            SplitLowRank &SF = *_block.data.SF;
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
            
            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
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
        if( _level < startLevel )
            break;
        int LH=LocalHeight();
        int LW=LocalWidth();
        if( _inTargetTeam )                                  
        {
            int offset=0;
            int totalrank=_colXMap.TotalWidth() + _UMap.TotalWidth();
            LowRank<Scalar,Conjugated> &F = *_block.data.F;
            Dense<Scalar> &Utmp = F.U;
//Print
//std::cout << Utmp.Height()*Utmp.Width() << std::endl;
            if( Utmp.Height()*Utmp.Width() != 0)
            {
                offset=Utmp.Width();
                totalrank+=Utmp.Width();
            }
            Utmp.Resize( LH, totalrank, LH );
            
            int numEntries = _UMap.Size();
            _UMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_UMap.Increment() )
            {
                Dense<Scalar>& U = *_UMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
            numEntries = _colXMap.Size();
            _colXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_colXMap.Increment() )
            {
                Dense<Scalar>& U = *_colXMap.CurrentEntry();
                std::memcpy
                ( Utmp.Buffer(0,offset), U.LockedBuffer(),
                  LH*U.Width()*sizeof(Scalar) );
                offset += U.Width();
            }
        }
        if( _inSourceTeam )
        {                                                      
            int offset=0;
            int totalrank=_rowXMap.TotalWidth() + _VMap.TotalWidth();
            LowRank<Scalar,Conjugated> &F = *_block.data.F;
            Dense<Scalar> &Vtmp = F.V;
//Print
//std::cout << Vtmp.Height()*Vtmp.Width() << std::endl;
            if( Vtmp.Height()*Vtmp.Width() != 0)
            {
                offset=Vtmp.Width();
                totalrank+=Vtmp.Width();
            }
            Vtmp.Resize( LW, totalrank, LW );
            
            int numEntries = _VMap.Size();
            _VMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_VMap.Increment() )
            {
                Dense<Scalar>& V = *_VMap.CurrentEntry();
                std::memcpy
                ( Vtmp.Buffer(0,offset), V.LockedBuffer(),
                  LW*V.Width()*sizeof(Scalar) );
                offset += V.Width();
            }
            numEntries = _rowXMap.Size();
            _rowXMap.ResetIterator();
            for( int i=0; i<numEntries; ++i,_rowXMap.Increment() )
            {
                Dense<Scalar>& V = *_rowXMap.CurrentEntry();
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
#ifndef RELEASE
    PopCallStack();
#endif
}
