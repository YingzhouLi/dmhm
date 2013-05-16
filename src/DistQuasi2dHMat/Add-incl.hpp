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
DistQuasi2dHMat<Scalar>::AddConstantToDiagonal( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AddConstantToDiagonal");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            _block.data.N->Child(t,t).AddConstantToDiagonal( alpha );
        break;
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            SplitDense& SD = *_block.data.SD;
            Scalar* DBuffer = SD.D.Buffer();
            const int m = SD.D.Height();
            const int n = SD.D.Width();
            const int DLDim = SD.D.LDim();
            for( int j=0; j<std::min(m,n); ++j )
                DBuffer[j+j*DLDim] += alpha;
        }
        break;
    case DENSE:
        {
            Scalar* DBuffer = _block.data.D->Buffer();
            const int m = _block.data.D->Height();
            const int n = _block.data.D->Width();
            const int DLDim = _block.data.D->LDim();
            for( int j=0; j<std::min(m,n); ++j )
                DBuffer[j+j*DLDim] += alpha;
        }
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
