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
DistHMat3d<Scalar>::AddConstantToDiagonal( Scalar alpha )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::AddConstantToDiagonal");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<8; ++t )
            block_.data.N->Child(t,t).AddConstantToDiagonal( alpha );
        break;
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case SPLIT_DENSE:
        if( inSourceTeam_ )
        {
            SplitDense& SD = *block_.data.SD;
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
            Scalar* DBuffer = block_.data.D->Buffer();
            const int m = block_.data.D->Height();
            const int n = block_.data.D->Width();
            const int DLDim = block_.data.D->LDim();
            for( int j=0; j<std::min(m,n); ++j )
                DBuffer[j+j*DLDim] += alpha;
        }
        break;
    default:
        break;
    }
}

} // namespace dmhm
