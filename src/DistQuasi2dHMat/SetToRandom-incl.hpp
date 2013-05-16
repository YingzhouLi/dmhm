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
DistQuasi2dHMat<Scalar>::SetToRandom()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::SetToRandom");
#endif
    const int maxRank = MaxRank();
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).SetToRandom();
        break;
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        const int localHeight = DF.ULocal.Height();
        const int localWidth = DF.VLocal.Height();

        DF.rank = maxRank;
        DF.ULocal.Resize( localHeight, maxRank );
        DF.VLocal.Resize( localWidth, maxRank );
        ParallelGaussianRandomVectors( DF.ULocal );
        ParallelGaussianRandomVectors( DF.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const int length = SF.D.Height();

        SF.rank = maxRank;
        SF.D.Resize( length, maxRank );
        ParallelGaussianRandomVectors( SF.D );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *_block.data.F;
        const int height = F.U.Height();
        const int width = F.V.Height();

        F.U.Resize( height, maxRank );
        F.V.Resize( width, maxRank );
        ParallelGaussianRandomVectors( F.U );
        ParallelGaussianRandomVectors( F.V );
        break;
    }
    case DIST_LOW_RANK_GHOST:
        _block.data.DFG->rank = maxRank;
       break;
    case SPLIT_LOW_RANK_GHOST:
        _block.data.SFG->rank = maxRank;
        break;
    case LOW_RANK_GHOST:
        _block.data.FG->rank = maxRank;
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
            ParallelGaussianRandomVectors( _block.data.SD->D );
        break;
    case DENSE:
        ParallelGaussianRandomVectors( *_block.data.D );
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
