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
DistHMat3d<Scalar>::SetToRandom()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::SetToRandom");
#endif
    const int maxRank = MaxRank();
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
                block_.data.N->Child(t,s).SetToRandom();
        break;
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *block_.data.DF;
        const int localHeight = DF.ULocal.Height();
        const int localWidth = DF.VLocal.Height();

        DF.rank = maxRank;
        DF.ULocal.Resize( localHeight, maxRank );
        DF.ULocal.Init();
        DF.VLocal.Resize( localWidth, maxRank );
        DF.VLocal.Init();
        ParallelGaussianRandomVectors( DF.ULocal );
        ParallelGaussianRandomVectors( DF.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        const int length = SF.D.Height();

        SF.rank = maxRank;
        SF.D.Resize( length, maxRank );
        SF.D.Init();
        ParallelGaussianRandomVectors( SF.D );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *block_.data.F;
        const int height = F.U.Height();
        const int width = F.V.Height();

        F.U.Resize( height, maxRank );
        F.U.Init();
        F.V.Resize( width, maxRank );
        F.V.Init();
        ParallelGaussianRandomVectors( F.U );
        ParallelGaussianRandomVectors( F.V );
        break;
    }
    case DIST_LOW_RANK_GHOST:
        block_.data.DFG->rank = maxRank;
       break;
    case SPLIT_LOW_RANK_GHOST:
        block_.data.SFG->rank = maxRank;
        break;
    case LOW_RANK_GHOST:
        block_.data.FG->rank = maxRank;
        break;
    case SPLIT_DENSE:
        if( inSourceTeam_ )
            ParallelGaussianRandomVectors( block_.data.SD->D );
        break;
    case DENSE:
        ParallelGaussianRandomVectors( *block_.data.D );
        break;
    default:
        break;
    }
}

} // namespace dmhm
