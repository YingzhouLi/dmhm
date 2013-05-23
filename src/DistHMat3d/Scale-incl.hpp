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
DistHMat3d<Scalar>::Scale( Scalar alpha )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::Scale");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
                block_.data.N->Child(t,s).Scale( alpha );
        break;

    case DIST_LOW_RANK:
        if( alpha == Scalar(0) )
        {
            block_.data.DF->rank = 0;
            if( inTargetTeam_ )
            {
                Dense<Scalar>& ULocal = block_.data.DF->ULocal;
                ULocal.Resize( ULocal.Height(), 0, ULocal.Height() );
            }
            if( inSourceTeam_ )
            {
                Dense<Scalar>& VLocal = block_.data.DF->VLocal;
                VLocal.Resize( VLocal.Height(), 0, VLocal.Height() );
            }
        }
        else if( inTargetTeam_ )
            hmat_tools::Scale( alpha, block_.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( alpha == Scalar(0) )
        {
            block_.data.SF->rank = 0;
            Dense<Scalar>& D = block_.data.SF->D;
            D.Resize( D.Height(), 0, D.Height() );
        }
        else if( inTargetTeam_ )
            hmat_tools::Scale( alpha, block_.data.SF->D );
        break;
    case LOW_RANK:
        hmat_tools::Scale( alpha, *block_.data.F );
        break;
    case DIST_LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            block_.data.DFG->rank = 0;
        break;
    case SPLIT_LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            block_.data.SFG->rank = 0;
        break;
    case LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            block_.data.FG->rank = 0;
        break;
    case SPLIT_DENSE:
        if( inSourceTeam_ )
            hmat_tools::Scale( alpha, block_.data.SD->D );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *block_.data.D );
        break;
    default:
        break;
    }
}

} // namespace dmhm
