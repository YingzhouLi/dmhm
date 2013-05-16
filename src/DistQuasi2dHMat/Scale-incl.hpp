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
DistQuasi2dHMat<Scalar>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Scale");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).Scale( alpha );
        break;

    case DIST_LOW_RANK:
        if( alpha == Scalar(0) )
        {
            _block.data.DF->rank = 0;
            if( _inTargetTeam )
            {
                Dense<Scalar>& ULocal = _block.data.DF->ULocal;
                ULocal.Resize( ULocal.Height(), 0, ULocal.Height() );
            }
            if( _inSourceTeam )
            {
                Dense<Scalar>& VLocal = _block.data.DF->VLocal;
                VLocal.Resize( VLocal.Height(), 0, VLocal.Height() );
            }
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( alpha == Scalar(0) )
        {
            _block.data.SF->rank = 0;
            Dense<Scalar>& D = _block.data.SF->D;
            D.Resize( D.Height(), 0, D.Height() );
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.SF->D );
        break;
    case LOW_RANK:
        hmat_tools::Scale( alpha, *_block.data.F );
        break;
    case DIST_LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            _block.data.DFG->rank = 0;
        break;
    case SPLIT_LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            _block.data.SFG->rank = 0;
        break;
    case LOW_RANK_GHOST:
        if( alpha == Scalar(0) )
            _block.data.FG->rank = 0;
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
            hmat_tools::Scale( alpha, _block.data.SD->D );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *_block.data.D );
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
