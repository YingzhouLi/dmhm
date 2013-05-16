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
DistQuasi2dHMat<Scalar>::CopyFrom( const DistQuasi2dHMat<Scalar>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::CopyFrom");
#endif
    DistQuasi2dHMat<Scalar>& A = *this;

    A._numLevels = B._numLevels;
    A._maxRank = B._maxRank;
    A._targetOffset = B._targetOffset;
    A._sourceOffset = B._sourceOffset;
    A._stronglyAdmissible = B._stronglyAdmissible;

    A._xSizeTarget = B._xSizeTarget;
    A._ySizeTarget = B._ySizeTarget;
    A._xSizeSource = B._xSizeSource;
    A._ySizeSource = B._ySizeSource;
    A._zSize = B._zSize;

    A._xTarget = B._xTarget;
    A._yTarget = B._yTarget;
    A._xSource = B._xSource;
    A._ySource = B._ySource;

    A._teams = B._teams;
    A._level = B._level;
    A._inTargetTeam = B._inTargetTeam;
    A._inSourceTeam = B._inSourceTeam;
    A._targetRoot = B._targetRoot;
    A._sourceRoot = B._sourceRoot;

    A._block.Clear();
    A._block.type = B._block.type;

    switch( B._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        A._block.data.N = A.NewNode();    
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int j=0; j<16; ++j )
            nodeA.children[j] = new DistQuasi2dHMat<Scalar>;

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).CopyFrom( nodeB.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
    {
        A._block.data.DF = new DistLowRank;
        DistLowRank& DFA = *A._block.data.DF;
        const DistLowRank& DFB = *B._block.data.DF;

        DFA.rank = DFB.rank;
        if( B._inTargetTeam )
            hmat_tools::Copy( DFB.ULocal, DFA.ULocal );
        if( B._inSourceTeam )
            hmat_tools::Copy( DFB.VLocal, DFA.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        A._block.data.SF = new SplitLowRank;
        SplitLowRank& SFA = *A._block.data.SF;
        const SplitLowRank& SFB = *B._block.data.SF;

        SFA.rank = SFB.rank;
        hmat_tools::Copy( SFB.D, SFA.D );
        break;
    }
    case LOW_RANK:
    {
        A._block.data.F = new LowRank<Scalar>;
        LowRank<Scalar>& FA = *A._block.data.F;
        const LowRank<Scalar>& FB = *B._block.data.F;

        hmat_tools::Copy( FB, FA );
        break;
    }
    case SPLIT_DENSE:
        A._block.data.SD = new SplitDense;
        if( B._inSourceTeam )
            hmat_tools::Copy( B._block.data.SD->D, A._block.data.SD->D );
        break;
    case DENSE:
        A._block.data.D = new Dense<Scalar>;
        hmat_tools::Copy( *B._block.data.D, *A._block.data.D );
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
