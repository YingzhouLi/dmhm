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
DistHMat3d<Scalar>::CopyFrom( const DistHMat3d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::CopyFrom");
#endif
    DistHMat3d<Scalar>& A = *this;

    A.numLevels_ = B.numLevels_;
    A.maxRank_ = B.maxRank_;
    A.targetOffset_ = B.targetOffset_;
    A.sourceOffset_ = B.sourceOffset_;
    A.stronglyAdmissible_ = B.stronglyAdmissible_;

    A.xSizeTarget_ = B.xSizeTarget_;
    A.ySizeTarget_ = B.ySizeTarget_;
    A.zSizeTarget_ = B.zSizeTarget_;
    A.xSizeSource_ = B.xSizeSource_;
    A.ySizeSource_ = B.ySizeSource_;
    A.zSizeSource_ = B.zSizeSource_;

    A.xTarget_ = B.xTarget_;
    A.yTarget_ = B.yTarget_;
    A.zTarget_ = B.zTarget_;
    A.xSource_ = B.xSource_;
    A.ySource_ = B.ySource_;
    A.zSource_ = B.zSource_;

    A.teams_ = B.teams_;
    A.level_ = B.level_;
    A.inTargetTeam_ = B.inTargetTeam_;
    A.inSourceTeam_ = B.inSourceTeam_;
    A.targetRoot_ = B.targetRoot_;
    A.sourceRoot_ = B.sourceRoot_;

    A.block_.Clear();
    A.block_.type = B.block_.type;

    switch( B.block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        A.block_.data.N = A.NewNode();
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int j=0; j<64; ++j )
            nodeA.children[j] = new DistHMat3d<Scalar>;

        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
                nodeA.Child(t,s).CopyFrom( nodeB.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
    {
        A.block_.data.DF = new DistLowRank;
        DistLowRank& DFA = *A.block_.data.DF;
        const DistLowRank& DFB = *B.block_.data.DF;

        DFA.rank = DFB.rank;
        if( B.inTargetTeam_ )
            hmat_tools::Copy( DFB.ULocal, DFA.ULocal );
        if( B.inSourceTeam_ )
            hmat_tools::Copy( DFB.VLocal, DFA.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        A.block_.data.SF = new SplitLowRank;
        SplitLowRank& SFA = *A.block_.data.SF;
        const SplitLowRank& SFB = *B.block_.data.SF;

        SFA.rank = SFB.rank;
        hmat_tools::Copy( SFB.D, SFA.D );
        break;
    }
    case LOW_RANK:
    {
        A.block_.data.F = new LowRank<Scalar>;
        LowRank<Scalar>& FA = *A.block_.data.F;
        const LowRank<Scalar>& FB = *B.block_.data.F;

        hmat_tools::Copy( FB, FA );
        break;
    }
    case SPLIT_DENSE:
        A.block_.data.SD = new SplitDense;
        if( B.inSourceTeam_ )
            hmat_tools::Copy( B.block_.data.SD->D, A.block_.data.SD->D );
        break;
    case DENSE:
        A.block_.data.D = new Dense<Scalar>;
        hmat_tools::Copy( *B.block_.data.D, *A.block_.data.D );
        break;
    default:
        break;
    }
}

} // namespace dmhm
