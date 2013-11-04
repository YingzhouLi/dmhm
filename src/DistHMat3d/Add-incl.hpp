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
    {
        for( int t=0; t<8; ++t )
            block_.data.N->Child(t,t).AddConstantToDiagonal( alpha );
        break;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    {
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    }
    case DENSE:
    {
        Scalar* DBuffer = block_.data.D->Buffer();
        const int m = block_.data.D->Height();
        const int n = block_.data.D->Width();
        const int DLDim = block_.data.D->LDim();
        for( int j=0; j<std::min(m,n); ++j )
            DBuffer[j+j*DLDim] += alpha;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat3d<Scalar>::Axpy
( Scalar alpha, DistHMat3d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::Axpy");
#endif
    DistHMat3d<Scalar>& A = *this;
    Scalar estimateA = A.ParallelEstimateTwoNorm( 2, 3 );
    Scalar estimateB = B.ParallelEstimateTwoNorm( 2, 3 );
    Real twoNorm = std::abs(estimateA)+std::abs(estimateB);
    AddMatrix( alpha, B );
    B.MultiplyHMatCompress( twoNorm );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::AddMatrix
( Scalar alpha, DistHMat3d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::AddMatrix");
#endif
    DistHMat3d<Scalar>& A = *this;
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& nodeA = *A.block_.data.N;
        Node& nodeB = *B.block_.data.N;
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
                nodeA.Child(t,s).AddMatrix( alpha, nodeB.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DFA = *A.block_.data.DF;
        DistLowRank& DFB = *B.block_.data.DF;
        if( A.inTargetTeam_ )
        {
            Dense<Scalar>& UA = DFA.ULocal;
            Dense<Scalar>& UB = DFB.ULocal;
            Dense<Scalar> Utmp;
            hmat_tools::Copy( UB, Utmp );
            int ra = DFA.rank;
            int rb = DFB.rank;
            DFB.rank = ra+rb;
            int LH = UA.Height();
            UB.Resize( LH, ra+rb );
            for( int i=0; i<ra; ++i )
                for(int j=0; j<LH; ++j )
                    UB.Set(j,i,alpha*UA.Get(j,i));
            for( int i=0; i<rb; ++i )
                for(int j=0; j<LH; ++j )
                    UB.Set(j,i,Utmp.Get(j,i));
        }
        if( A.inSourceTeam_ )
        {
            Dense<Scalar>& VA = DFA.VLocal;
            Dense<Scalar>& VB = DFB.VLocal;
            Dense<Scalar> Vtmp;
            hmat_tools::Copy( VB, Vtmp );
            int ra = DFA.rank;
            int rb = DFB.rank;
            DFB.rank = ra+rb;
            int LW = VA.Height();
            VB.Resize( LW, ra+rb );
            for( int i=0; i<ra; ++i )
                for(int j=0; j<LW; ++j )
                    VB.Set(j,i,VA.Get(j,i));
            for( int i=0; i<rb; ++i )
                for(int j=0; j<LW; ++j )
                    VB.Set(j,i,Vtmp.Get(j,i));
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SFA = *A.block_.data.SF;
        SplitLowRank& SFB = *B.block_.data.SF;
        if( A.inTargetTeam_ )
        {
            Dense<Scalar>& UA = SFA.D;
            Dense<Scalar>& UB = SFB.D;
            Dense<Scalar> Utmp;
            hmat_tools::Copy( UB, Utmp );
            int ra = SFA.rank;
            int rb = SFB.rank;
            SFB.rank = ra+rb;
            int LH = UA.Height();
            UB.Resize( LH, ra+rb );
            for( int i=0; i<ra; ++i )
                for(int j=0; j<LH; ++j )
                    UB.Set(j,i,alpha*UA.Get(j,i));
            for( int i=0; i<rb; ++i )
                for(int j=0; j<LH; ++j )
                    UB.Set(j,i,Utmp.Get(j,i));
        }
        if( A.inSourceTeam_ )
        {
            Dense<Scalar>& VA = SFA.D;
            Dense<Scalar>& VB = SFB.D;
            Dense<Scalar> Vtmp;
            hmat_tools::Copy( VB, Vtmp );
            int ra = SFA.rank;
            int rb = SFB.rank;
            SFB.rank = ra+rb;
            int LW = VA.Height();
            VB.Resize( LW, ra+rb );
            for( int i=0; i<ra; ++i )
                for(int j=0; j<LW; ++j )
                    VB.Set(j,i,VA.Get(j,i));
            for( int i=0; i<rb; ++i )
                for(int j=0; j<LW; ++j )
                    VB.Set(j,i,Vtmp.Get(j,i));
        }
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar>& FA = *A.block_.data.F;
        LowRank<Scalar>& FB = *B.block_.data.F;
        Dense<Scalar>& UA = FA.U;
        Dense<Scalar>& UB = FB.U;
        Dense<Scalar> Utmp;
        hmat_tools::Copy( UB, Utmp );
        int ra = UA.Width();
        int rb = UB.Width();
        int LH = UA.Height();
        UB.Resize( LH, ra+rb );
        for( int i=0; i<ra; ++i )
            for(int j=0; j<LH; ++j )
                UB.Set(j,i,alpha*UA.Get(j,i));
        for( int i=0; i<rb; ++i )
            for(int j=0; j<LH; ++j )
                UB.Set(j,i,Utmp.Get(j,i));
        Dense<Scalar>& VA = FA.V;
        Dense<Scalar>& VB = FB.V;
        Dense<Scalar> Vtmp;
        hmat_tools::Copy( VB, Vtmp );
        int LW = VA.Height();
        VB.Resize( LW, ra+rb );
        for( int i=0; i<ra; ++i )
            for(int j=0; j<LW; ++j )
                VB.Set(j,i,VA.Get(j,i));
        for( int i=0; i<rb; ++i )
            for(int j=0; j<LW; ++j )
                VB.Set(j,i,Vtmp.Get(j,i));
        break;
    }
    case SPLIT_DENSE:
    {
        if( A.inSourceTeam_ )
        {
            SplitDense& SDA = *A.block_.data.SD;
            SplitDense& SDB = *B.block_.data.SD;
            hmat_tools::Axpy( alpha, SDA.D, SDB.D );
        }
        break;
    }
    case DENSE:
    {
        Dense<Scalar>& DA = *A.block_.data.D;
        Dense<Scalar>& DB = *B.block_.data.D;
        hmat_tools::Axpy( alpha, DA, DB );
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
