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
DistHMat2d<Scalar>::MultiplyHMatMainSetUp
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSetUp");
#endif
    const DistHMat2d<Scalar>& A = *this;

    C.numLevels_ = A.numLevels_;
    C.maxRank_ = A.maxRank_;
    C.targetOffset_ = A.targetOffset_;
    C.sourceOffset_ = B.sourceOffset_;
    C.stronglyAdmissible_ = ( A.stronglyAdmissible_ || B.stronglyAdmissible_ );

    C.xSizeTarget_ = A.xSizeTarget_;
    C.ySizeTarget_ = A.ySizeTarget_;
    C.xSizeSource_ = B.xSizeSource_;
    C.ySizeSource_ = B.ySizeSource_;

    C.xTarget_ = A.xTarget_;
    C.yTarget_ = A.yTarget_;
    C.xSource_ = B.xSource_;
    C.ySource_ = B.ySource_;

    C.teams_ = A.teams_;
    C.level_ = A.level_;
    C.inTargetTeam_ = A.inTargetTeam_;
    C.inSourceTeam_ = B.inSourceTeam_;
    C.targetRoot_ = A.targetRoot_;
    C.sourceRoot_ = B.sourceRoot_;
    
    mpi::Comm team = teams_->Team( A.level_ );
    const int teamSize = mpi::CommSize( team );
    if( C.Admissible() ) // C is low-rank
    {
        if( teamSize > 1 )
        {
            if( C.inSourceTeam_ || C.inTargetTeam_ )
            {
                C.block_.type = DIST_LOW_RANK;
                C.block_.data.DF = new DistLowRank;
                DistLowRank& DF = *C.block_.data.DF;
                DF.rank = 0;
                DF.ULocal.Resize( C.LocalHeight(), 0 );
                DF.VLocal.Resize( C.LocalWidth(), 0 );
            }
            else
            {
                C.block_.type = DIST_LOW_RANK_GHOST;
                C.block_.data.DFG = new DistLowRankGhost;
                DistLowRankGhost& DFG = *C.block_.data.DFG;
                DFG.rank = 0;
            }
        }
        else // teamSize == 1
        {
            if( C.sourceRoot_ == C.targetRoot_ )
            {
                if( C.inSourceTeam_ || C.inTargetTeam_ )
                {
                    C.block_.type = LOW_RANK;
                    C.block_.data.F = new LowRank<Scalar>;
                    LowRank<Scalar>& F = *C.block_.data.F;
                    F.U.Resize( C.Height(), 0 );
                    F.V.Resize( C.Width(), 0 );
                }
                else
                {
                    C.block_.type = LOW_RANK_GHOST;
                    C.block_.data.FG = new LowRankGhost;
                    LowRankGhost& FG = *C.block_.data.FG;
                    FG.rank = 0;
                }
            }
            else
            {
                if( C.inSourceTeam_ || C.inTargetTeam_ )
                {
                    C.block_.type = SPLIT_LOW_RANK;
                    C.block_.data.SF = new SplitLowRank;
                    SplitLowRank& SF = *C.block_.data.SF;
                    SF.rank = 0;
                    if( C.inTargetTeam_ )
                        SF.D.Resize( C.Height(), 0 );
                    else
                        SF.D.Resize( C.Width(), 0 );
                }
                else
                {
                    C.block_.type = SPLIT_LOW_RANK_GHOST;
                    C.block_.data.SFG = new SplitLowRankGhost;
                    SplitLowRankGhost& SFG = *C.block_.data.SFG;
                    SFG.rank = 0;
                }
            }
        }
    }
    else if( C.numLevels_ > 1 ) // C is hierarchical
    {
        if( teamSize > 1 )
        {
            if( C.inSourceTeam_ || C.inTargetTeam_ )
            {
                C.block_.type = DIST_NODE;
                C.block_.data.N = C.NewNode();
                Node& node = *C.block_.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistHMat2d<Scalar>;
            }
            else
            {
                C.block_.type = DIST_NODE_GHOST;
                C.block_.data.N = C.NewNode();
                Node& node = *C.block_.data.N;
                for( int j=0; j<16; ++j )
                    node.children[j] = new DistHMat2d<Scalar>;
            }
        }
        else
        {
            if( C.sourceRoot_ == C.targetRoot_ )
            {
                if( C.inSourceTeam_ || C.inTargetTeam_ )
                {
                    C.block_.type = NODE;
                    C.block_.data.N = C.NewNode();
                    Node& node = *C.block_.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistHMat2d<Scalar>;
                }
                else
                {
                    C.block_.type = NODE_GHOST;
                    C.block_.data.N = C.NewNode();
                    Node& node = *C.block_.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistHMat2d<Scalar>;
                }
            }
            else
            {
                if( C.inSourceTeam_ || C.inTargetTeam_ )
                {
                    C.block_.type = SPLIT_NODE;
                    C.block_.data.N = C.NewNode();
                    Node& node = *C.block_.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistHMat2d<Scalar>;
                }
                else
                {
                    C.block_.type = SPLIT_NODE_GHOST;
                    C.block_.data.N = C.NewNode();
                    Node& node = *C.block_.data.N;
                    for( int j=0; j<16; ++j )
                        node.children[j] = 
                            new DistHMat2d<Scalar>;
                }
            }
        }
    }
    else // C is dense
    {
        if( C.sourceRoot_ == C.targetRoot_ )
        {
            if( C.inSourceTeam_ || C.inTargetTeam_ )
            {
                C.block_.type = DENSE;
                C.block_.data.D = new Dense<Scalar>( A.Height(), B.Width() );
                hmat_tools::Scale( Scalar(0), *C.block_.data.D );
            }
            else
                C.block_.type = DENSE_GHOST;
        }
        else
        {
            if( C.inSourceTeam_ || C.inTargetTeam_ )
            {
                C.block_.type = SPLIT_DENSE;
                C.block_.data.SD = new SplitDense;
                if( C.inSourceTeam_ )
                {
                    C.block_.data.SD->D.Resize( A.Height(), B.Width() );
                    hmat_tools::Scale( Scalar(0), C.block_.data.SD->D );
                }
            }
            else
                C.block_.type = SPLIT_DENSE_GHOST;
        }
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPrecompute
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPrecompute");
#endif
    DistHMat2d<Scalar>& A = *this;
    if( !A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_ )
    {
        C.block_.type = EMPTY;
        return;
    }
    if( C.block_.type == EMPTY )
        A.MultiplyHMatMainSetUp( B, C );

    if( A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    // Handle all H H cases here
    const bool admissibleC = C.Admissible();
    const int key = A.sourceOffset_;
    const int sampleRank = SampleRank( C.MaxRank() );
    if( !admissibleC )
    {
        // Take care of the H += H H cases first
        switch( A.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                // Start H += H H
                Node& nodeA = *A.block_.data.N;
                Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainPrecompute
                            ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                              startLevel, endLevel, startUpdate, endUpdate, r );
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }
    else if( A.level_ >= startLevel && A.level_ < endLevel && 
             update >= startUpdate && update < endUpdate )
    {
        // Handle precomputation of A's row space
        if( !A.beganRowSpaceComp_ )
        {
            switch( A.block_.type )
            {
            case DIST_NODE:
            case SPLIT_NODE:
            case NODE:
                switch( B.block_.type )
                {
                case DIST_NODE:
                case DIST_NODE_GHOST:
                case SPLIT_NODE:
                case SPLIT_NODE_GHOST:
                case NODE:
                case NODE_GHOST:
                {
                    A.rowOmega_.Resize( A.LocalHeight(), sampleRank );
                    ParallelGaussianRandomVectors( A.rowOmega_ );

                    A.rowT_.Resize( A.LocalWidth(), sampleRank );
                    hmat_tools::Scale( Scalar(0), A.rowT_ );

                    A.AdjointMultiplyDenseInitialize
                    ( A.rowContext_, sampleRank );

                    A.AdjointMultiplyDensePrecompute
                    ( A.rowContext_, Scalar(1), A.rowOmega_, A.rowT_ );

                    A.beganRowSpaceComp_ = true;
                    break;
                }
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
        // Handle precomputation of B's column space
        if( !B.beganColSpaceComp_ )
        {
            switch( A.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
                switch( B.block_.type )
                {
                case DIST_NODE:
                case SPLIT_NODE:
                case NODE:
                {
                    B.colOmega_.Resize( B.LocalWidth(), sampleRank ); 
                    ParallelGaussianRandomVectors( B.colOmega_ );

                    B.colT_.Resize( B.LocalHeight(), sampleRank );
                    hmat_tools::Scale( Scalar(0), B.colT_ );

                    B.MultiplyDenseInitialize( B.colContext_, sampleRank );
                    B.MultiplyDensePrecompute
                    ( B.colContext_, Scalar(1), B.colOmega_, B.colT_ );

                    B.beganColSpaceComp_ = true;
                    break;
                }
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
    }

    if( A.level_ < startLevel || A.level_ >= endLevel || 
        update < startUpdate  || update >= endUpdate )
        return;

    switch( A.block_.type )
    {
    case DIST_NODE:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            break;
        case DIST_LOW_RANK:
        {
            // Start H/F += H F
            const DistLowRank& DFB = *B.block_.data.DF;
            C.UMap_.Set( key, new Dense<Scalar>(C.LocalHeight(),DFB.rank) );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get(key);

            hmat_tools::Scale( Scalar(0), C.UMap_.Get(key) );
            A.MultiplyDenseInitialize( context, DFB.rank );
            A.MultiplyDensePrecompute
            ( context, alpha, DFB.ULocal, C.UMap_.Get(key) );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We must be in the left team
            const DistLowRankGhost& DFGB = *B.block_.data.DFG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            A.MultiplyDenseInitialize( context, DFGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            // Start H/F += H F
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( C.inTargetTeam_ )
            {
                // Our process owns the left and right sides
                C.mainContextMap_.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C.mainContextMap_.Get( key );

                A.MultiplyDenseInitialize( context, SFB.rank );
            }
            else
            {
                // We are the middle process
                C.mainContextMap_.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C.mainContextMap_.Get( key );

                Dense<Scalar> dummy( 0, SFB.rank );
                A.MultiplyDenseInitialize( context, SFB.rank );
                A.MultiplyDensePrecompute( context, alpha, SFB.D, dummy );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            A.MultiplyDenseInitialize( context, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We are the middle and right processes
            const LowRank<Scalar>& FB = *B.block_.data.F;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            Dense<Scalar> dummy( 0, FB.Rank() );
            A.MultiplyDenseInitialize( context, FB.Rank() );
            A.MultiplyDensePrecompute( context, alpha, FB.U, dummy );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Start H/F += H F
            // We are the left process
            const LowRankGhost& FGB = *B.block_.data.FG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            A.MultiplyDenseInitialize( context, FGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case NODE:
            break;
        case SPLIT_LOW_RANK:
        {
            // Start H/F += H F
            // We are the left and middle processes
            const SplitLowRank& SFB = *B.block_.data.SF;
            C.UMap_.Set( key, new Dense<Scalar>( C.Height(), SFB.rank ) );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
            A.MultiplyDenseInitialize( context, SFB.rank );
            A.MultiplyDensePrecompute
            ( context, alpha, SFB.D, C.UMap_.Get( key ) );
            break;
        }
        case LOW_RANK:
        {
            // Start H/F += H F
            // We own all of A, B, and C
            const LowRank<Scalar>& FB = *B.block_.data.F;
            C.UMap_.Set( key, new Dense<Scalar>( C.Height(), FB.Rank() ) );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
            A.MultiplyDenseInitialize( context, FB.Rank() );
            A.MultiplyDensePrecompute
            ( context, alpha, FB.U, C.UMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A.block_.data.DF;
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            // Start H/F += F H
            C.VMap_.Set( key, new Dense<Scalar>( C.LocalWidth(), DFA.rank ) );
            hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, DFA.rank );
            B.TransposeMultiplyDensePrecompute
            ( context, alpha, DFA.VLocal, C.VMap_.Get( key ) );
            break;
        }
        case DIST_LOW_RANK:
        {
            // Start H/F += F F
            if( A.inSourceTeam_ )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                const int kLocal = A.LocalWidth();
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    C.ZMap_.Set( key, new Dense<Scalar>( DFA.rank, DFB.rank ) );
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    const char option = 'T';
                    blas::Gemm
                    ( option, 'N', DFA.rank, DFB.rank, kLocal,
                      Scalar(1), DFA.VLocal.LockedBuffer(), DFA.VLocal.LDim(),
                                 DFB.ULocal.LockedBuffer(), DFB.ULocal.LDim(),
                      Scalar(0), ZC.Buffer(),               ZC.LDim() );
                }
            }
            break;
        }
        case DIST_NODE_GHOST:
        case DIST_LOW_RANK_GHOST:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            // Start H/F += F H
            // We are in the right team
            const DistLowRankGhost& DFGA = *A.block_.data.DFG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, DFGA.rank );
            break;
        }
        case DIST_LOW_RANK:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A.block_.data.SF;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            // We are either the middle process or both the left and right
            if( A.inSourceTeam_ )
            {
                Dense<Scalar> dummy( 0, SFA.rank );
                C.mainContextMap_.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C.mainContextMap_.Get( key );
                B.TransposeMultiplyDenseInitialize( context, SFA.rank );
                B.TransposeMultiplyDensePrecompute
                ( context, alpha, SFA.D, dummy );
            }
            else
            {
                C.mainContextMap_.Set( key, new MultiplyDenseContext );
                MultiplyDenseContext& context = C.mainContextMap_.Get( key );
                B.TransposeMultiplyDenseInitialize( context, SFA.rank );
            }
            break;
        }
        case NODE:
        {
            // We are the middle and right process
            C.VMap_.Set( key, new Dense<Scalar>( B.Width(), SFA.rank ) );
            Dense<Scalar>& CV = C.VMap_.Get( key );
                
            hmat_tools::Scale( Scalar(0), CV );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, SFA.rank );
            B.TransposeMultiplyDensePrecompute( context, alpha, SFA.D, CV );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A.inSourceTeam_ )
            {
                const SplitLowRank& SFB = *B.block_.data.SF;
                const int k = A.Width();
                if( SFA.rank != 0 && SFB.rank != 0 )
                {
                    C.ZMap_.Set( key, new Dense<Scalar>( SFA.rank, SFB.rank ) );
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    const char option = 'T';
                    blas::Gemm
                    ( option, 'N', SFA.rank, SFB.rank, k,
                      Scalar(1), SFA.D.LockedBuffer(), SFA.D.LDim(),
                                 SFB.D.LockedBuffer(), SFB.D.LDim(),
                      Scalar(0), ZC.Buffer(),          ZC.LDim() );
                }
            }
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank<Scalar>& FB = *B.block_.data.F;
            const int k = A.Width();
            if( SFA.rank != 0 && FB.Rank() != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( SFA.rank, FB.Rank() ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                const char option = 'T';
                blas::Gemm
                ( option, 'N', SFA.rank, FB.Rank(), k,
                  Scalar(1), SFA.D.LockedBuffer(), SFA.D.LDim(),
                             FB.U.LockedBuffer(),  FB.U.LDim(),
                  Scalar(0), ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case DENSE:
        {
            // We must be both the middle and right process
            const int k = B.Height();
            const int n = B.Width();
            C.VMap_.Set( key, new Dense<Scalar>( n, SFA.rank ) );
            const Dense<Scalar>& DB = *B.block_.data.D;
            Dense<Scalar>& VC = C.VMap_.Get( key );
            const char option = 'T';
            blas::Gemm
            ( option, 'N', n, SFA.rank, k,
              alpha,     DB.LockedBuffer(),    DB.LDim(),
                         SFA.D.LockedBuffer(), SFA.D.LDim(),
              Scalar(0), VC.Buffer(),          VC.LDim() );
            break;
        }
        case SPLIT_NODE_GHOST:
        case NODE_GHOST:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
        case SPLIT_DENSE:
        case SPLIT_DENSE_GHOST:
        case DENSE_GHOST:
            // We are the left process, so there is no work to do
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
    {
        // We must be the right process
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            const SplitLowRankGhost& SFGA = *A.block_.data.SFG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, SFGA.rank );
            break;
        }
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK:
    {
        const LowRank<Scalar>& FA = *A.block_.data.F;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            // We must be the left and middle process
            Dense<Scalar> dummy( 0, FA.Rank() );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, FA.Rank() );
            B.TransposeMultiplyDensePrecompute
            ( context, alpha, FA.V, dummy );
            break;
        }
        case NODE:
        {
            // We must own all of A, B, and C
            C.VMap_.Set( key, new Dense<Scalar>( B.Width(), FA.Rank() ) );
            hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, FA.Rank() );
            B.TransposeMultiplyDensePrecompute
            ( context, alpha, FA.V, C.VMap_.Get( key ) );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            // We must be the left and middle process
            const SplitLowRank& SFB = *B.block_.data.SF;
            const int k = B.Height();
            if( FA.Rank() != 0 && SFB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( FA.Rank(), SFB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                const char option = 'T';
                blas::Gemm
                ( option, 'N', FA.Rank(), SFB.rank, k,
                  Scalar(1), FA.V.LockedBuffer(),  FA.V.LDim(),
                             SFB.D.LockedBuffer(), SFB.D.LDim(),
                  Scalar(0), ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case LOW_RANK:
        {
            // We must own all of A, B, and C
            const LowRank<Scalar>& FB = *B.block_.data.F;
            const int k = B.Height();
            if( FA.Rank() != 0 && FB.Rank() != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( FA.Rank(), FB.Rank() ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                const char option = 'T';
                blas::Gemm
                ( option, 'N', FA.Rank(), FB.Rank(), k,
                  Scalar(1), FA.V.LockedBuffer(), FA.V.LDim(),
                             FB.U.LockedBuffer(), FB.U.LDim(),
                  Scalar(0), ZC.Buffer(),         ZC.LDim() );
            }
            break;
        }
        case SPLIT_DENSE:
            // We must be the left and middle process, but there is no
            // work to be done (split dense owned by right process)
            break;
        case DENSE:
        {
            // We must own all of A, B, and C
            const int k = B.Height();
            const int n = B.Width();
            C.VMap_.Set( key, new Dense<Scalar>( n, FA.Rank() ) );
            const Dense<Scalar>& DB = *B.block_.data.D;
            Dense<Scalar>& VC = C.VMap_.Get( key );
            const char option = 'T';
            blas::Gemm
            ( option, 'N', n, FA.Rank(), k,
              alpha,     DB.LockedBuffer(),   DB.LDim(),
                         FA.V.LockedBuffer(), FA.V.LDim(),
              Scalar(0), VC.Buffer(),         VC.LDim() );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK_GHOST:
    {
        // We must be the right process
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            const LowRankGhost& FGA = *A.block_.data.FG;
            C.mainContextMap_.Set( key, new MultiplyDenseContext );
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            B.TransposeMultiplyDenseInitialize( context, FGA.rank );
            break;
        }
        case SPLIT_LOW_RANK:
        case SPLIT_DENSE:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDense& SDA = *A.block_.data.SD;
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            if( A.inSourceTeam_ )
            {
                const SplitLowRank& SFB = *B.block_.data.SF;
                const int m = A.Height();
                const int k = A.Width();
                if( m != 0 && SFB.rank != 0 )
                {
                    C.ZMap_.Set( key, new Dense<Scalar>( m, SFB.rank ) );
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    blas::Gemm
                    ( 'N', 'N', m, SFB.rank, k,
                      alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                                 SFB.D.LockedBuffer(), SFB.D.LDim(),
                      Scalar(0), ZC.Buffer(),          ZC.LDim() );
                }
            }
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar>& FB = *B.block_.data.F;
            const int m = A.Height();
            const int k = A.Width();
            if( m != 0 && FB.Rank() != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( m, FB.Rank() ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', m, FB.Rank(), k,
                  alpha,     SDA.D.LockedBuffer(), SDA.D.LDim(),
                             FB.U.LockedBuffer(),  FB.U.LDim(),
                  Scalar(0), ZC.Buffer(),          ZC.LDim() );
            }
            break;
        }
        case DENSE:
        {
            const Dense<Scalar>& DB = *B.block_.data.D;
            if( admissibleC )
            {
                // F += D D
                if( C.storedDenseUpdate_ )
                    hmat_tools::Multiply( alpha, SDA.D, DB, Scalar(1), C.D_ );
                else
                {
                    hmat_tools::Multiply( alpha, SDA.D, DB, C.D_ );
                    C.haveDenseUpdate_ = true;
                    C.storedDenseUpdate_ = true;
                }
            }
            else
            {
                // D += D D
                SplitDense& SDC = *C.block_.data.SD;
                hmat_tools::Multiply
                ( alpha, SDA.D, DB, Scalar(1), SDC.D );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK_GHOST:
            break;
        case SPLIT_DENSE:
            if( admissibleC && C.inSourceTeam_ )
                C.haveDenseUpdate_ = true;
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE_GHOST:
            if( admissibleC )
                C.haveDenseUpdate_ = true;
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
        if( B.block_.type == SPLIT_DENSE && admissibleC )
            C.haveDenseUpdate_ = true;
        break;
    case DENSE:
    {
        const Dense<Scalar>& DA = *A.block_.data.D;
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B.block_.data.SF;
            const int m = A.Height();
            const int k = A.Width();
            C.UMap_.Set( key, new Dense<Scalar>( m, SFB.rank ) );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            blas::Gemm
            ( 'N', 'N', m, SFB.rank, k,
              alpha,     DA.LockedBuffer(),    DA.LDim(),
                         SFB.D.LockedBuffer(), SFB.D.LDim(),
              Scalar(0), UC.Buffer(),          UC.LDim() );
            break;
        }
        case LOW_RANK:
        {
            const LowRank<Scalar>& FB = *B.block_.data.F;
            if( admissibleC )
            {
                LowRank<Scalar>& FC = *C.block_.data.F;
                LowRank<Scalar> temp;
                hmat_tools::Multiply( alpha, DA, FB, temp );
                hmat_tools::RoundedUpdate
                ( C.MaxRank(), Scalar(1), temp, Scalar(1), FC );
            }
            else
            {
                Dense<Scalar>& DC = *C.block_.data.D;
                hmat_tools::Multiply( alpha, DA, FB, Scalar(1), DC );
            }
            break;
        }
        case SPLIT_DENSE:
            if( admissibleC )
                C.haveDenseUpdate_ = true;
            break;
        case DENSE:
        {
            const Dense<Scalar>& DB = *B.block_.data.D;
            if( admissibleC )
            {
                // F += D D
                if( C.storedDenseUpdate_ )
                    hmat_tools::Multiply( alpha, DA, DB, Scalar(1), C.D_ );
                else
                {
                    hmat_tools::Multiply( alpha, DA, DB, C.D_ );
                    C.haveDenseUpdate_ = true;
                    C.storedDenseUpdate_ = true;
                }
            }
            else
            {
                // D += D D
                Dense<Scalar>& DC = *C.block_.data.D;
                hmat_tools::Multiply( alpha, DA, DB, Scalar(1), DC );
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE_GHOST:
        if( B.block_.type == SPLIT_DENSE && admissibleC )
            C.haveDenseUpdate_ = true;
        break;
    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSums
( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSums");
#endif
    DistHMat2d<Scalar>& A = *this;

    // Compute the message sizes for each reduce
    const unsigned numTeamLevels = teams_->NumLevels();
    const unsigned numReduces = numTeamLevels-1;
    std::vector<int> sizes( numReduces, 0 );
    A.MultiplyHMatMainSumsCountA( sizes, startLevel, endLevel );
    B.MultiplyHMatMainSumsCountB( sizes, startLevel, endLevel );
    A.MultiplyHMatMainSumsCountC
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( unsigned i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( unsigned i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatMainSumsPackA( buffer, offsetsCopy, startLevel, endLevel );
    B.MultiplyHMatMainSumsPackB( buffer, offsetsCopy, startLevel, endLevel );
    A.MultiplyHMatMainSumsPackC
    ( B, C, buffer, offsetsCopy, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the reduces with log2(p) messages
    A.teams_->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of communicators have data)
    A.MultiplyHMatMainSumsUnpackA( buffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainSumsUnpackB( buffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainSumsUnpackC
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsCountA
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsCountA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDenseSumsCount( sizes, rowContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsCountA
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsPackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel && 
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )   
            TransposeMultiplyDenseSumsPack( rowContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsPackA
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsUnpackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDenseSumsUnpack( rowContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsUnpackA
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsCountB
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsCountB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseSumsCount( sizes, colContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsCountB
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsPackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseSumsPack( colContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsPackB
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsUnpackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseSumsUnpack( colContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainSumsUnpackB
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsCountC
( const DistHMat2d<Scalar>& B,
  const DistHMat2d<Scalar>& C,
  std::vector<int>& sizes, 
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsCountC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                const Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                A.MultiplyDenseSumsCount( sizes, DFB.rank );
            }
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                B.TransposeMultiplyDenseSumsCount( sizes, DFA.rank );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inSourceTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                sizes[teamLevel] += DFA.rank*DFB.rank;
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsPackC
( const DistHMat2d<Scalar>& B, 
        DistHMat2d<Scalar>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsPackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseSumsPack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseSumsPack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inSourceTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( &buffer[offsets[teamLevel]], 
                      C.ZMap_.Get( key ).LockedBuffer(), DFA.rank*DFB.rank );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainSumsUnpackC
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainSumsUnpackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainSumsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseSumsUnpack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseSumsUnpack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inSourceTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( C.ZMap_.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFB.rank );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassData
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C,
  int startLevel, int endLevel, int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassData");
#endif
    DistHMat2d<Scalar>& A = *this;

#ifdef TIME_MULTIPLY
    Timer timer;    
    timer.Start( 0 );
#endif

    // Compute send and recv sizes
    std::map<int,int> sendSizes, recvSizes;
    A.MultiplyHMatMainPassDataCountA
    ( sendSizes, recvSizes, startLevel, endLevel );
    B.MultiplyHMatMainPassDataCountB
    ( sendSizes, recvSizes, startLevel, endLevel );
    A.MultiplyHMatMainPassDataCountC
    ( B, C, sendSizes, recvSizes, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 0 );
    timer.Start( 1 );
#endif

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
    std::vector<Scalar> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    A.MultiplyHMatMainPassDataPackA
    ( sendBuffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainPassDataPackB
    ( sendBuffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainPassDataPackC
    ( B, C, sendBuffer, offsets, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 1 );
    timer.Start( 2 );
#endif

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
    int offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 2 );
#endif

    mpi::Barrier( comm );

#ifdef TIME_MULTIPLY
    timer.Start( 3 );
#endif

    // Start the non-blocking sends
    const int numSends = sendSizes.size();
    std::vector<mpi::Request> sendRequests( numSends );
    offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 3 );
    timer.Start( 4 );
#endif

    // Unpack as soon as we have received our data
    mpi::WaitAll( numRecvs, &recvRequests[0] );
#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 4 );
    timer.Start( 5 );
#endif
    A.MultiplyHMatMainPassDataUnpackA
    ( recvBuffer, recvOffsets, startLevel, endLevel );
    B.MultiplyHMatMainPassDataUnpackB
    ( recvBuffer, recvOffsets, startLevel, endLevel );
    A.MultiplyHMatMainPassDataUnpackC
    ( B, C, recvBuffer, recvOffsets, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 5 );
    timer.Start( 6 );
#endif

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );

#ifdef TIME_MULTIPLY
    mpi::Barrier( mpi::COMM_WORLD );
    timer.Stop( 6 );

    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    std::ostringstream os;
    os << "Multiply-PassData-" << commRank << ".log";
    std::ofstream file( os.str().c_str(), std::ios::app | std::ios::out );
    file << "Compute send/recv sizes:  " << timer.GetTime( 0 ) << " seconds.\n"
         << "Pack send/recv buffers:   " << timer.GetTime( 1 ) << " seconds.\n"
         << "Start non-blocking recvs: " << timer.GetTime( 2 ) << " seconds.\n"
         << "Start non-blocking sends: " << timer.GetTime( 3 ) << " seconds.\n"
         << "Wait for recvs to finish: " << timer.GetTime( 4 ) << " seconds.\n"
         << "Unpack recv buffer:       " << timer.GetTime( 5 ) << " seconds.\n"
         << "Wait for sends to finish: " << timer.GetTime( 6 ) << " seconds.\n"
         << std::endl;
    file.close();
#endif
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataCountA
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataCountA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, rowContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataCountA
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataPackA
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataPackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDensePassDataPack
            ( rowContext_, rowOmega_, sendBuffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataPackA
                    ( sendBuffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataUnpackA
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel ) 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataUnpackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDensePassDataUnpack
            ( rowContext_, recvBuffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataUnpackA
                    ( recvBuffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataCountB
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataCountB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDensePassDataCount
            ( sendSizes, recvSizes, colContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataCountB
                    ( sendSizes, recvSizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataPackB
( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataPackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDensePassDataPack( colContext_, sendBuffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataPackB
                    ( sendBuffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataUnpackB
( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataUnpackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDensePassDataUnpack( colContext_, recvBuffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainPassDataUnpackB
                    ( recvBuffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataCountC
( const DistHMat2d<Scalar>& B,
  const DistHMat2d<Scalar>& C,
  std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataCountC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    // Take care of the H += H H cases first
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        switch( A.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                if( A.level_+1 < endLevel )
                {
                    // Start H += H H
                    const Node& nodeA = *A.block_.data.N;
                    const Node& nodeB = *B.block_.data.N;
                    const Node& nodeC = *C.block_.data.N;
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            for( int r=0; r<4; ++r )
                                nodeA.Child(t,r).MultiplyHMatMainPassDataCountC
                                ( nodeB.Child(r,s), nodeC.Child(t,s), 
                                  sendSizes, recvSizes, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    if( A.level_ < startLevel || A.level_ >= endLevel ||
        update < startUpdate  || update >= endUpdate )
        return;

    mpi::Comm team = teams_->Team( level_ );
    const int teamRank = mpi::CommRank( team );
    switch( A.block_.type )
    {
    case DIST_NODE:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        {
            // Pass data count for H/F += H F
            const DistLowRank& DFB = *B.block_.data.DF;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, DFB.rank );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data count for H/F += H F. This should only contribute
            // to the recv sizes.
            const DistLowRankGhost& DFGB = *B.block_.data.DFG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, DFGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/F += H F
            const SplitLowRank& SFB = *B.block_.data.SF;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass datal for H/F += H F
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for H/F += H F
            const LowRank<Scalar>& FB = *B.block_.data.F;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for H/F += H F
            const LowRankGhost& FGB = *B.block_.data.FG;
            A.MultiplyDensePassDataCount( sendSizes, recvSizes, FGB.rank );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
        // The only possiblities are recursion, F += H H, and H/F += H F; the
        // first two are not handled here, and the last does not require any
        // work here because the precompute step handled everything.
        break;
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        // The only non-recursive possibilities are H/F += H F and F += H H;
        // the former does not require our participation here and the latter
        // is handled by CountA and CountB.
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A.block_.data.DF;
        switch( B.block_.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, DFA.rank );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && (A.inSourceTeam_ != A.inTargetTeam_) )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                if( A.inSourceTeam_ )
                    AddToMap( sendSizes, A.targetRoot_, DFA.rank*DFB.rank );
                if( A.inTargetTeam_ )
                    AddToMap( recvSizes, A.sourceRoot_, DFA.rank*DFB.rank );
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F
            if( teamRank == 0 )
            {
                const DistLowRankGhost& DFGB = *B.block_.data.DFG;
                AddToMap( recvSizes, A.sourceRoot_, DFA.rank*DFGB.rank );
            }
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    {
        const DistLowRankGhost& DFGA = *A.block_.data.DFG;
        switch( B.block_.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, DFGA.rank );
            break;
        case DIST_LOW_RANK:
            // Pass data for for H/F += F F is between other (two) team(s)
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A.block_.data.SF;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, SFA.rank );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( A.inSourceTeam_ )
                AddToMap( sendSizes, A.targetRoot_, SFA.rank*SFB.rank );
            else
                AddToMap( recvSizes, A.sourceRoot_, SFA.rank*SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            AddToMap( recvSizes, A.sourceRoot_, SFA.rank*SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for H/D/F += F F
            const LowRank<Scalar>& FB = *B.block_.data.F;
            AddToMap( sendSizes, A.targetRoot_, SFA.rank*FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B.block_.data.FG;
            AddToMap( recvSizes, A.sourceRoot_, SFA.rank*FGB.rank );
            break;
        }
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B.inTargetTeam_ )
                AddToMap( sendSizes, B.sourceRoot_, B.Height()*SFA.rank );
            else
                AddToMap( recvSizes, B.targetRoot_, B.Height()*SFA.rank );
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
    {
        const SplitLowRankGhost& SFGA = *A.block_.data.SFG; 
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, SFGA.rank );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is between other two processes
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            AddToMap( recvSizes, B.targetRoot_, B.Height()*SFGA.rank );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK:
    {
        const LowRank<Scalar>& FA = *A.block_.data.F;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, FA.Rank() );
            break;
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            // There is no pass data
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            AddToMap( sendSizes, B.sourceRoot_, B.Height()*FA.Rank() );
            break;
        case DENSE:
            // There is no pass data
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK_GHOST:
    {
        const LowRankGhost& FGA = *A.block_.data.FG;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataCount
            ( sendSizes, recvSizes, FGA.rank );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is in other process
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            AddToMap( recvSizes, B.targetRoot_, B.Height()*FGA.rank );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            // Pass data for D/F += D F
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( A.inSourceTeam_ )
                AddToMap( sendSizes, A.targetRoot_, A.Height()*SFB.rank );
            else
                AddToMap( recvSizes, A.sourceRoot_, A.Height()*SFB.rank );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            AddToMap( recvSizes, A.sourceRoot_, A.Height()*SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // Pass data for D/F += D F
            const LowRank<Scalar>& FB = *B.block_.data.F;
            AddToMap( sendSizes, A.targetRoot_, A.Height()*FB.Rank() );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B.block_.data.FG;
            AddToMap( recvSizes, A.sourceRoot_, A.Height()*FGB.rank );
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B.inSourceTeam_ )
                AddToMap( recvSizes, B.targetRoot_, A.Height()*A.Width() );
            else
                AddToMap( sendSizes, B.sourceRoot_, A.Height()*A.Width() );
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            AddToMap( recvSizes, B.targetRoot_, A.Height()*A.Width() );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            break;
        case SPLIT_DENSE:
            AddToMap( sendSizes, B.sourceRoot_, A.Height()*A.Width() );
            break;
        case DENSE:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
            AddToMap( recvSizes, B.targetRoot_, A.Height()*A.Width() );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataPackC
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataPackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    // Take care of the H += H H cases first
    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        switch( A.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                if( A.level_+1 < endLevel )
                {
                    // Start H += H H
                    const Node& nodeA = *A.block_.data.N;
                    const Node& nodeB = *B.block_.data.N;
                    Node& nodeC = *C.block_.data.N;
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            for( int r=0; r<4; ++r )
                                nodeA.Child(t,r).MultiplyHMatMainPassDataPackC
                                ( nodeB.Child(r,s), nodeC.Child(t,s), 
                                  sendBuffer, offsets, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    if( A.level_ < startLevel || A.level_ >= endLevel ||
        update < startUpdate  || update >= endUpdate )
        return;

    mpi::Comm team = teams_->Team( level_ );
    const int teamRank = mpi::CommRank( team );
    switch( A.block_.type )
    {
    case DIST_NODE:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        case DIST_LOW_RANK_GHOST:
            // Pass data pack for H/F += H F
            A.MultiplyDensePassDataPack
            ( C.mainContextMap_.Get( key ), sendBuffer, offsets );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
            // Pass data pack for H/F += H F
            A.MultiplyDensePassDataPack
            ( C.mainContextMap_.Get( key ), sendBuffer, offsets );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
        // The only possiblities are recursion, F += H H, and H/F += H F; the
        // first two are not handled here, and the last does not require any
        // work here because the precompute step handled everything.
        break;
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        // The only non-recursive possibilities are H/F += H F and F += H H;
        // the former does not require our participation here and the latter
        // is handled by CountA and CountB.
        break;
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A.block_.data.DF;
        switch( B.block_.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            if( A.inSourceTeam_ )
                B.TransposeMultiplyDensePassDataPack
                ( C.mainContextMap_.Get( key ), 
                  DFA.VLocal, sendBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( teamRank == 0 && (A.inSourceTeam_ != A.inTargetTeam_) )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                if( A.inSourceTeam_ && DFA.rank != 0 && DFB.rank != 0  )
                {
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    MemCopy
                    ( &sendBuffer[offsets[A.targetRoot_]], ZC.LockedBuffer(),
                      DFA.rank*DFB.rank );
                    offsets[A.targetRoot_] += DFA.rank*DFB.rank;
                    ZC.Clear();
                    C.ZMap_.Erase( key );
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
            // Pass data for H/F += F F is only receiving here
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
        // We, at most, receive here
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A.block_.data.SF;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            if( A.inSourceTeam_ )
                B.TransposeMultiplyDensePassDataPack
                ( C.mainContextMap_.Get( key ), SFA.D, sendBuffer, offsets );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( A.inSourceTeam_ && SFA.rank != 0 && SFB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( &sendBuffer[offsets[A.targetRoot_]], ZC.LockedBuffer(),
                  SFA.rank*SFB.rank );
                offsets[A.targetRoot_] += SFA.rank*SFB.rank;
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
            // Pass data for H/D/F += F F is just a receive for us
            break;
        case LOW_RANK:
        {
            // Pass data for H/D/F += F F
            const LowRank<Scalar>& FB = *B.block_.data.F;
            if( SFA.rank != 0 && FB.Rank() != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( &sendBuffer[offsets[A.targetRoot_]], ZC.LockedBuffer(),
                  SFA.rank*FB.Rank() );
                offsets[A.targetRoot_] += SFA.rank*FB.Rank();
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        case LOW_RANK_GHOST:
            // Pass data for H/D/F += F F is just a receive for us
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B.inTargetTeam_ && B.Height() != 0 && SFA.rank != 0 )
            {
                MemCopy
                ( &sendBuffer[offsets[B.sourceRoot_]], SFA.D.LockedBuffer(),
                  B.Height()*SFA.rank );
                offsets[B.sourceRoot_] += B.Height()*SFA.rank;
            }
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // Pass data for D/F += F D is in other process
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
        break;
    case LOW_RANK:
    {
        const LowRank<Scalar>& FA = *A.block_.data.F;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataPack
            ( C.mainContextMap_.Get( key ), FA.V, sendBuffer, offsets );
            break;
        case NODE:
        case SPLIT_LOW_RANK:
        case LOW_RANK:
            // There is no pass data
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B.Height() != 0 && FA.Rank() != 0 )
            {
                MemCopy
                ( &sendBuffer[offsets[B.sourceRoot_]], FA.V.LockedBuffer(),
                  B.Height()*FA.Rank() );
                offsets[B.sourceRoot_] += B.Height()*FA.Rank();
            }
            break;
        }
        case DENSE:
            // There is no pass data
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK_GHOST:
        break;
    case SPLIT_DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            // Pass data for D/F += D F
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( A.inSourceTeam_ && A.Height() != 0 && SFB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( &sendBuffer[offsets[A.targetRoot_]], ZC.LockedBuffer(),
                  A.Height()*SFB.rank );
                offsets[A.targetRoot_] += A.Height()*SFB.rank;
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
            break;
        case LOW_RANK:
        {
            // Pass data for D/F += D F
            const LowRank<Scalar>& FB = *B.block_.data.F;
            if( A.Height() != 0 && FB.Rank() != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( &sendBuffer[offsets[A.targetRoot_]], ZC.LockedBuffer(), 
                  A.Height()*FB.Rank() );
                offsets[A.targetRoot_] += A.Height()*FB.Rank();
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        case LOW_RANK_GHOST:
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += D D
            if( B.inTargetTeam_ && A.Height() != 0 && A.Width() != 0 )
            {
                const SplitDense& SDA = *A.block_.data.SD;
                MemCopy
                ( &sendBuffer[offsets[B.sourceRoot_]], SDA.D.LockedBuffer(),
                  A.Height()*A.Width() );
                offsets[B.sourceRoot_] += A.Height()*A.Width();
            }
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
        break;
    case DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case LOW_RANK:
            break;
        case SPLIT_DENSE:
        {
            if( A.Height() != 0 && A.Width() != 0 )
            {
                const Dense<Scalar>& DA = *A.block_.data.D;
                MemCopy
                ( &sendBuffer[offsets[B.sourceRoot_]], DA.LockedBuffer(),
                  A.Height()*A.Width() );
                offsets[B.sourceRoot_] += A.Height()*A.Width();
            }
            break;
        }
        case DENSE:
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE_GHOST:
    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPassDataUnpackC
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPassDataUnpackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    // Take care of the H += H H cases first
    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        switch( A.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                if( A.level_+1 < endLevel )
                {
                    // Start H += H H
                    const Node& nodeA = *A.block_.data.N;
                    const Node& nodeB = *B.block_.data.N;
                    Node& nodeC = *C.block_.data.N;
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            for( int r=0; r<4; ++r )
                                nodeA.Child(t,r).MultiplyHMatMainPassDataUnpackC
                                ( nodeB.Child(r,s), nodeC.Child(t,s), 
                                  recvBuffer, offsets, startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    if( A.level_ < startLevel || A.level_ >= endLevel ||
        update < startUpdate  || update >= endUpdate )
        return;

    mpi::Comm team = teams_->Team( level_ );
    const int teamRank = mpi::CommRank( team );
    switch( A.block_.type )
    {
    case DIST_NODE:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the CountA/CountB subroutines.
            break;
        case DIST_LOW_RANK:
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data unpack for H/F += H F
            A.MultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            // The only possibilities are recursion and F += H H, and the latter
            // is handled in the UnpackA/UnpackB subroutines.
            break;
        case SPLIT_LOW_RANK:
        case SPLIT_LOW_RANK_GHOST:
        case LOW_RANK:
        case LOW_RANK_GHOST:
        {
            // Pass data for H/F += H F
            A.MultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A.block_.data.DF;
        switch( B.block_.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        case DIST_NODE_GHOST:
            // Pass data for H/F += F H is between other (two) team(s)
            break;
        case DIST_LOW_RANK:
            // Pass data for H/F += F F
            if( A.inTargetTeam_ && !A.inSourceTeam_ )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    C.ZMap_.Set( key, new Dense<Scalar>( DFA.rank, DFB.rank ) );
                    if( teamRank == 0 )
                    {
                        Dense<Scalar>& ZC = C.ZMap_.Get( key ); 
                        MemCopy
                        ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                          DFA.rank*DFB.rank );
                        offsets[A.sourceRoot_] += DFA.rank*DFB.rank;
                    }
                }
            }
            break;
        case DIST_LOW_RANK_GHOST:
        {
            // Pass data for H/F += F F
            const DistLowRankGhost& DFGB = *B.block_.data.DFG;
            if( DFA.rank != 0 && DFGB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( DFA.rank, DFGB.rank ) );
                if( teamRank == 0 )
                {
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    MemCopy
                    ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                      DFA.rank*DFGB.rank );
                    offsets[A.sourceRoot_] += DFA.rank*DFGB.rank;
                }
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        case DIST_LOW_RANK:
            // Pass data for for H/F += F F is between other (two) team(s)
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A.block_.data.SF;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        case SPLIT_NODE_GHOST:
            // Pass data for H/F += F H is between other two processes
            break;
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // Pass data for H/D/F += F F
            // We're either the middle process or both the left and right
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( A.inTargetTeam_ && SFA.rank != 0 && SFB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( SFA.rank, SFB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                  SFA.rank*SFB.rank );
                offsets[A.sourceRoot_] += SFA.rank*SFB.rank;
            }
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            if( SFA.rank != 0 && SFGB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( SFA.rank, SFGB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                  SFA.rank*SFGB.rank );
                offsets[A.sourceRoot_] += SFA.rank*SFGB.rank;
            }
            break;
        }
        case LOW_RANK:
            // Pass data for H/D/F += F F is a send
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for H/D/F += F F
            const LowRankGhost& FGB = *B.block_.data.FG;
            if( SFA.rank != 0 && FGB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( SFA.rank, FGB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                  SFA.rank*FGB.rank );
                offsets[A.sourceRoot_] += SFA.rank*FGB.rank;
            }
            break;
        }
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B.inSourceTeam_ && B.Height() != 0 && SFA.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( B.Height(), SFA.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[B.targetRoot_]],
                  B.Height()*SFA.rank );
                offsets[B.targetRoot_] += B.Height()*SFA.rank;
            }
            break;
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // Pass data for D/F += F D is in other process
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
    {
        const SplitLowRankGhost& SFGA = *A.block_.data.SFG; 
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is between other two processes
            break;
        case SPLIT_DENSE:
            // Pass data for D/F += F D
            if( B.inSourceTeam_ && B.Height() != 0 && SFGA.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( B.Height(), SFGA.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[B.targetRoot_]],
                  B.Height()*SFGA.rank );
                offsets[B.targetRoot_] += B.Height()*SFGA.rank;
            }
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK_GHOST:
    {
        const LowRankGhost& FGA = *A.block_.data.FG;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // Pass data for H/F += F H
            B.TransposeMultiplyDensePassDataUnpack
            ( C.mainContextMap_.Get( key ), recvBuffer, offsets );
            break;
        case SPLIT_LOW_RANK:
            // Pass data for H/D/F += F F is in other process
            break;
        case SPLIT_DENSE:
        {
            // Pass data for D/F += F D
            if( B.Height() != 0 && FGA.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( B.Height(), FGA.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[B.targetRoot_]],
                  B.Height()*FGA.rank );
                offsets[B.targetRoot_] += B.Height()*FGA.rank;
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            // Pass data for D/F += D F
            if( A.inTargetTeam_ )
            {
                const SplitLowRank& SFB = *B.block_.data.SF;
                if( A.Height() != 0 && SFB.rank != 0 )
                {
                    C.ZMap_.Set
                    ( key, new Dense<Scalar>( A.Height(), SFB.rank ) );
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    MemCopy
                    ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                      A.Height()*SFB.rank );
                    offsets[A.sourceRoot_] += A.Height()*SFB.rank;
                }
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            if( A.Height() != 0 && SFGB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( A.Height(), SFGB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                  A.Height()*SFGB.rank );
                offsets[A.sourceRoot_] += A.Height()*SFGB.rank;
            }
            break;
        }
        case LOW_RANK:
            break;
        case LOW_RANK_GHOST:
        {
            // Pass data for D/F += D F
            const LowRankGhost& FGB = *B.block_.data.FG;
            if( A.Height() != 0 && FGB.rank != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( A.Height(), FGB.rank ) );
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                MemCopy
                ( ZC.Buffer(), &recvBuffer[offsets[A.sourceRoot_]],
                  A.Height()*FGB.rank );
                offsets[A.sourceRoot_] += A.Height()*FGB.rank;
            }
            break;
        }
        case SPLIT_DENSE:
        {
            // Pass data for D/F += D D
            const int m = A.Height();    
            const int k = A.Width();
            if( m != 0 && k != 0 )
            {
                if( C.inSourceTeam_ )
                {
                    C.ZMap_.Set( key, new Dense<Scalar>( m, k ) );
                    MemCopy
                    ( C.ZMap_.Get( key ).Buffer(), 
                      &recvBuffer[offsets[B.targetRoot_]], m*k );
                    offsets[B.targetRoot_] += m*k;
                }
            }
            break;
        }
        case SPLIT_DENSE_GHOST:
        case DENSE:
        case DENSE_GHOST:
            // These pass data do not exist or do not involve us
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            break;
        case SPLIT_DENSE:
        {
            const int m = A.Height();
            const int k = A.Width();
            if( m != 0 && k != 0 )
            {
                C.ZMap_.Set( key, new Dense<Scalar>( m, k ) );
                MemCopy
                ( C.ZMap_.Get( key ).Buffer(), 
                  &recvBuffer[offsets[B.targetRoot_]], m*k );
                offsets[B.targetRoot_] += m*k; 
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcasts
( DistHMat2d<Scalar>& B,
  DistHMat2d<Scalar>& C, 
  int startLevel, int endLevel, int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcasts");
#endif
    DistHMat2d<Scalar>& A = *this;

    // Compute the message sizes for each broadcast
    const unsigned numTeamLevels = teams_->NumLevels();
    const unsigned numBroadcasts = numTeamLevels-1;
    std::vector<int> sizes( numBroadcasts, 0 );
    A.MultiplyHMatMainBroadcastsCountA( sizes, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsCountB( sizes, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsCountC
    ( B, C, sizes, startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Pack all of the data to be broadcast into a single buffer
    // (only roots of communicators contribute)
    int totalSize = 0;
    for( unsigned i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( unsigned i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    A.MultiplyHMatMainBroadcastsPackA
    ( buffer, offsetsCopy, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsPackB
    ( buffer, offsetsCopy, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsPackC
    ( B, C, buffer, offsetsCopy, 
      startLevel, endLevel, startUpdate, endUpdate, 0 );

    // Perform the broadcasts with log2(p) messages
    A.teams_->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers
    A.MultiplyHMatMainBroadcastsUnpackA
    ( buffer, offsets, startLevel, endLevel );
    B.MultiplyHMatMainBroadcastsUnpackB
    ( buffer, offsets, startLevel, endLevel );
    A.MultiplyHMatMainBroadcastsUnpackC
    ( B, C, buffer, offsets, startLevel, endLevel, startUpdate, endUpdate, 0 );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsCountA
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsCountA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDenseBroadcastsCount( sizes, rowContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsCountA
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsPackA
( std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsPackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDenseBroadcastsPack
            ( rowContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsPackA
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsUnpackA
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsUnpackA");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganRowSpaceComp_ && !finishedRowSpaceComp_ )
            TransposeMultiplyDenseBroadcastsUnpack
            ( rowContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsUnpackA
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsCountB
( std::vector<int>& sizes, int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsCountB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseBroadcastsCount( sizes, colContext_.numRhs );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsCountB
                    ( sizes, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsPackB
( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
  int startLevel, int endLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsPackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseBroadcastsPack( colContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            const Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsPackB
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsUnpackB
( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsUnpackB");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        if( level_ >= startLevel && level_ < endLevel &&
            beganColSpaceComp_ && !finishedColSpaceComp_ )
            MultiplyDenseBroadcastsUnpack( colContext_, buffer, offsets );

        if( level_+1 < endLevel )
        {
            Node& node = *block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyHMatMainBroadcastsUnpackB
                    ( buffer, offsets, startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsCountC
( const DistHMat2d<Scalar>& B,
  const DistHMat2d<Scalar>& C, 
  std::vector<int>& sizes, 
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsCountC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                const Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsCountC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), sizes,
                              startLevel, endLevel, startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFB = *B.block_.data.DF;
                A.MultiplyDenseBroadcastsCount( sizes, DFB.rank );
            }
            break;
        }
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                B.TransposeMultiplyDenseBroadcastsCount( sizes, DFA.rank );
            }
            break;
        }
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inTargetTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                sizes[teamLevel] += DFA.rank*DFB.rank;
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRankGhost& DFGB = *B.block_.data.DFG;
                const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                sizes[teamLevel] += DFA.rank*DFGB.rank;
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsPackC
( const DistHMat2d<Scalar>& B, 
        DistHMat2d<Scalar>& C,
  std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel,
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsPackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsPackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        }
        case DIST_LOW_RANK:
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseBroadcastsPack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseBroadcastsPack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inTargetTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                mpi::Comm team = teams_->Team( level_ );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 && DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( &buffer[offsets[teamLevel]], 
                      C.ZMap_.Get( key ).LockedBuffer(), DFA.rank*DFB.rank );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRankGhost& DFGB = *B.block_.data.DFG;
                mpi::Comm team = teams_->Team( level_ );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 && DFA.rank != 0 && DFGB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( &buffer[offsets[teamLevel]],
                      C.ZMap_.Get( key ).LockedBuffer(), DFA.rank*DFGB.rank );
                    offsets[teamLevel] += DFA.rank*DFGB.rank;
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainBroadcastsUnpackC
( const DistHMat2d<Scalar>& B,
        DistHMat2d<Scalar>& C,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainBroadcastsUnpackC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    switch( A.block_.type )
    {
    case DIST_NODE:
        switch( B.block_.type )
        {
        case DIST_NODE:
            if( !admissibleC && A.level_+1 < endLevel )
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;
                for( int t=0; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        for( int r=0; r<4; ++r )
                            nodeA.Child(t,r).MultiplyHMatMainBroadcastsUnpackC
                            ( nodeB.Child(r,s), nodeC.Child(t,s), 
                              buffer, offsets, startLevel, endLevel,
                              startUpdate, endUpdate, r );
            }
            break;
        case DIST_LOW_RANK:
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                A.MultiplyDenseBroadcastsUnpack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        default:
            break;
        }
        break;
    case DIST_LOW_RANK:
        switch( B.block_.type )
        {
        case DIST_NODE:
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
                B.TransposeMultiplyDenseBroadcastsUnpack
                ( C.mainContextMap_.Get( key ), buffer, offsets );
            break;
        case DIST_LOW_RANK:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate &&
                A.inTargetTeam_ )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRank& DFB = *B.block_.data.DF;
                if( DFA.rank != 0 && DFB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( C.ZMap_.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFB.rank );
                    offsets[teamLevel] += DFA.rank*DFB.rank;
                }
            }
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            if( A.level_ >= startLevel && A.level_ < endLevel &&
                update >= startUpdate && update < endUpdate )
            {
                const DistLowRank& DFA = *A.block_.data.DF;
                const DistLowRankGhost& DFGB = *B.block_.data.DFG;
                if( DFA.rank != 0 && DFGB.rank != 0 )
                {
                    const unsigned teamLevel = A.teams_->TeamLevel(A.level_);
                    MemCopy
                    ( C.ZMap_.Get( key ).Buffer(), &buffer[offsets[teamLevel]],
                      DFA.rank*DFGB.rank );
                    offsets[teamLevel] += DFA.rank*DFGB.rank;
                }
            }
            break;
        }
        default:
            break;
        }
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPostcompute
( Scalar alpha, DistHMat2d<Scalar>& B,
                DistHMat2d<Scalar>& C,
  int startLevel, int endLevel, int startUpdate, int endUpdate )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPostcompute");
#endif
    DistHMat2d<Scalar>& A = *this;

    A.MultiplyHMatMainPostcomputeA( startLevel, endLevel );
    B.MultiplyHMatMainPostcomputeB( startLevel, endLevel );
    A.MultiplyHMatMainPostcomputeC
    ( alpha, B, C, startLevel, endLevel, startUpdate, endUpdate, 0 );
    C.MultiplyHMatMainPostcomputeCCleanup( startLevel, endLevel );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPostcomputeA
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPostcomputeA");
#endif
    DistHMat2d<Scalar>& A = *this;

    // Handle postcomputation of A's row space
    if( A.level_ >= startLevel && A.level_ < endLevel &&
        A.beganRowSpaceComp_ && !A.finishedRowSpaceComp_ )
    {
        A.AdjointMultiplyDensePostcompute
        ( A.rowContext_, Scalar(1), A.rowOmega_, A.rowT_ );
        A.rowContext_.Clear();
        A.finishedRowSpaceComp_ = true;
    }

    switch( A.block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( A.level_+1 < endLevel )
        {
            Node& nodeA = *A.block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeA.Child(t,s).MultiplyHMatMainPostcomputeA
                    ( startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPostcomputeB
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPostcomputeB");
#endif
    DistHMat2d<Scalar>& B = *this;

    // Handle postcomputation of B's column space
    if( B.level_ >= startLevel && B.level_ < endLevel &&
        B.beganColSpaceComp_ && !B.finishedColSpaceComp_ )
    {
        B.MultiplyDensePostcompute
        ( B.colContext_, Scalar(1), B.colOmega_, B.colT_ );
        B.colContext_.Clear();
        B.finishedColSpaceComp_ = true;
    }

    switch( B.block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        if( B.level_+1 < endLevel )
        {
            Node& nodeB = *B.block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeB.Child(t,s).MultiplyHMatMainPostcomputeB
                    ( startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPostcomputeC
( Scalar alpha, const DistHMat2d<Scalar>& B,
                      DistHMat2d<Scalar>& C,
  int startLevel, int endLevel, 
  int startUpdate, int endUpdate, int update ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPostcomputeC");
#endif
    const DistHMat2d<Scalar>& A = *this;
    if( (!A.inTargetTeam_ && !A.inSourceTeam_ && !B.inSourceTeam_) ||
        A.Height() == 0 || A.Width() == 0 || B.Width() == 0 )
        return;

    // Handle all H H recursion here
    const int key = A.sourceOffset_;
    const bool admissibleC = C.Admissible();
    if( !admissibleC )
    {
        switch( A.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            switch( B.block_.type )
            {
            case DIST_NODE:
            case DIST_NODE_GHOST:
            case SPLIT_NODE:
            case SPLIT_NODE_GHOST:
            case NODE:
            case NODE_GHOST:
            {
                if( A.level_+1 < endLevel )
                {
                    const Node& nodeA = *A.block_.data.N;
                    const Node& nodeB = *B.block_.data.N;
                    Node& nodeC = *C.block_.data.N;
                    for( int t=0; t<4; ++t )
                        for( int s=0; s<4; ++s )
                            for( int r=0; r<4; ++r )
                                nodeA.Child(t,r).MultiplyHMatMainPostcomputeC
                                ( alpha, nodeB.Child(r,s), nodeC.Child(t,s),
                                  startLevel, endLevel,
                                  startUpdate, endUpdate, r );
                }
                return;
                break;
            }
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    if( A.level_ < startLevel || A.level_ >= endLevel ||
        update < startUpdate || update >= endUpdate )
        return;

    // Handle the non-recursive part of the postcompute
    switch( A.block_.type )
    {
    case DIST_NODE:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
        case DIST_NODE_GHOST:
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B.block_.data.DF;
            A.MultiplyDensePostcompute
            ( C.mainContextMap_.Get( key ), 
              alpha, DFB.ULocal, C.UMap_.Get( key ) );
            C.mainContextMap_.Get( key ).Clear();
            C.mainContextMap_.Erase( key );

            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFB.VLocal, C.VMap_.Get( key ) );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B.block_.data.DFG; 
            Dense<Scalar> dummy( 0, DFGB.rank );
            C.UMap_.Set( key, new Dense<Scalar>(LocalHeight(), DFGB.rank ) );
            hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
            A.MultiplyDensePostcompute
            ( C.mainContextMap_.Get( key ), alpha, dummy, C.UMap_.Get( key ) );
            C.mainContextMap_.Get( key ).Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case DIST_NODE:
            break;
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B.block_.data.DF;
            C.VMap_.Set( key, new Dense<Scalar>( B.LocalWidth(), DFB.rank ) );
            hmat_tools::Copy( DFB.VLocal, C.VMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case SPLIT_NODE_GHOST:
        case NODE:
        case NODE_GHOST:
            break;
        case SPLIT_LOW_RANK:
        {
            // We are either the middle process or both the left and right
            const SplitLowRank& SFB = *B.block_.data.SF;
            if( C.inTargetTeam_ )
            {
                C.UMap_.Set( key, new Dense<Scalar>( C.Height(), SFB.rank ) );
                hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
                Dense<Scalar> dummy( 0, SFB.rank );
                A.MultiplyDensePostcompute
                ( C.mainContextMap_.Get( key ), 
                  alpha, dummy, C.UMap_.Get( key ) );

                C.VMap_.Set( key, new Dense<Scalar>( C.Width(), SFB.rank ) );
                hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            }
            C.mainContextMap_.Get( key ).Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            C.UMap_.Set( key, new Dense<Scalar>( C.Height(), SFGB.rank ) );
            hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
            Dense<Scalar> dummy( 0, SFGB.rank );
            A.MultiplyDensePostcompute
            ( C.mainContextMap_.Get( key ), alpha, dummy, C.UMap_.Get( key ) );
            C.mainContextMap_.Get( key ).Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
            const LowRank<Scalar>& FB = *B.block_.data.F;
            C.VMap_.Set( key, new Dense<Scalar>( C.Width(), FB.Rank() ) );
            hmat_tools::Copy( FB.V, C.VMap_.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            const LowRankGhost& FGB = *B.block_.data.FG;
            C.UMap_.Set( key, new Dense<Scalar>( C.Height(), FGB.rank ) );
            hmat_tools::Scale( Scalar(0), C.UMap_.Get( key ) );
            Dense<Scalar> dummy( 0, FGB.rank );
            A.MultiplyDensePostcompute
            ( C.mainContextMap_.Get( key ), alpha, dummy, C.UMap_.Get( key ) );
            C.mainContextMap_.Get( key ).Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case NODE:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        case NODE:
        case SPLIT_LOW_RANK:
            break;
        case LOW_RANK:
        {
            const LowRank<Scalar>& FB = *B.block_.data.F;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C.VMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            break;
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B.block_.data.SF; 
            C.VMap_.Set( key, new Dense<Scalar>( B.LocalWidth(), SFB.rank ) );
            hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DFA = *A.block_.data.DF;
        switch( B.block_.type )
        {
        case DIST_NODE:
        {
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );

            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFA.ULocal, C.UMap_.Get( key ) );
            B.TransposeMultiplyDensePostcompute
            ( context, alpha, DFA.VLocal, C.VMap_.Get( key ) );
            context.Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case DIST_NODE_GHOST:
        {
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFA.ULocal, C.UMap_.Get( key ) );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B.block_.data.DF;
            C.UMap_.Set( key, new Dense<Scalar>( A.LocalHeight(), DFB.rank ) );
            C.VMap_.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            Dense<Scalar>& VC = C.VMap_.Get( key );

            if( A.inTargetTeam_ && DFA.rank != 0 && DFB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', A.LocalHeight(), DFB.rank, DFA.rank,
                  alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                             ZC.LockedBuffer(),         ZC.LDim(),
                  Scalar(0), UC.Buffer(),               UC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            hmat_tools::Copy( DFB.VLocal, VC );
            break;
        }
        case DIST_LOW_RANK_GHOST:
        {
            const DistLowRankGhost& DFGB = *B.block_.data.DFG; 
            C.UMap_.Set( key, new Dense<Scalar>( A.LocalHeight(), DFGB.rank ) );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            
            if( DFA.rank != 0 && DFGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', A.LocalHeight(), DFGB.rank, DFA.rank,
                  alpha,     DFA.ULocal.LockedBuffer(), DFA.ULocal.LDim(),
                             ZC.LockedBuffer(),         ZC.LDim(),
                  Scalar(0), UC.Buffer(),               UC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DIST_LOW_RANK_GHOST:
    {
        const DistLowRankGhost& DFGA = *A.block_.data.DFG;
        switch( B.block_.type )
        {
        case DIST_NODE:
        { 
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            Dense<Scalar> dummy( 0, DFGA.rank );
            C.VMap_.Set( key, new Dense<Scalar>(C.LocalWidth(),DFGA.rank) );
            hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
            B.TransposeMultiplyDensePostcompute
            ( context, alpha, dummy, C.VMap_.Get( key ) );
            context.Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case DIST_LOW_RANK:
        {
            const DistLowRank& DFB = *B.block_.data.DF;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( DFB.VLocal, C.VMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SFA = *A.block_.data.SF;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // We are either the middle process or both the left and the 
            // right. The middle process doesn't have any work left.
            if( A.inTargetTeam_ )
            {
                MultiplyDenseContext& context = C.mainContextMap_.Get( key );
                C.UMap_.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );
                Dense<Scalar> dummy( 0, SFA.rank );
                C.VMap_.Set( key, new Dense<Scalar>(C.LocalWidth(),SFA.rank) );
                hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
                B.TransposeMultiplyDensePostcompute
                ( context, alpha, dummy, C.VMap_.Get( key ) );
                context.Clear();
                C.mainContextMap_.Erase( key );
            }
            break;
        case SPLIT_NODE_GHOST:
            // We are the left process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );
            break;
        case NODE:
            // The precompute is not needed
            break;
        case NODE_GHOST:
            // We are the left process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );
            break;
        case SPLIT_LOW_RANK:
            // We are either the middle process or both the left and right.
            // The middle process is done.
            if( A.inTargetTeam_ )
            {
                const SplitLowRank& SFB = *B.block_.data.SF;
                C.UMap_.Set( key, new Dense<Scalar>( A.Height(), SFB.rank ) );
                Dense<Scalar>& UC = C.UMap_.Get( key );

                if( SFA.rank != 0 && SFB.rank != 0 )
                {
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    blas::Gemm
                    ( 'N', 'N', A.Height(), SFB.rank, SFA.rank,
                      alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                                 ZC.LockedBuffer(),    ZC.LDim(),
                      Scalar(0), UC.Buffer(),          UC.LDim() );
                    ZC.Clear();
                    C.ZMap_.Erase( key );
                }
                else
                    hmat_tools::Scale( Scalar(0), UC );

                C.VMap_.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            C.UMap_.Set( key, new Dense<Scalar>( A.Height(), SFGB.rank ) );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            if( SFA.rank != 0 && SFGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', A.Height(), SFGB.rank, SFA.rank,
                  alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  Scalar(0), UC.Buffer(),          UC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            break;
        }
        case LOW_RANK:
        {
            // We must be the middle and right process
            const LowRank<Scalar>& FB = *B.block_.data.F;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C.VMap_.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We must be the left process
            const LowRankGhost& FGB = *B.block_.data.FG;
            C.UMap_.Set( key, new Dense<Scalar>( A.Height(), FGB.rank ) );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            if( SFA.rank != 0 && FGB.rank != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', A.Height(), FGB.rank, SFA.rank,
                  alpha,     SFA.D.LockedBuffer(), SFA.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  Scalar(0), UC.Buffer(),          UC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            break;
        }
        case SPLIT_DENSE:
            // We are either the middle process or both the left and right
            if( A.inTargetTeam_ )
            {
                C.UMap_.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );

                const SplitDense& SDB = *B.block_.data.SD;
                C.VMap_.Set( key, new Dense<Scalar>( C.Width(), SFA.rank ) );
                Dense<Scalar>& VC = C.VMap_.Get( key );
                if( SFA.rank != 0 && B.Height() != 0 )
                {
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    blas::Gemm
                    ( 'T', 'N', C.Width(), SFA.rank, B.Height(),
                      alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                                 ZC.LockedBuffer(),    ZC.LDim(),
                      Scalar(0), VC.Buffer(),          VC.LDim() );
                    ZC.Clear();
                    C.ZMap_.Erase( key );
                }
                else
                    hmat_tools::Scale( Scalar(0), VC );
            }
            break;
        case SPLIT_DENSE_GHOST:
            // We are the left process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );
            break;
        case DENSE:
            // We are the middle and right process, there is nothing left to do
            break;
        case DENSE_GHOST:
            // We are the left process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFA.D, C.UMap_.Get( key ) );
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_LOW_RANK_GHOST:
    {
        // We are the right process
        const SplitLowRankGhost& SFGA = *A.block_.data.SFG;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            Dense<Scalar> dummy( 0, SFGA.rank );
            C.VMap_.Set( key, new Dense<Scalar>(C.LocalWidth(),SFGA.rank) );
            hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
            B.TransposeMultiplyDensePostcompute
            ( context, alpha, dummy, C.VMap_.Get( key ) );
            context.Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B.block_.data.SF;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B.block_.data.SD;
            C.VMap_.Set( key, new Dense<Scalar>( C.Width(), SFGA.rank ) );
            Dense<Scalar>& VC = C.VMap_.Get( key );
            if( SFGA.rank != 0 && B.Height() != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'T', 'N', C.Width(), SFGA.rank, B.Height(),
                  alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  Scalar(0), VC.Buffer(),          VC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), VC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK:
    {
        const LowRank<Scalar>& FA = *A.block_.data.F;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
            // We are the left and middle process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C.UMap_.Get( key ) );
            break;
        case NODE:
            // We own all of A, B, and C
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C.UMap_.Get( key ) );
            break;
        case SPLIT_LOW_RANK:
        {
            // We are the left and middle process
            const SplitLowRank& SFB = *B.block_.data.SF;
            const int m = A.Height();
            const int k = FA.Rank();
            C.UMap_.Set( key, new Dense<Scalar>( m, SFB.rank ) );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            if( SFB.rank != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', m, SFB.rank, k,
                  alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                             ZC.LockedBuffer(),   ZC.LDim(),
                  Scalar(0), UC.Buffer(),         UC.LDim() );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            break;
        }
        case LOW_RANK:
        {
            // We own all of A, B, and C
            const LowRank<Scalar>& FB = *B.block_.data.F;
            const int m = A.Height();
            const int k = FA.Rank();

            C.UMap_.Set( key, new Dense<Scalar>( m, FB.Rank() ) );
            C.VMap_.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            Dense<Scalar>& VC = C.VMap_.Get( key );
            if( FB.Rank() != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'N', 'N', m, FB.Rank(), k,
                  alpha,     FA.U.LockedBuffer(), FA.U.LDim(),
                             ZC.LockedBuffer(),   ZC.LDim(),
                  Scalar(0), UC.Buffer(),         UC.LDim() );
            }
            else
                hmat_tools::Scale( Scalar(0), UC );
            hmat_tools::Copy( FB.V, VC );
            break;
        }
        case SPLIT_DENSE:
        {
            // We are the left and middle process
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C.UMap_.Get( key ) );
            break;
        }
        case DENSE:
        {
            // We own all of A, B, and C
            C.UMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FA.U, C.UMap_.Get( key ) );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case LOW_RANK_GHOST:
    {
        // We are the right process
        const LowRankGhost& FGA = *A.block_.data.FG;
        switch( B.block_.type )
        {
        case SPLIT_NODE:
        {
            MultiplyDenseContext& context = C.mainContextMap_.Get( key );
            Dense<Scalar> dummy( 0, FGA.rank );
            C.VMap_.Set( key, new Dense<Scalar>(C.LocalWidth(),FGA.rank) );
            hmat_tools::Scale( Scalar(0), C.VMap_.Get( key ) );
            B.TransposeMultiplyDensePostcompute
            ( context, alpha, dummy, C.VMap_.Get( key ) );
            context.Clear();
            C.mainContextMap_.Erase( key );
            break;
        }
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B.block_.data.SF;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B.block_.data.SD;
            C.VMap_.Set( key, new Dense<Scalar>( C.Width(), FGA.rank ) );
            Dense<Scalar>& VC = C.VMap_.Get( key );
            if( FGA.rank != 0 && B.Height() != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                blas::Gemm
                ( 'T', 'N', C.Width(), FGA.rank, B.Height(),
                  alpha,     SDB.D.LockedBuffer(), SDB.D.LDim(),
                             ZC.LockedBuffer(),    ZC.LDim(),
                  Scalar(0), VC.Buffer(),          VC.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            else
                hmat_tools::Scale( Scalar(0), VC );
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
            // We are either the middle process or the left and right

            // TODO: This could be removed by modifying the PassData
            //       unpacking routine to perform this step.
            if( A.inTargetTeam_ )
            {
                const SplitLowRank& SFB = *B.block_.data.SF;
                C.UMap_.Set( key, new Dense<Scalar> ); 
                Dense<Scalar>& UC = C.UMap_.Get( key );
                if( A.Height() != 0 && SFB.rank != 0 )
                    hmat_tools::Copy( C.ZMap_.Get( key ), UC );
                else
                    UC.Resize( A.Height(), SFB.rank ); 

                C.VMap_.Set( key, new Dense<Scalar> );
                hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            }
            break;
        case SPLIT_LOW_RANK_GHOST:
        {
            // We are the left process
            const SplitLowRankGhost& SFGB = *B.block_.data.SFG;
            C.UMap_.Set( key, new Dense<Scalar> );
            Dense<Scalar>& UC = C.UMap_.Get( key );
            if( A.Height() != 0 && SFGB.rank != 0 )
                hmat_tools::Copy( C.ZMap_.Get( key ), UC );
            else
                UC.Resize( A.Height(), SFGB.rank );
            break;
        }
        case LOW_RANK:
        {
            // We are the middle and right process
            const LowRank<Scalar>& FB = *B.block_.data.F;    
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( FB.V, C.VMap_.Get( key ) );
            break;
        }
        case LOW_RANK_GHOST:
        {
            // We are the left process
            const LowRankGhost& FGB = *B.block_.data.FG;
            C.UMap_.Set( key, new Dense<Scalar> ); 
            Dense<Scalar>& UC = C.UMap_.Get( key );
            // TODO: This could be removed by modifying the PassData
            //       unpacking routine to perform this step.
            if( A.Height() != 0 && FGB.rank != 0 )
                hmat_tools::Copy( C.ZMap_.Get( key ), UC );
            else
                UC.Resize( A.Height(), FGB.rank );
            break;
        }
        case SPLIT_DENSE:
            if( C.inSourceTeam_ )
            {
                const SplitDense& SDB = *B.block_.data.SD;
                const int m = C.Height();
                const int n = C.Width();
                const int k = A.Width();
                if( admissibleC )
                {
                    if( C.storedDenseUpdate_ )
                    {
                        if( m != 0 && k != 0 )
                        {
                            Dense<Scalar>& ZC = C.ZMap_.Get( key );
                            blas::Gemm
                            ( 'N', 'N', m, n, k,
                              alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                         SDB.D.LockedBuffer(), SDB.D.LDim(),
                              Scalar(1), C.D_.Buffer(),        C.D_.LDim() );
                            ZC.Clear();
                            C.ZMap_.Erase( key );
                        }
                    }
                    else
                    {
                        C.D_.Resize( m, n );
                        if( m != 0 && k != 0 )
                        {
                            Dense<Scalar>& ZC = C.ZMap_.Get( key );
                            blas::Gemm
                            ( 'N', 'N', m, n, k,
                              alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                         SDB.D.LockedBuffer(), SDB.D.LDim(),
                              Scalar(0), C.D_.Buffer(),        C.D_.LDim() );
                            ZC.Clear();
                            C.ZMap_.Erase( key );
                        }
                        else
                            hmat_tools::Scale( Scalar(0), C.D_ );
                        C.storedDenseUpdate_ = true;
                    }
                }
                else if( m != 0 && k != 0 )
                {
                    Dense<Scalar>& ZC = C.ZMap_.Get( key );
                    Dense<Scalar>& D = *C.block_.data.D;
                    blas::Gemm
                    ( 'N', 'N', m, n, k,
                      alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                 SDB.D.LockedBuffer(), SDB.D.LDim(),
                      Scalar(1), D.Buffer(),           D.LDim() );
                    ZC.Clear();
                    C.ZMap_.Erase( key );
                }
            }
            break;
        case SPLIT_DENSE_GHOST:
            // We are the left process.
            break;
        case DENSE:
            // We are the right process and there is nothing left to do.
            break;
        case DENSE_GHOST:
            // We are the left process.
            break;
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case SPLIT_DENSE_GHOST:
    {
        // We are the right process
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B.block_.data.SF;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B.block_.data.SD;
            const int m = C.Height();
            const int n = C.Width();
            const int k = A.Width();
            if( admissibleC )
            {
                if( C.storedDenseUpdate_ )
                {
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C.ZMap_.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          Scalar(1), C.D_.Buffer(),        C.D_.LDim() );
                        ZC.Clear();
                        C.ZMap_.Erase( key );
                    }
                }
                else
                {
                    C.D_.Resize( m, n );
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C.ZMap_.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          Scalar(0), C.D_.Buffer(),        C.D_.LDim() );
                        ZC.Clear();
                        C.ZMap_.Erase( key );
                    }
                    else
                        hmat_tools::Scale( Scalar(0), C.D_ );
                    C.storedDenseUpdate_ = true;
                }
            }
            else if( m != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                Dense<Scalar>& D = *C.block_.data.D;
                blas::Gemm
                ( 'N', 'N', m, n, k,
                  alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  Scalar(1), D.Buffer(),           D.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case DENSE:
        break;
    case DENSE_GHOST:
    {
        // We are the right process
        switch( B.block_.type )
        {
        case SPLIT_LOW_RANK:
        {
            const SplitLowRank& SFB = *B.block_.data.SF;
            C.VMap_.Set( key, new Dense<Scalar> );
            hmat_tools::Copy( SFB.D, C.VMap_.Get( key ) );
            break;
        }
        case SPLIT_DENSE:
        {
            const SplitDense& SDB = *B.block_.data.SD;
            const int m = C.Height();
            const int n = C.Width();
            const int k = A.Width();
            if( admissibleC )
            {
                if( C.storedDenseUpdate_ )
                {
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C.ZMap_.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          Scalar(1), C.D_.Buffer(),        C.D_.LDim() );
                        ZC.Clear();
                        C.ZMap_.Erase( key );
                    }
                }
                else
                {
                    C.D_.Resize( m, n );
                    if( m != 0 && k != 0 )
                    {
                        Dense<Scalar>& ZC = C.ZMap_.Get( key );
                        blas::Gemm
                        ( 'N', 'N', m, n, k,
                          alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                                     SDB.D.LockedBuffer(), SDB.D.LDim(),
                          Scalar(0), C.D_.Buffer(),        C.D_.LDim() );
                        ZC.Clear();
                        C.ZMap_.Erase( key );
                    }
                    else
                        hmat_tools::Scale( Scalar(0), C.D_ );
                    C.storedDenseUpdate_ = true;
                }
            }
            else if( m != 0 && k != 0 )
            {
                Dense<Scalar>& ZC = C.ZMap_.Get( key );
                Dense<Scalar>& D = *C.block_.data.D;
                blas::Gemm
                ( 'N', 'N', m, n, k,
                  alpha,     ZC.LockedBuffer(),    ZC.LDim(),
                             SDB.D.LockedBuffer(), SDB.D.LDim(),
                  Scalar(1), D.Buffer(),           D.LDim() );
                ZC.Clear();
                C.ZMap_.Erase( key );
            }
            break;
        }
        default:
        {
#ifndef RELEASE
            std::ostringstream s;
            s << "Invalid H-matrix combination: " 
              << BlockTypeString(A.block_.type) << ", "
              << BlockTypeString(B.block_.type) << ", "
              << BlockTypeString(C.block_.type);
            throw std::logic_error( s.str().c_str() );
#endif
            break;
        }
        }
        break;
    }
    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyHMatMainPostcomputeCCleanup
( int startLevel, int endLevel )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyHMatMainPostcomputeCCleanup");
#endif
    DistHMat2d<Scalar>& C = *this;
    C.mainContextMap_.Clear();
    C.ZMap_.Clear();

    switch( C.block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        if( C.level_+1 < endLevel )
        {
            Node& nodeC = *C.block_.data.N;
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    nodeC.Child(t,s).MultiplyHMatMainPostcomputeCCleanup
                    ( startLevel, endLevel );
        }
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
