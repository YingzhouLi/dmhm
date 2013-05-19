/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

// C := alpha A B
template<typename Scalar>
void
HMat3d<Scalar>::Multiply
( Scalar alpha, const HMat3d<Scalar>& B,
                      HMat3d<Scalar>& C ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat3d::Multiply H := H H");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( numLevels_ != B.numLevels_ )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
#endif
    const HMat3d<Scalar>& A = *this;

    C.numLevels_ = A.numLevels_;
    C.maxRank_ = A.maxRank_;
    C.targetOffset_ = A.targetOffset_;
    C.sourceOffset_ = B.sourceOffset_;
    C.symmetric_ = false;
    C.stronglyAdmissible_ = ( A.stronglyAdmissible_ || B.stronglyAdmissible_ );

    C.xSizeTarget_ = A.xSizeTarget_;
    C.ySizeTarget_ = A.ySizeTarget_;
    C.zSizeTarget_ = A.zSizeTarget_;
    C.xSizeSource_ = B.xSizeSource_;
    C.ySizeSource_ = B.ySizeSource_;
    C.zSizeSource_ = B.zSizeSource_;
    C.xTarget_ = A.xTarget_;
    C.yTarget_ = A.yTarget_;
    C.zTarget_ = A.zTarget_;
    C.xSource_ = B.xSource_;
    C.ySource_ = B.ySource_;
    C.zSource_ = B.zSource_;

    C.block_.Clear();
    if( C.Admissible() )
    {
        C.block_.type = LOW_RANK;
        C.block_.data.F = new LowRank<Scalar>;
        if( A.IsLowRank() && B.IsLowRank() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, *C.block_.data.F );
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            hmat_tools::Copy( A.block_.data.F->U, C.block_.data.F->U );
            B.TransposeMultiply
            ( alpha, A.block_.data.F->V, C.block_.data.F->V );
        }
        else if( A.IsLowRank() && B.IsDense() )
        {
            hmat_tools::Copy( A.block_.data.F->U, C.block_.data.F->U );
            hmat_tools::TransposeMultiply
            ( alpha, *B.block_.data.D, A.block_.data.F->V, C.block_.data.F->V );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            Multiply( alpha, B.block_.data.F->U, C.block_.data.F->U );
            // C.F.V := B.F.V
            hmat_tools::Copy( B.block_.data.F->V, C.block_.data.F->V );
        }
        else if( A.IsHierarchical() && B.IsHierarchical() )
        {
            // C.F := alpha H H
            const int sampleRank = SampleRank( C.MaxRank() );
            hmat_tools::Multiply
            ( sampleRank, alpha, *this, B, *C.block_.data.F );
        }
        else if( A.IsDense() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, 
              B.block_.data.F->U, C.block_.data.F->U );
            // C.F.V := B.F.V
            hmat_tools::Copy( B.block_.data.F->V, C.block_.data.F->V );
        }
        else if( A.IsDense() && B.IsDense() )
            hmat_tools::Multiply
            ( C.MaxRank(),
              alpha, *A.block_.data.D, *B.block_.data.D, *C.block_.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination");
#endif
    }
    else if( C.NumLevels() > 1 )
    {
        // A product of two matrices will be assumed non-symmetric.
        C.block_.type = NODE;
        C.block_.data.N = C.NewNode();

#ifndef RELEASE
        if( A.IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B
            LowRank<Scalar> W;
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, W );

            // Form C :~= W
            C.ImportLowRank( W );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRank<Scalar> W;
            hmat_tools::Copy( A.block_.data.F->U, W.U );
            B.TransposeMultiply( alpha, A.block_.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRank( W );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRank<Scalar> W;
            Multiply( alpha, B.block_.data.F->U, W.U );
            hmat_tools::Copy( B.block_.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRank( W );
        }
        else
        {
#ifndef RELEASE
            if( A.Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
#endif
            const Node& nodeA = *A.block_.data.N;
            const Node& nodeB = *B.block_.data.N;
            Node& nodeC = *C.block_.data.N;

            for( int t=0; t<8; ++t )
            {
                for( int s=0; s<8; ++s )
                {
                    // Create the H-matrix here
                    nodeC.children[s+8*t] = new HMat3d<Scalar>;

                    // Initialize the [t,s] box of C with the first product
                    nodeA.Child(t,0).Multiply
                    ( alpha, nodeB.Child(0,s), nodeC.Child(t,s) );

                    // Add the other three products onto it
                    for( int u=1; u<8; ++u )
                        nodeA.Child(t,u).Multiply
                        ( alpha, nodeB.Child(u,s), 1, nodeC.Child(t,s) );
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( A.IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        C.block_.type = DENSE;
        C.block_.data.D = new Dense<Scalar>;

        if( A.IsDense() && B.IsDense() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, *B.block_.data.D, *C.block_.data.D );
        else if( A.IsDense() && B.IsLowRank() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, *B.block_.data.F, *C.block_.data.D );
        else if( A.IsLowRank() && B.IsDense() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.D, *C.block_.data.D );
        else /* both low-rank */
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, *C.block_.data.D );
    }
}

// C := alpha A B + beta C
template<typename Scalar>
void
HMat3d<Scalar>::Multiply
( Scalar alpha, const HMat3d<Scalar>& B,
  Scalar beta,        HMat3d<Scalar>& C ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat3d::Multiply (H := H H + H)");
    if( Width() != B.Height() || 
        Height() != C.Height() || B.Width() != C.Width() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( NumLevels() != B.NumLevels() || 
        NumLevels() != C.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
    if( C.Symmetric() )
        throw std::logic_error("Symmetric updates not yet supported.");
#endif
    const HMat3d<Scalar>& A = *this;
    if( C.Admissible() )
    {
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // W := alpha A.F B.F
            LowRank<Scalar> W;
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, W );

            // C.F :~= W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // W := alpha A.F B
            LowRank<Scalar> W;
            hmat_tools::Copy( A.block_.data.F->U, W.U );
            B.TransposeMultiply( alpha, A.block_.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsLowRank() && B.IsDense() )
        {
            // W := alpha A.F B.D
            LowRank<Scalar> W;
            hmat_tools::Copy( A.block_.data.F->U, W.U );
            hmat_tools::TransposeMultiply
            ( alpha, *B.block_.data.D, A.block_.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // W := alpha A B.F
            LowRank<Scalar> W;
            Multiply( alpha, B.block_.data.F->U, W.U );
            hmat_tools::Copy( B.block_.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsHierarchical() && B.IsHierarchical() )
        {
            // W := alpha A B
            LowRank<Scalar> W;
            const int sampleRank = SampleRank( C.MaxRank() );
            hmat_tools::Multiply( sampleRank, alpha, *this, B, W );

            // C.F :~= W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsDense() && B.IsLowRank() )
        {
            // W := alpha A.D B.F
            LowRank<Scalar> W;
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, B.block_.data.F->U, W.U );
            hmat_tools::Copy( B.block_.data.F->V, W.V );

            // C.F :=~ W + beta C.F
            hmat_tools::RoundedUpdate
            ( C.MaxRank(), Scalar(1), W, beta, *C.block_.data.F );
        }
        else if( A.IsDense() && B.IsDense() )
            hmat_tools::Multiply
            ( C.MaxRank(),
              alpha, *A.block_.data.D, *B.block_.data.D, 
              beta, *C.block_.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination.");
#endif
    }
    else if( C.NumLevels() > 1 )
    {

#ifndef RELEASE
        if( A.IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B 
            LowRank<Scalar> W;
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, W );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRank( Scalar(1), W );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRank<Scalar> W;
            hmat_tools::Copy( A.block_.data.F->U, W.U );
            B.TransposeMultiply( alpha, A.block_.data.F->V, W.V );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRank( Scalar(1), W );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRank<Scalar> W;
            Multiply( alpha, B.block_.data.F->U, W.U );
            hmat_tools::Copy( B.block_.data.F->V, W.V );

            // Form C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRank( Scalar(1), W );
        }
        else
        {
            if( A.Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
            else 
            {
                const Node& nodeA = *A.block_.data.N;
                const Node& nodeB = *B.block_.data.N;
                Node& nodeC = *C.block_.data.N;

                for( int t=0; t<8; ++t )
                {
                    for( int s=0; s<8; ++s )
                    {
                        // Scale the [t,s] box of C in the first product
                        nodeA.Child(t,0).Multiply
                        ( alpha, nodeB.Child(0,s), beta, nodeC.Child(t,s) ); 
        
                        // Add the other three products onto it
                        for( int u=1; u<8; ++u )
                            nodeA.Child(t,u).Multiply
                            ( alpha, nodeB.Child(u,s), 
                              Scalar(1), nodeC.Child(t,s) ); 
                    }
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( A.IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        if( A.IsDense() && B.IsDense() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, *B.block_.data.D, 
              beta, *C.block_.data.D );
        else if( A.IsDense() && B.IsLowRank() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.D, *B.block_.data.F, 
              beta, *C.block_.data.D );
        else if( A.IsLowRank() && B.IsDense() )
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.D, 
              beta, *C.block_.data.D );
        else /* both low-rank */
            hmat_tools::Multiply
            ( alpha, *A.block_.data.F, *B.block_.data.F, 
              beta, *C.block_.data.D );
    }
}

} // namespace dmhm
