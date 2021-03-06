/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

// A := inv(A) using recursive Schur complements
template<typename Scalar>
void
HMat3d<Scalar>::DirectInvert()
{
#ifndef RELEASE
    CallStackEntry entry("HMat3d::DirectInvert");
    if( Height() != Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
#endif
    switch( block_.type )
    {
    case NODE:
    {
        // We will form the inverse in the original matrix, so we only need to
        // create a temporary matrix.
        HMat3d<Scalar> B;
        B.CopyFrom( *this );

        // Initialize our soon-to-be inverse as the identity
        SetToIdentity();

        Node& nodeA = *block_.data.N;
        Node& nodeB = *B.block_.data.N;

        for( int l=0; l<8; ++l )
        {
            // A_ll := inv(B_ll)
            nodeA.Child(l,l).CopyFrom( nodeB.Child(l,l) );
            nodeA.Child(l,l).DirectInvert();

            // NOTE: Can be skipped for upper-triangular matrices
            for( int j=0; j<l; ++j )
            {
                // A_lj := A_ll A_lj
                HMat3d<Scalar> C;
                C.CopyFrom( nodeA.Child(l,j) );
                nodeA.Child(l,l).Multiply( Scalar(1), C, nodeA.Child(l,j) );
            }

            // NOTE: Can be skipped for lower-triangular matrices
            for( int j=l+1; j<8; ++j )
            {
                // B_lj := A_ll B_lj
                HMat3d<Scalar> C;
                C.CopyFrom( nodeB.Child(l,j) );
                nodeA.Child(l,l).Multiply( Scalar(1), C, nodeB.Child(l,j) );
            }

            for( int i=l+1; i<8; ++i )
            {
                // NOTE: Can be skipped for upper triangular matrices.
                for( int j=0; j<=l; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).Multiply
                    ( Scalar(-1), nodeA.Child(l,j),
                      Scalar(1),  nodeA.Child(i,j) );
                }
                // NOTE: Can be skipped for either lower or upper-triangular
                //       matrices, effectively decoupling the diagonal block
                //       inversions.
                for( int j=l+1; j<8; ++j )
                {
                    // B_ij -= B_il B_lj
                    nodeB.Child(i,l).Multiply
                    ( Scalar(-1), nodeB.Child(l,j),
                      Scalar(1),  nodeB.Child(i,j) );
                }
            }
        }

        // NOTE: Can be skipped for lower-triangular matrices.
        for( int l=7; l>=0; --l )
        {
            for( int i=l-1; i>=0; --i )
            {
                // NOTE: For upper-triangular matrices, change the loop to
                //       for( int j=l; j<8; ++j )
                for( int j=0; j<8; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).Multiply
                    ( Scalar(-1), nodeA.Child(l,j),
                      Scalar(1),  nodeA.Child(i,j) );
                }
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric inversion not yet supported.");
#endif
        break;
    }
    case DENSE:
        hmat_tools::Invert( *block_.data.D );
        break;
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Mistake in inversion code.");
#endif
        break;
    }
    }
}

// A := inv(A) using Schulz iterations, X_k+1 := (2I - X_k A) X_k
template<typename Scalar>
void
HMat3d<Scalar>::SchulzInvert
( int numIterations, BASE(Scalar) theta, BASE(Scalar) confidence )
{
#ifndef RELEASE
    CallStackEntry entry("HMat3d::SchulzInvert");
    if( Height() != Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
    if( theta <= 1 )
        throw std::logic_error("Theta must be > 1");
    if( confidence <= 0 )
        throw std::logic_error("Confidence must be positive");
#endif
    bool stopflag = false;
    if( numIterations <= 0 )
    {
        numIterations = 30;
        stopflag = true;
    }

    const Scalar estimate =
        hmat_tools::EstimateTwoNorm( *this, theta, confidence );
    const Scalar alpha = Scalar(2)/(estimate*estimate);

    // Initialize X_0 := alpha A^H
    HMat3d<Scalar> X;
    X.AdjointFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<numIterations; ++k )
    {
        // Form Z := 2I - X_k A
        HMat3d<Scalar> Z;
        X.Multiply( Scalar(-1), *this, Z );

        if(stopflag)
        {
            Z.AddConstantToDiagonal( Scalar(1) );
            Scalar estimateZ =
            hmat_tools::EstimateTwoNorm( Z, theta, confidence );
            if( Abs(estimateZ) < 1e-4 )
            {
                Z.AddConstantToDiagonal( Scalar(1) );
                HMat3d<Scalar> XCopy;
                XCopy.CopyFrom( X );
                Z.Multiply( Scalar(1), XCopy, X );
                break;
            }
            Z.AddConstantToDiagonal( Scalar(1) );
        }
        else
            Z.AddConstantToDiagonal( Scalar(2) );

        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        HMat3d<Scalar> XCopy;
        XCopy.CopyFrom( X );
        Z.Multiply( Scalar(1), XCopy, X );
    }

    CopyFrom( X );
}

} // namespace dmhm
