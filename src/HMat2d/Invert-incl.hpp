/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

#include<string>
#include<fstream>
#include<sstream>

namespace dmhm {

// A := inv(A) using recursive Schur complements
template<typename Scalar>
void
HMat2d<Scalar>::DirectInvert()
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::DirectInvert");
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
        HMat2d<Scalar> B;
        B.CopyFrom( *this );
// Initialize our soon-to-be inverse as the identity
        SetToIdentity();

        Node& nodeA = *block_.data.N;
        Node& nodeB = *B.block_.data.N;

        for( int l=0; l<4; ++l )
        {
            // A_ll := inv(B_ll)
            nodeA.Child(l,l).CopyFrom( nodeB.Child(l,l) );
            nodeA.Child(l,l).DirectInvert();

            // NOTE: Can be skipped for upper-triangular matrices
            for( int j=0; j<l; ++j )
            {
                // A_lj := A_ll A_lj
                HMat2d<Scalar> C;
                C.CopyFrom( nodeA.Child(l,j) );
                nodeA.Child(l,l).Multiply( Scalar(1), C, nodeA.Child(l,j) );
            }

            // NOTE: Can be skipped for lower-triangular matrices
            for( int j=l+1; j<4; ++j )
            {
                // B_lj := A_ll B_lj
                HMat2d<Scalar> C;
                C.CopyFrom( nodeB.Child(l,j) );
                nodeA.Child(l,l).Multiply( Scalar(1), C, nodeB.Child(l,j) );
            }

            for( int i=l+1; i<4; ++i )
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
                for( int j=l+1; j<4; ++j )
                {
                    // B_ij -= B_il B_lj
                    nodeB.Child(i,l).Multiply
                    ( Scalar(-1), nodeB.Child(l,j),
                      Scalar(1),  nodeB.Child(i,j) );
                }
            }
        }

        // NOTE: Can be skipped for lower-triangular matrices.
        for( int l=3; l>=0; --l )
        {
            for( int i=l-1; i>=0; --i )
            {
                // NOTE: For upper-triangular matrices, change the loop to
                //       for( int j=l; j<4; ++j )
                for( int j=0; j<4; ++j )
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
HMat2d<Scalar>::SchulzInvert
( int numIterations, Real theta, Real confidence, Real stopTol )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::SchulzInvert");
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
    HMat2d<Scalar> X;
    X.AdjointFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<numIterations; ++k )
    {
        // Form Z := 2I - X_k A
        HMat2d<Scalar> Z;
        X.Multiply( Scalar(-1), *this, Z );

        if(stopflag)
        {
            Z.AddConstantToDiagonal( Scalar(1) );
            Scalar estimateZ =
            hmat_tools::EstimateTwoNorm( Z, theta, confidence );
#ifndef RELEASE
            std::cout << "Iteration: " << k << std::endl;
            std::cout << "||I-X_k A||_2: " << Abs(estimateZ) << std::endl;
#endif
            if( Abs(estimateZ) < stopTol )
            {
                Z.AddConstantToDiagonal( Scalar(1) );
                HMat2d<Scalar> XCopy;
                XCopy.CopyFrom( X );
                Z.Multiply( Scalar(1), XCopy, X );
                break;
            }
            Z.AddConstantToDiagonal( Scalar(1) );
        }
        else
            Z.AddConstantToDiagonal( Scalar(2) );

        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        HMat2d<Scalar> XCopy;
        XCopy.CopyFrom( X );
        Z.Multiply( Scalar(1), XCopy, X );

    }

    CopyFrom( X );
}

} // namespace dmhm
