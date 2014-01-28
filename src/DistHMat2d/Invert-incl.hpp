/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

namespace dmhm {

// A := inv(A) using Schulz iterations, X_k+1 := (2I - X_k A) X_k
template<typename Scalar>
void
DistHMat2d<Scalar>::SchulzInvert
( int numIterations, int multType, BASE(Scalar) theta, BASE(Scalar) confidence )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::SchulzInvert");
    if( Height() != Width() )
        throw std::logic_error("Cannot invert non-square matrices");
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
        ParallelEstimateTwoNorm( theta, confidence );
    const Scalar alpha = Scalar(2)/(estimate*estimate);

    // Initialize X_0 := alpha A^H
    DistHMat2d<Scalar> X;
    X.AdjointFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<numIterations; ++k )
    {
#ifndef RELEASE
        {
            mpi::Comm team = teams_->Team(0);
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
                std::cout << "Iteration: " << k << std::endl;
        }
#endif
        // Form Z := 2I - X_k A
        DistHMat2d<Scalar> Z;
        X.Multiply( Scalar(-1), *this, Z, multType );

		if(stopflag)
        {
            Z.AddConstantToDiagonal( Scalar(1) );
            Scalar estimateZ =
            Z.ParallelEstimateTwoNorm( theta, confidence );
            if( Abs(estimateZ) < 1e-4 )
            {
                Z.AddConstantToDiagonal( Scalar(1) );
                DistHMat2d<Scalar> XCopy;
                XCopy.CopyFrom( X );
                Z.Multiply( Scalar(1), XCopy, X );
                break;
            }
            Z.AddConstantToDiagonal( Scalar(1) );
        }
        else
        	Z.AddConstantToDiagonal( Scalar(2) );
#ifndef RELEASE
        /*
        {
            mpi::Comm team = teams_->Team(0);
            const int teamRank = mpi::CommRank( team );
            const Scalar normestimate =
                Z.ParallelEstimateTwoNorm( theta, confidence );

            if( teamRank == 0 )
                std::cout << "2-Norm of 2I - X_k A:  " << normestimate
                          << std::endl;
        }
        */
#endif
        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        DistHMat2d<Scalar> XCopy;
        XCopy.CopyFrom( X );
        Z.Multiply( Scalar(1), XCopy, X, multType );

#ifndef RELEASE
        {
            mpi::Comm team = teams_->Team(0);
            const int teamRank = mpi::CommRank( team );
            const Scalar normestimate =
                X.ParallelEstimateTwoNorm( theta, confidence );
            if( teamRank == 0 )
                std::cout << "2-Norm of X_{k+1}:  " << normestimate
                          << std::endl;
        }
#endif
    }

    CopyFrom( X );
}

} // namespace dmhm
