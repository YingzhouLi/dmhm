/*
   Distributed-Memory Hierarchical Matrices (DMHM): a prototype implementation
   of distributed-memory H-matrix arithmetic. 

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "dmhm.hpp"

// A := inv(A) using Schulz iterations, X_k+1 := (2I - X_k A) X_k
template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::SchulzInvert
( int numIterations, 
  typename RealBase<Scalar>::type theta, 
  typename RealBase<Scalar>::type confidence )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::SchulzInvert");
    if( Height() != Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( theta <= 1 )
        throw std::logic_error("Theta must be > 1");
    if( confidence <= 0 )
        throw std::logic_error("Confidence must be positive");
#endif
    if( numIterations <= 0 )
        throw std::logic_error("Must use at least 1 iteration.");

    const Scalar estimate = 
        ParallelEstimateTwoNorm( theta, confidence );
    const Scalar alpha = ((Scalar)2) / (estimate*estimate);

    // Initialize X_0 := alpha A^H
    DistQuasi2dHMat<Scalar,Conjugated> X;
    X.AdjointFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<numIterations; ++k )
    {
        // Form Z := 2I - X_k A
        DistQuasi2dHMat<Scalar,Conjugated> Z;
        X.Multiply( (Scalar)-1, *this, Z, 2 );
        Z.AddConstantToDiagonal( (Scalar)2 );

        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        DistQuasi2dHMat<Scalar,Conjugated> XCopy;
        XCopy.CopyFrom( X );
        Z.Multiply( (Scalar)1, XCopy, X, 2 );
    }

    CopyFrom( X );
#ifndef RELEASE
    PopCallStack();
#endif
}

