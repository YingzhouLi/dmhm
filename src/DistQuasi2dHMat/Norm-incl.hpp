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

template<typename Scalar, bool Conjugated>
REALBASE(Scalar)
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::ParallelEstimateTwoNorm
( Real theta, Real confidence )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::ParallelEstimateTwoNorm");
    if( theta <= 1 )
        throw std::logic_error("Theta must be > 1");
    if( confidence <= 0 )
        throw std::logic_error("Confidence must be positive.");
#endif
    const int n = LocalHeight();
    const int k = ceil(log(0.8*sqrt(n)*pow(10,confidence))/log(theta));
    MPI_Comm team = _teams->Team( 0 );
#ifndef RELEASE
    std::cerr << "Going to use A^" << k  << " in order to estimate "
              << "||A||_2 within " << (theta-1.0)*100 << "% with probability "
              << "1-10^{-" << confidence << "}" << std::endl;
#endif
    // Sample the unit sphere
    Vector<Scalar> x( n );
    {
        ParallelGaussianRandomVector( x );
        const Real LocaltwoNormSqr = pow( hmat_tools::TwoNorm( x ), 2 );
        Real twoNorm;
        mpi::AllReduce(&LocaltwoNormSqr, &twoNorm, 1, MPI_SUM, team);
        twoNorm = sqrt(twoNorm);
        hmat_tools::Scale( ((Scalar)1)/twoNorm, x );
    }

    Real estimate = theta; 
    const Real root = ((Real)1) / ((Real)k);
    Vector<Scalar> y;
    for( int i=0; i<k; ++i )
    {
        Multiply( (Scalar)1, x, y );
        hmat_tools::Copy( y, x );
        const Real LocaltwoNormSqr = pow( hmat_tools::TwoNorm( x ), 2 );
        Real twoNorm;
        mpi::AllReduce(&LocaltwoNormSqr, &twoNorm, 1, MPI_SUM, team);
        twoNorm = sqrt(twoNorm);
        hmat_tools::Scale( ((Scalar)1)/twoNorm, x );
        estimate *= pow( twoNorm, root );
    }
#ifndef RELEASE
    std::cerr << "Estimated ||A||_2 as " << estimate << std::endl;
    PopCallStack();
#endif
    return estimate;
}
