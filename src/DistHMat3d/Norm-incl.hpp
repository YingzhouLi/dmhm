/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

template<typename Scalar>
BASE(Scalar)
DistHMat3d<Scalar>::ParallelEstimateTwoNorm( Real theta, Real confidence )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::ParallelEstimateTwoNorm");
    if( theta <= 1 )
        throw std::logic_error("Theta must be > 1");
    if( confidence <= 0 )
        throw std::logic_error("Confidence must be positive.");
#endif
    const int n = Height();
    const int k = ceil(log(0.8*sqrt(n)*pow(10,confidence))/log(theta));
    mpi::Comm team = teams_->Team( 0 );
#ifndef RELEASE
    {
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            std::cout << "Going to use A^" << k
                      << " in order to estimate "
                      << "||A||_2 within " << (theta-1.0)*100
                      << "% with probability "
                      << "1-10^{-" << confidence << "}" << std::endl;
        }
    }
#endif
    // Sample the unit sphere
    Vector<Scalar> x( n );
    {
        ParallelGaussianRandomVector( x );
        const Real LocaltwoNormSqr = pow( hmat_tools::TwoNorm( x ), 2 );
        Real twoNorm;
        mpi::AllReduce(&LocaltwoNormSqr, &twoNorm, 1, mpi::SUM, team);
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
        mpi::AllReduce(&LocaltwoNormSqr, &twoNorm, 1, mpi::SUM, team);
        twoNorm = sqrt(twoNorm);
        hmat_tools::Scale( ((Scalar)1)/twoNorm, x );
        estimate *= pow( twoNorm, root );
    }
#ifndef RELEASE
    {
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
            std::cout << "Estimated ||A||_2 as " << estimate << std::endl;
    }
#endif
    return estimate;
}

} // namespace dmhm
