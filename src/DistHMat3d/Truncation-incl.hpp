/*
   Copyright (c) 2011-2013 Jack Poulson, Yingzhou Li, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

template<typename Scalar>
void
DistHMat3d<Scalar>::EVDTrunc
( Dense<Scalar>& Q, Vector<Real>& w, Real relTol )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::EVDTrunc");
#endif
    const int k = w.Size();
    if( k == 0 )
        return;

    const Real maxEig = w[k-1];
    const Real tolerance = relTol*maxEig;
    int cutoff;
    for( cutoff=0; cutoff<k; ++cutoff )
        if( w[cutoff] > tolerance )
            break;
    if( cutoff == k )
        cutoff = k-1;

    w.Erase( w.Begin(), w.Begin()+cutoff );
    Q.EraseCols( 0, cutoff-1 );
}

template<typename Scalar>
void
DistHMat3d<Scalar>::SVDTrunc
( Dense<Scalar>& U, Vector<Real>& s,
  Dense<Scalar>& VH, Real relTol, Real twoNorm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::SVDTrunc");
#endif
    const int m = U.Width();
    const int n = VH.Height();
    const int k = s.Size();
    if( k == 0 )
        return;

    const Real tolerance = relTol*twoNorm;
    int cutoff;
    for( cutoff=std::min( k, MaxRank() )-1; cutoff>=0; --cutoff )
        if( s[cutoff] > tolerance )
            break;
    s.Erase( s.Begin()+cutoff+1, s.Begin()+k );
    U.EraseCols( cutoff+1, m-1 );
    VH.EraseRows( cutoff+1, n-1 );
}

} // namespace dmhm
