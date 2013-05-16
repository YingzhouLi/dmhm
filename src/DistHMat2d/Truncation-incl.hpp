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
DistHMat2d<Scalar>::EVDTrunc
( Dense<Scalar>& Q, std::vector<Real>& w, Real error )
{
    int ldq=Q.LDim();
#ifndef RELEASE
    PushCallStack("DistHMat2d::EVDTrunc");
    if( ldq ==0 )
        throw std::logic_error("ldq was 0");
#endif
    int L;
    for(L=0; L<ldq; ++L )
        if( w[L]>error*w[ldq-1])
            break;

    w.erase( w.begin(), w.begin()+L );
    Q.EraseCol(0, L-1);

#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar>
void
DistHMat2d<Scalar>::SVDTrunc
( Dense<Scalar>& U, std::vector<Real>& w,
  Dense<Scalar>& VH, Real error )
{
    int ldq=w.size();
#ifndef RELEASE
    PushCallStack("DistHMat2d::EVDTrunc");
    if( ldq ==0 )
        throw std::logic_error("ldq was 0");
#endif
    int ldu=U.LDim();
    int ldvh=VH.LDim();
    int k=ldvh;
    int L;
    for(L=std::min(k,maxRank_)-1; L>=0; --L )
        if( w[L]>error*w[0]*w.size() )
            break;
    w.resize(L+1);
    U.Resize(ldu, L+1, ldu);
    VH.EraseRow(L+1, ldvh-1);

#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
