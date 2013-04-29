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

template<typename Scalar, bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar, Conjugated>::EVDTrunc
( Dense<Scalar>& Q, std::vector<Real>& w, Real error )
{
    int ldq=Q.LDim();
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::EVDTrunc");
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


template<typename Scalar, bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar, Conjugated>::SVDTrunc
( Dense<Scalar>& U, std::vector<Real>& w,
  Dense<Scalar>& VH, Real error )
{
    int ldq=w.size();
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::EVDTrunc");
    if( ldq ==0 )
        throw std::logic_error("ldq was 0");
#endif
    int ldu=U.LDim();
    int ldvh=VH.LDim();
    int k=ldvh;
    int L;
    for(L=k-1; L>=0; --L )
        if( w[L]>error*w[0])
            break;

    w.resize(L+1);
    U.Resize(ldu, L+1, ldu);
    VH.EraseRow(L+1, ldvh-1);

#ifndef RELEASE
    PopCallStack();
#endif
}
