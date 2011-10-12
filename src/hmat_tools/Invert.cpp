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

template<typename Scalar>
void dmhm::hmat_tools::Invert
( Dense<Scalar>& D )
{
#ifndef RELEASE
    PushCallStack("hmat_tools::Invert");
    if( D.Height() != D.Width() )
        throw std::logic_error("Tried to invert a non-square dense matrix.");
#endif
    const int n = D.Height();
    const int lworkLDLT = lapack::LDLTWorkSize( n );
    const int lworkInvertLDLT = lapack::InvertLDLTWorkSize( n );
    const int lwork = std::max( lworkLDLT, lworkInvertLDLT );
    std::vector<int> ipiv( n );
    std::vector<Scalar> work( lwork );
    if( D.Symmetric() )
    {
        lapack::LDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
        lapack::InvertLDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0] );
    }
    else
    {
        lapack::LU( n, n, D.Buffer(), D.LDim(), &ipiv[0] );
        lapack::InvertLU( n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void dmhm::hmat_tools::Invert( Dense<float>& D );
template void dmhm::hmat_tools::Invert( Dense<double>& D );
template void dmhm::hmat_tools::Invert( Dense<std::complex<float> >& D );
template void dmhm::hmat_tools::Invert( Dense<std::complex<double> >& D );
