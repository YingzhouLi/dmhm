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

template<typename Scalar,bool Conjugated>
void
dmhm::DistQuasi2dHMat<Scalar,Conjugated>::AddConstantToDiagonal
( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AddConstantToDiagonal");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            _block.data.N->Child(t,t).AddConstantToDiagonal( alpha );
        break;
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
        {
            SplitDense& SD = *_block.data.SD;
            Scalar* DBuffer = SD.D.Buffer();
            const int m = SD.D.Height();
            const int n = SD.D.Width();
            const int DLDim = SD.D.LDim();
            for( int j=0; j<std::min(m,n); ++j )
                DBuffer[j+j*DLDim] += alpha;
        }
        break;
    case DENSE:
        {
            Scalar* DBuffer = _block.data.D->Buffer();
            const int m = _block.data.D->Height();
            const int n = _block.data.D->Width();
            const int DLDim = _block.data.D->LDim();
            for( int j=0; j<std::min(m,n); ++j )
                DBuffer[j+j*DLDim] += alpha;
        }
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}
