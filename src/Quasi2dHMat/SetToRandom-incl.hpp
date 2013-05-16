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
Quasi2dHMat<Scalar>::SetToRandom()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::SetToRandom");
#endif
    switch( _block.type )
    {
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).SetToRandom();
        break;
    }
    case NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetry not yet supported.");
#endif
        break;
    }
    case DENSE:
        ParallelGaussianRandomVectors( *_block.data.D );
        break;
    case LOW_RANK:
    {
        LowRank<Scalar>& F = *_block.data.F;
        const int maxRank = MaxRank();
        const int height = F.U.Height();
        const int width = F.V.Height();

        F.U.Resize( height, maxRank );
        F.V.Resize( width,  maxRank );
        ParallelGaussianRandomVectors( F.U );
        ParallelGaussianRandomVectors( F.V );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
