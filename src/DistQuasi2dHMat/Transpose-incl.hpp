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
DistQuasi2dHMat<Scalar>::Transpose()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Transpose");
#endif
    // This requires communication and is not yet written
    throw std::logic_error("DistQuasi2dHMat::Transpose is not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
DistQuasi2dHMat<Scalar>::TransposeFrom
( const DistQuasi2dHMat<Scalar>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::TransposeFrom");
#endif
    DistQuasi2dHMat<Scalar>& A = *this;

    A._numLevels = B._numLevels;
    A._maxRank = B._maxRank;
    A._targetOffset = B._targetOffset;
    A._sourceOffset = B._sourceOffset;
    A._stronglyAdmissible = B._stronglyAdmissible;

    A._xSizeTarget = B._xSizeTarget;
    A._ySizeTarget = B._ySizeTarget;
    A._xSizeSource = B._xSizeSource;
    A._ySizeSource = B._ySizeSource;
    A._zSize = B._zSize;

    A._xTarget = B._xTarget;
    A._yTarget = B._yTarget;
    A._xSource = B._xSource;
    A._ySource = B._ySource;

    A._teams = B._teams;
    A._level = B._level;
    A._inTargetTeam = B._inTargetTeam;
    A._inSourceTeam = B._inSourceTeam;
    A._targetRoot = B._targetRoot;
    A._sourceRoot = B._sourceRoot;

    A._block.Clear();
    A._block.type = B._block.type;

    // This requires communication and is not yet written
    throw std::logic_error("DistQuasi2dHMat::TransposeFrom is not yet written");

#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
