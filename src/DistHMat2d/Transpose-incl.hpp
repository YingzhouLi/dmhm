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
DistHMat2d<Scalar>::Transpose()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Transpose");
#endif
    // This requires communication and is not yet written
    throw std::logic_error("DistHMat2d::Transpose is not yet written");
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeFrom
( const DistHMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeFrom");
#endif
    DistHMat2d<Scalar>& A = *this;

    A.numLevels_ = B.numLevels_;
    A.maxRank_ = B.maxRank_;
    A.targetOffset_ = B.targetOffset_;
    A.sourceOffset_ = B.sourceOffset_;
    A.stronglyAdmissible_ = B.stronglyAdmissible_;

    A.xSizeTarget_ = B.xSizeTarget_;
    A.ySizeTarget_ = B.ySizeTarget_;
    A.xSizeSource_ = B.xSizeSource_;
    A.ySizeSource_ = B.ySizeSource_;

    A.xTarget_ = B.xTarget_;
    A.yTarget_ = B.yTarget_;
    A.xSource_ = B.xSource_;
    A.ySource_ = B.ySource_;

    A.teams_ = B.teams_;
    A.level_ = B.level_;
    A.inTargetTeam_ = B.inTargetTeam_;
    A.inSourceTeam_ = B.inSourceTeam_;
    A.targetRoot_ = B.targetRoot_;
    A.sourceRoot_ = B.sourceRoot_;

    A.block_.Clear();
    A.block_.type = B.block_.type;

    // This requires communication and is not yet written
    throw std::logic_error("DistHMat2d::TransposeFrom is not yet written");
}

} // namespace dmhm
