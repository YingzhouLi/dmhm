/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_ABSTRACT_HMAT_HPP
#define DMHM_ABSTRACT_HMAT_HPP 1

#include "dmhm/core/dense.hpp"
#include "dmhm/core/vector.hpp"

#include "dmhm/core/low_rank.hpp"

namespace dmhm {

template<typename Scalar>
class AbstractHMat
{
public:
    /*
     * Public virtual member functions
     */
    virtual int Height() const = 0;
    virtual int Width() const = 0;
    virtual int NumLevels() const = 0;
    virtual int MaxRank() const = 0;
    virtual int SourceOffset() const = 0;
    virtual int TargetOffset() const = 0;
    virtual bool Symmetric() const = 0;
    virtual bool StronglyAdmissible() const = 0;

    // Display the equivalent dense matrix
    virtual void Print
    ( const std::string tag, std::ostream& os=std::cout ) const = 0;

    // y := alpha A x + beta y
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y )
    const = 0;

    // y := alpha A x
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^T x + beta y
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y )
    const = 0;

    // y := alpha A^T x
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^H x + beta y
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y )
    const = 0;

    // y := alpha A^H x
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // C := alpha A B + beta C
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B,
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A B
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;

    // C := alpha A^T B + beta C
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B,
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A^T B
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;

    // C := alpha A^H B + beta C
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B,
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A^H B
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;
};

} // namespace dmhm

#endif // ifndef DMHM_ABSTRACT_HMAT_HPP
