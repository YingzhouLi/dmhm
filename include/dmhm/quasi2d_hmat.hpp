/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_QUASI2DHMAT_HPP
#define DMHM_QUASI2DHMAT_HPP 1

#include "dmhm/building_blocks/abstract_hmat.hpp"
#include "dmhm/hmat_tools.hpp"

namespace dmhm {

// Forward declare friend classes
template<typename Scalar> class DistQuasi2dHMat;

template<typename Scalar>
class Quasi2dHMat : public AbstractHMat<Scalar>
{
public:    
    typedef BASE(Scalar) Real;
    friend class DistQuasi2dHMat<Scalar>;

    /*
     * Public static member functions
     */
    static int SampleRank( int approxRank ) { return approxRank + 4; }

    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels );

    /*
     * Public non-static member functions
     */
    Quasi2dHMat();

    // Create a square top-level H-matrix
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    Quasi2dHMat
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMat
    ( const LowRank<Scalar>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMat
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    
    // Create a potentially non-square non-top-level H-matrix
    Quasi2dHMat
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMat
    ( const LowRank<Scalar>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMat
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );

    // Reconstruct an H-matrix from its packed form
    Quasi2dHMat( const std::vector<byte>& packedHMat );

    ~Quasi2dHMat();
    void Clear();

    void SetToRandom();

    // Fulfillments of AbstractHMat
    virtual int Height() const;
    virtual int Width() const;
    virtual int NumLevels() const;
    virtual int MaxRank() const;
    virtual int SourceOffset() const;
    virtual int TargetOffset() const;
    virtual bool Symmetric() const;
    virtual bool StronglyAdmissible() const;

    // Routines useful for packing and unpacking the Quasi2dHMat to/from
    // a contiguous buffer.
    std::size_t PackedSize() const;
    std::size_t Pack( byte* packedHMat ) const;
    std::size_t Pack( std::vector<byte>& packedHMat ) const;
    std::size_t Unpack( const byte* packedHMat );
    std::size_t Unpack( const std::vector<byte>& packedHMat );

    int XSizeSource() const { return _xSizeSource; }
    int XSizeTarget() const { return _xSizeTarget; }
    int YSizeSource() const { return _ySizeSource; }
    int YSizeTarget() const { return _ySizeTarget; }
    int ZSize() const { return _zSize; }
    int XSource() const { return _xSource; }
    int YSource() const { return _ySource; }
    int XTarget() const { return _xTarget; }
    int YTarget() const { return _yTarget; }

    bool IsDense() const { return _block.type == DENSE; }
    bool IsHierarchical() const
    { return _block.type == NODE || _block.type == NODE_SYMMETRIC; }
    bool IsLowRank() const { return _block.type == LOW_RANK; }

    /* 
     * Write a representation of the H-matrix structure to file. 
     */
    // Compile this output with pdflatex+TikZ
    void LatexWriteStructure( const std::string filebase ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptWriteStructure( const std::string filebase ) const;

    //------------------------------------------------------------------------//
    // Fulfillments of AbstractHMat interface                                 //
    //------------------------------------------------------------------------//

    // Multiply the H-matrix by identity and print the result
    virtual void Print( const std::string tag ) const;

    // y := alpha H x + beta y
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A x
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^T x + beta y
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^T x
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x + beta y
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x + beta y (temporarily conjugate x in-place)
    void AdjointMultiply
    ( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x (temporarily conjugate x in-place)
    void AdjointMultiply
    ( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C := alpha A B + beta C
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A B
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A^T B
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^H B + beta C
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A^H B + beta C (temporarily conjugate B in place)
    void AdjointMultiply
    ( Scalar alpha, Dense<Scalar>& B, 
      Scalar beta,  Dense<Scalar>& C ) const;

    // C := alpha A^H B
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^H B (temporarily conjugate B in place)
    void AdjointMultiply
    ( Scalar alpha, Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    //------------------------------------------------------------------------//
    // Computational routines specific to Quasi2dHMat                         //
    //------------------------------------------------------------------------//

    // A := B
    void CopyFrom( const Quasi2dHMat<Scalar>& B );
    
    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const Quasi2dHMat<Scalar>& B );

    // A := B^T
    void TransposeFrom( const Quasi2dHMat<Scalar>& B );

    // A := B^H
    void AdjointFrom( const Quasi2dHMat<Scalar>& B );

    // A := alpha A
    void Scale( Scalar alpha );

    // A := I
    void SetToIdentity();

    // A := A + alpha I
    void AddConstantToDiagonal( Scalar alpha );

    // A :~= alpha B + A
    void UpdateWith( Scalar alpha, const Quasi2dHMat<Scalar>& B );

    // C :~= alpha A B
    void Multiply
    ( Scalar alpha, const Quasi2dHMat<Scalar>& B, 
                          Quasi2dHMat<Scalar>& C ) const;

    // C :~= alpha A B + beta C
    void Multiply
    ( Scalar alpha, const Quasi2dHMat<Scalar>& B, 
      Scalar beta,        Quasi2dHMat<Scalar>& C ) const;

    // A :~= inv(A) using recursive Schur complements
    void DirectInvert();

    // A :~= inv(A) using Schulz iteration, 
    //     X_k+1 = X_k (2I - A X_k) = (2I - X_k A) X_k,
    // where X_k -> inv(A) if X_0 = alpha A^H,
    // with 0 < alpha < 2/||A||_2^2.
    //
    // Require the condition number estimation to be accurate enough that
    //   Estimate <= ||A||_2 <= theta Estimate, where 1 < theta,
    // with probability at least 1-10^{-confidence}.
    //
    // The values for theta and confidence are currently hardcoded.
    void SchulzInvert( int numIterations, Real theta=1.5, Real confidence=6 );

private:
    /*
     * Private static member functions
     */
    static void BuildMapOnQuadrant
    ( int* map, int& index, int level, int numLevels,
      int xSize, int ySize, int zSize, int thisXSize, int thisYSize );

    /*
     * Private data structures
     */
    struct Node
    {
        std::vector<Quasi2dHMat*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
        Node
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget,
          int zSize );
        ~Node();
        Quasi2dHMat& Child( int i, int j );
        const Quasi2dHMat& Child( int i, int j ) const;
    };
    Node* NewNode() const;

    struct NodeSymmetric
    {
        std::vector<Quasi2dHMat*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();
        Quasi2dHMat& Child( int i, int j );
        const Quasi2dHMat& Child( int i, int j ) const;
    };
    NodeSymmetric* NewNodeSymmetric() const;

    enum BlockType 
    { 
        NODE, 
        NODE_SYMMETRIC, 
        LOW_RANK, 
        DENSE 
    };

    struct Block
    {
        BlockType type;
        union Data
        {
            Node* N;
            NodeSymmetric* NS;
            LowRank<Scalar>* F;
            Dense<Scalar>* D;
            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Block();
        ~Block();
        void Clear();
    };

    /*
     * Private member data
     */
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    bool _symmetric;
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Block _block;

    /*
     * Private non-static member functions
     */
    void PackedSizeRecursion( std::size_t& packedSize ) const;
    void PackRecursion( byte*& head ) const;

    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void ImportLowRank( const LowRank<Scalar>& F );
    
    void UpdateWithLowRank( Scalar alpha, const LowRank<Scalar>& F );

    void ImportSparse
    ( const Sparse<Scalar>& S, int iOffset=0, int jOffset=0 );

    void UnpackRecursion( const byte*& head );

   // y += alpha A x
    void UpdateVectorWithNodeSymmetric
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C += alpha A B
    void UpdateWithNodeSymmetric
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;

    void LatexWriteStructureRecursion
    ( std::ofstream& file, int globalHeight ) const;

    void MScriptWriteStructureRecursion( std::ofstream& file ) const;
};

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::Height() const
{ return _xSizeTarget*_ySizeTarget*_zSize; }

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::Width() const
{ return _xSizeSource*_ySizeSource*_zSize; }

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::NumLevels() const
{ return _numLevels; }

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::MaxRank() const
{ return _maxRank; }

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::SourceOffset() const
{ return _sourceOffset; }

template<typename Scalar>
inline int
Quasi2dHMat<Scalar>::TargetOffset() const
{ return _targetOffset; }

template<typename Scalar>
inline bool
Quasi2dHMat<Scalar>::Symmetric() const
{ return _symmetric; }

template<typename Scalar>
inline bool
Quasi2dHMat<Scalar>::StronglyAdmissible() const
{ return _stronglyAdmissible; }

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::Node::Node
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize )
: children(16)
{
    xSourceSizes[0] = xSizeSource/2;
    xSourceSizes[1] = xSizeSource - xSourceSizes[0];
    ySourceSizes[0] = ySizeSource/2;
    ySourceSizes[1] = ySizeSource - ySourceSizes[0];
            
    sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0]*zSize;
    sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0]*zSize;
    sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1]*zSize;
    sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1]*zSize;

    xTargetSizes[0] = xSizeTarget/2;
    xTargetSizes[1] = xSizeTarget - xTargetSizes[0];
    yTargetSizes[0] = ySizeTarget/2;
    yTargetSizes[1] = ySizeTarget - yTargetSizes[0];

    targetSizes[0] = xTargetSizes[0]*yTargetSizes[0]*zSize;
    targetSizes[1] = xTargetSizes[1]*yTargetSizes[0]*zSize;
    targetSizes[2] = xTargetSizes[0]*yTargetSizes[1]*zSize;
    targetSizes[3] = xTargetSizes[1]*yTargetSizes[1]*zSize;
}

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar>
inline Quasi2dHMat<Scalar>& 
Quasi2dHMat<Scalar>::Node::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[j+4*i]; 
}

template<typename Scalar>
inline const Quasi2dHMat<Scalar>& 
Quasi2dHMat<Scalar>::Node::Child( int i, int j ) const
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[j+4*i]; 
}

template<typename Scalar>
inline typename Quasi2dHMat<Scalar>::Node*
Quasi2dHMat<Scalar>::NewNode() const
{
    return 
        new typename Quasi2dHMat<Scalar>::Node
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize );
}

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::NodeSymmetric::NodeSymmetric
( int xSize, int ySize, int zSize )
: children(10)
{
    xSizes[0] = xSize/2;
    xSizes[1] = xSize - xSizes[0];
    ySizes[0] = ySize/2;
    ySizes[1] = ySize - ySizes[0];

    sizes[0] = xSizes[0]*ySizes[0]*zSize;
    sizes[1] = xSizes[1]*ySizes[0]*zSize;
    sizes[2] = xSizes[0]*ySizes[1]*zSize;
    sizes[3] = xSizes[1]*ySizes[1]*zSize;
}

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar>
inline Quasi2dHMat<Scalar>& 
Quasi2dHMat<Scalar>::NodeSymmetric::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[(i*(i+1))/2 + j]; 
}

template<typename Scalar>
inline const Quasi2dHMat<Scalar>& 
Quasi2dHMat<Scalar>::NodeSymmetric::Child( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[(i*(i+1))/2 + j];
}

template<typename Scalar>
inline typename Quasi2dHMat<Scalar>::NodeSymmetric*
Quasi2dHMat<Scalar>::NewNodeSymmetric() const
{
    return 
        new typename Quasi2dHMat<Scalar>::NodeSymmetric
        ( _xSizeSource, _ySizeSource, _zSize );
}

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::Block::Block()
: type(NODE), data() 
{ }

template<typename Scalar>
inline
Quasi2dHMat<Scalar>::Block::~Block()
{ Clear(); }

template<typename Scalar>
inline void
Quasi2dHMat<Scalar>::Block::Clear()
{
    switch( type )
    {
    case NODE:           delete data.N;  break;
    case NODE_SYMMETRIC: delete data.NS; break;
    case LOW_RANK:       delete data.F;  break;
    case DENSE:          delete data.D;  break;
    }
    type = NODE;
    data.N = 0;
}

} // namespace dmhm

#endif // ifndef DMHM_QUASI2DHMAT_HPP
