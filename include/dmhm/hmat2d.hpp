/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_HMAT2D_HPP
#define DMHM_HMAT2D_HPP 1

#include "dmhm/core/abstract_hmat.hpp"
#include "dmhm/hmat_tools.hpp"
#include "dmhm/graphics.hpp"

namespace dmhm {

// Forward declare friend classes
template<typename Scalar> class DistHMat2d;

template<typename Scalar>
class HMat2d : public AbstractHMat<Scalar>
{
public:    
    typedef BASE(Scalar) Real;
    friend class DistHMat2d<Scalar>;

    /*
     * Public static member functions
     */
    static int SampleRank( int approxRank ) 
    { return approxRank + Oversample(); }

    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& map, int xSize, int ySize, int numLevels );

    /*
     * Public non-static member functions
     */
    HMat2d();

    // Create a square top-level H-matrix
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    HMat2d
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSize, int ySize );
    HMat2d
    ( const LowRank<Scalar>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize );
    HMat2d
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize );
    
    // Create a potentially non-square non-top-level H-matrix
    HMat2d
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    HMat2d
    ( const LowRank<Scalar>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    HMat2d
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );

    // Reconstruct an H-matrix from its packed form
    HMat2d( const std::vector<byte>& packedHMat );

    virtual ~HMat2d();
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

    // Routines useful for packing and unpacking the HMat2d to/from
    // a contiguous buffer.
    std::size_t PackedSize() const;
    std::size_t Pack( byte* packedHMat ) const;
    std::size_t Pack( std::vector<byte>& packedHMat ) const;
    std::size_t Unpack( const byte* packedHMat );
    std::size_t Unpack( const std::vector<byte>& packedHMat );

    int XSizeSource() const { return xSizeSource_; }
    int XSizeTarget() const { return xSizeTarget_; }
    int YSizeSource() const { return ySizeSource_; }
    int YSizeTarget() const { return ySizeTarget_; }
    int XSource() const { return xSource_; }
    int YSource() const { return ySource_; }
    int XTarget() const { return xTarget_; }
    int YTarget() const { return yTarget_; }

    bool IsDense() const { return block_.type == DENSE; }
    bool IsHierarchical() const
    { return block_.type == NODE || block_.type == NODE_SYMMETRIC; }
    bool IsLowRank() const { return block_.type == LOW_RANK; }

    /* 
     * Visualize the H-matrix structure
     */
#ifdef HAVE_QT5
    // Display structure with Qt
    void Display( std::string title="" ) const;
#endif
    // Compile this output with pdflatex+TikZ
    void LatexStructure( const std::string filebase ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptStructure( const std::string filebase ) const;

    //------------------------------------------------------------------------//
    // Fulfillments of AbstractHMat interface                                 //
    //------------------------------------------------------------------------//

    // Multiply the H-matrix by identity and print the result
    virtual void Print
    ( const std::string tag, std::ostream& os=std::cout ) const;

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
    // Computational routines specific to HMat2d                         //
    //------------------------------------------------------------------------//

    // A := B
    void CopyFrom( const HMat2d<Scalar>& B );
    
    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const HMat2d<Scalar>& B );

    // A := B^T
    void TransposeFrom( const HMat2d<Scalar>& B );

    // A := B^H
    void AdjointFrom( const HMat2d<Scalar>& B );

    // A := alpha A
    void Scale( Scalar alpha );

    // A := I
    void SetToIdentity();

    // A := A + alpha I
    void AddConstantToDiagonal( Scalar alpha );

    // A :~= alpha B + A
    void UpdateWith( Scalar alpha, const HMat2d<Scalar>& B );

    // C :~= alpha A B
    void Multiply
    ( Scalar alpha, const HMat2d<Scalar>& B, 
                          HMat2d<Scalar>& C ) const;

    // C :~= alpha A B + beta C
    void Multiply
    ( Scalar alpha, const HMat2d<Scalar>& B, 
      Scalar beta,        HMat2d<Scalar>& C ) const;

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
      int xSize, int ySize, int thisXSize, int thisYSize );

    /*
     * Private data structures
     */
    struct Node
    {
        std::vector<HMat2d*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
        Node
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget );
        ~Node();
        HMat2d& Child( int i, int j );
        const HMat2d& Child( int i, int j ) const;
    };
    Node* NewNode() const;

    struct NodeSymmetric
    {
        std::vector<HMat2d*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize );
        ~NodeSymmetric();
        HMat2d& Child( int i, int j );
        const HMat2d& Child( int i, int j ) const;
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
    int numLevels_;
    int maxRank_;
    int sourceOffset_, targetOffset_;
    bool symmetric_;
    bool stronglyAdmissible_;

    int xSizeSource_, xSizeTarget_;
    int ySizeSource_, ySizeTarget_;
    int xSource_, xTarget_;
    int ySource_, yTarget_;
    Block block_;

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

#ifdef HAVE_QT5
    void DisplayRecursion
    ( Dense<double>* matrix, int mRatio, int nRatio ) const;
#endif

    void LatexStructureRecursion
    ( std::ofstream& file, int globalHeight ) const;

    void MScriptStructureRecursion( std::ofstream& file ) const;
};

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline int
HMat2d<Scalar>::Height() const
{ return xSizeTarget_*ySizeTarget_; }

template<typename Scalar>
inline int
HMat2d<Scalar>::Width() const
{ return xSizeSource_*ySizeSource_; }

template<typename Scalar>
inline int
HMat2d<Scalar>::NumLevels() const
{ return numLevels_; }

template<typename Scalar>
inline int
HMat2d<Scalar>::MaxRank() const
{ return maxRank_; }

template<typename Scalar>
inline int
HMat2d<Scalar>::SourceOffset() const
{ return sourceOffset_; }

template<typename Scalar>
inline int
HMat2d<Scalar>::TargetOffset() const
{ return targetOffset_; }

template<typename Scalar>
inline bool
HMat2d<Scalar>::Symmetric() const
{ return symmetric_; }

template<typename Scalar>
inline bool
HMat2d<Scalar>::StronglyAdmissible() const
{ return stronglyAdmissible_; }

template<typename Scalar>
inline
HMat2d<Scalar>::Node::Node
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget )
: children(16)
{
    xSourceSizes[0] = xSizeSource/2;
    xSourceSizes[1] = xSizeSource - xSourceSizes[0];
    ySourceSizes[0] = ySizeSource/2;
    ySourceSizes[1] = ySizeSource - ySourceSizes[0];
            
    sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0];
    sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0];
    sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1];
    sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1];

    xTargetSizes[0] = xSizeTarget/2;
    xTargetSizes[1] = xSizeTarget - xTargetSizes[0];
    yTargetSizes[0] = ySizeTarget/2;
    yTargetSizes[1] = ySizeTarget - yTargetSizes[0];

    targetSizes[0] = xTargetSizes[0]*yTargetSizes[0];
    targetSizes[1] = xTargetSizes[1]*yTargetSizes[0];
    targetSizes[2] = xTargetSizes[0]*yTargetSizes[1];
    targetSizes[3] = xTargetSizes[1]*yTargetSizes[1];
}

template<typename Scalar>
inline
HMat2d<Scalar>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar>
inline HMat2d<Scalar>& 
HMat2d<Scalar>::Node::Child( int i, int j )
{ 
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[j+4*i]; 
}

template<typename Scalar>
inline const HMat2d<Scalar>& 
HMat2d<Scalar>::Node::Child( int i, int j ) const
{ 
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[j+4*i]; 
}

template<typename Scalar>
inline typename HMat2d<Scalar>::Node*
HMat2d<Scalar>::NewNode() const
{
    return 
        new typename HMat2d<Scalar>::Node
        ( xSizeSource_, xSizeTarget_, ySizeSource_, ySizeTarget_ );
}

template<typename Scalar>
inline
HMat2d<Scalar>::NodeSymmetric::NodeSymmetric( int xSize, int ySize )
: children(10)
{
    xSizes[0] = xSize/2;
    xSizes[1] = xSize - xSizes[0];
    ySizes[0] = ySize/2;
    ySizes[1] = ySize - ySizes[0];

    sizes[0] = xSizes[0]*ySizes[0];
    sizes[1] = xSizes[1]*ySizes[0];
    sizes[2] = xSizes[0]*ySizes[1];
    sizes[3] = xSizes[1]*ySizes[1];
}

template<typename Scalar>
inline
HMat2d<Scalar>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar>
inline HMat2d<Scalar>& 
HMat2d<Scalar>::NodeSymmetric::Child( int i, int j )
{ 
#ifndef RELEASE
    CallStackEntry entry("HMat2d::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[(i*(i+1))/2 + j]; 
}

template<typename Scalar>
inline const HMat2d<Scalar>& 
HMat2d<Scalar>::NodeSymmetric::Child( int i, int j ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[(i*(i+1))/2 + j];
}

template<typename Scalar>
inline typename HMat2d<Scalar>::NodeSymmetric*
HMat2d<Scalar>::NewNodeSymmetric() const
{
    return 
        new typename HMat2d<Scalar>::NodeSymmetric
        ( xSizeSource_, ySizeSource_ );
}

template<typename Scalar>
inline
HMat2d<Scalar>::Block::Block()
: type(NODE), data() 
{ }

template<typename Scalar>
inline
HMat2d<Scalar>::Block::~Block()
{ Clear(); }

template<typename Scalar>
inline void
HMat2d<Scalar>::Block::Clear()
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

#endif // ifndef DMHM_HMAT2D_HPP
