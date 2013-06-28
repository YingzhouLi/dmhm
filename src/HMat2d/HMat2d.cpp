/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
#ifdef HAVE_QT5
 #include <QApplication>
#endif

#include "./Invert-incl.hpp"
#include "./MultiplyDense-incl.hpp"
#include "./MultiplyHMat-incl.hpp"
#include "./MultiplyVector-incl.hpp"
#include "./Pack-incl.hpp"
#include "./SetToRandom-incl.hpp"

namespace dmhm {

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//

template<typename Scalar>
void 
HMat2d<Scalar>::BuildMapOnQuadrant
( int* map, int& index, int level, int numLevels,
  int xSize, int ySize, int thisXSize, int thisYSize )
{
    if( level == numLevels-1 )
    {
        // Stamp these indices into the buffer 
        for( int j=0; j<thisYSize; ++j )
        {
            int* thisRow = &map[j*xSize];
            for( int i=0; i<thisXSize; ++i )
                thisRow[i] = index++;
        }
    }
    else
    {
        const int leftWidth = thisXSize/2;
        const int rightWidth = thisXSize - leftWidth;
        const int bottomHeight = thisYSize/2;
        const int topHeight = thisYSize - bottomHeight;

        // Recurse on the lower-left quadrant 
        BuildMapOnQuadrant
        ( &map[0], index, level+1, numLevels,
          xSize, ySize, leftWidth, bottomHeight );
        // Recurse on the lower-right quadrant
        BuildMapOnQuadrant
        ( &map[leftWidth], index, level+1, numLevels,
          xSize, ySize, rightWidth, bottomHeight );
        // Recurse on the upper-left quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize], index, level+1, numLevels,
          xSize, ySize, leftWidth, topHeight );
        // Recurse on the upper-right quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize+leftWidth], index, level+1, numLevels,
          xSize, ySize, rightWidth, topHeight );
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::BuildNaturalToHierarchicalMap
( std::vector<int>& map, int xSize, int ySize, int numLevels )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::BuildNaturalToHierarchicalMap");
#endif
    map.resize( xSize*ySize );

    // Fill the mapping from the 'natural' x-y-z ordering
    int index = 0;
    BuildMapOnQuadrant
    ( &map[0], index, 0, numLevels, xSize, ySize, xSize, ySize );
#ifndef RELEASE
    if( index != xSize*ySize )
        throw std::logic_error("Map recursion is incorrect.");
#endif
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

// Create an empty H-matrix
template<typename Scalar>
HMat2d<Scalar>::HMat2d()
: numLevels_(0),
  maxRank_(0),
  sourceOffset_(0), targetOffset_(0),
  symmetric_(false), 
  stronglyAdmissible_(false),
  xSizeSource_(0), xSizeTarget_(0),
  ySizeSource_(0), ySizeTarget_(0),
  xSource_(0), xTarget_(0),
  ySource_(0), yTarget_(0)
{ }

// Create a square top-level H-matrix
template<typename Scalar>
HMat2d<Scalar>::HMat2d
( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
  int xSize, int ySize )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(0), targetOffset_(0),
  symmetric_(symmetric), 
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSize), xSizeTarget_(xSize),
  ySizeSource_(ySize), ySizeTarget_(ySize),
  xSource_(0), xTarget_(0),
  ySource_(0), yTarget_(0)
{ }

template<typename Scalar>
HMat2d<Scalar>::HMat2d
( const LowRank<Scalar>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(0), targetOffset_(0),
  symmetric_(false), 
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSize), xSizeTarget_(xSize),
  ySizeSource_(ySize), ySizeTarget_(ySize),
  xSource_(0), xTarget_(0),
  ySource_(0), yTarget_(0)
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::HMat2d");
#endif
    ImportLowRank( F );
}

template<typename Scalar>
HMat2d<Scalar>::HMat2d
( const Sparse<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible, int xSize, int ySize )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(0), targetOffset_(0),
  symmetric_(S.symmetric),
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSize), xSizeTarget_(xSize),
  ySizeSource_(ySize), ySizeTarget_(ySize),
  xSource_(0), xTarget_(0),
  ySource_(0), yTarget_(0)
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::HMat2d");
#endif
    ImportSparse( S );
}

// Create a potentially non-square non-top-level H-matrix
template<typename Scalar>
HMat2d<Scalar>::HMat2d
( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(sourceOffset), targetOffset_(targetOffset),
  symmetric_(symmetric),
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget),
  xSource_(xSource), xTarget_(xTarget),
  ySource_(ySource), yTarget_(yTarget)
{ }

template<typename Scalar>
HMat2d<Scalar>::HMat2d
( const LowRank<Scalar>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(sourceOffset), targetOffset_(targetOffset),
  symmetric_(false),
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget),
  xSource_(xSource), xTarget_(xTarget),
  ySource_(ySource), yTarget_(yTarget)
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::HMat2d");
#endif
    ImportLowRank( F );
}

template<typename Scalar>
HMat2d<Scalar>::HMat2d
( const Sparse<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: numLevels_(numLevels),
  maxRank_(maxRank),
  sourceOffset_(sourceOffset), targetOffset_(targetOffset),
  symmetric_(S.symmetric && sourceOffset==targetOffset),
  stronglyAdmissible_(stronglyAdmissible),
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget), 
  xSource_(xSource), xTarget_(xTarget),
  ySource_(ySource), yTarget_(yTarget)
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::HMat2d");
#endif
    ImportSparse( S, targetOffset, sourceOffset );
}

template<typename Scalar>
HMat2d<Scalar>::~HMat2d()
{ }

template<typename Scalar>
void
HMat2d<Scalar>::Clear()
{ block_.Clear(); }

template<typename Scalar>
void
HMat2d<Scalar>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Print");
#endif
    const int n = Width();
    Dense<Scalar> I( n, n );
    MemZero( I.Buffer(), I.LDim()*n );
    for( int j=0; j<n; ++j )
        I.Set(j,j,Scalar(1));

    Dense<Scalar> HFull;
    Multiply( Scalar(1), I, HFull );
    HFull.Print( tag, os );
}

#ifdef HAVE_QT5
template<typename Scalar>
void
HMat2d<Scalar>::Display( std::string title ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Display");
#endif
    const int m = Height();
    const int n = Width();
    const int mRatio = 2;
    const int nRatio = 2;
    const int mPix = m*mRatio;
    const int nPix = n*nRatio;
    Dense<double>* A = new Dense<double>( mPix, nPix );

    // Initialize the matrix to all zeros
    for( int j=0; j<n; ++j )
        for( int i=0; i<m; ++i )
            A->Set( i, j, 0 );
 
    // Now fill in the H-matrix blocks recursively
    DisplayRecursion( A, mRatio, nRatio );

    QString qTitle = QString::fromStdString( title );
    DisplayWindow* displayWindow = new DisplayWindow;
    displayWindow->Display( A, qTitle );
    displayWindow->show();

    // Spend at most 200 milliseconds rendering
    QCoreApplication::instance()->processEvents( QEventLoop::AllEvents, 200 );
}
#endif // ifdef HAVE_QT5

template<typename Scalar>
void
HMat2d<Scalar>::LatexStructure( const std::string filebase ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::LatexStructure");
#endif

    std::ofstream file( (filebase+".tex").c_str() );
    double scale = 12.8;
    file << "\\documentclass[11pt]{article}\n"
         << "\\usepackage{tikz}\n"
         << "\\begin{document}\n"
         << "\\begin{center}\n"
         << "\\begin{tikzpicture}[scale=" << scale << "]\n";
    LatexStructureRecursion( file, Height() );
    file << "\\end{tikzpicture}\n"
         << "\\end{center}\n"
         << "\\end{document}" << std::endl;
}

template<typename Scalar>
void
HMat2d<Scalar>::MScriptStructure( const std::string filebase ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::MScriptStructure");
#endif
    std::ofstream file( (filebase+".dat").c_str() );
    MScriptStructureRecursion( file );
}

/*\
|*| Computational routines specific to HMat2d
\*/

// A := B
template<typename Scalar>
void
HMat2d<Scalar>::CopyFrom( const HMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::CopyFrom");
#endif
    numLevels_ = B.NumLevels();
    maxRank_ = B.MaxRank();
    sourceOffset_ = B.SourceOffset();
    targetOffset_ = B.TargetOffset();
    symmetric_ = B.Symmetric();
    stronglyAdmissible_ = B.StronglyAdmissible();
    xSizeSource_ = B.XSizeSource();
    xSizeTarget_ = B.XSizeTarget();
    ySizeSource_ = B.YSizeSource();
    ySizeTarget_ = B.YSizeTarget();
    xSource_ = B.XSource();
    xTarget_ = B.XTarget();
    ySource_ = B.YSource();
    yTarget_ = B.YTarget();

    block_.Clear();
    block_.type = B.block_.type;
    switch( block_.type )
    {
    case NODE:
    {
        block_.data.N = NewNode();
        Node& nodeA = *block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int i=0; i<16; ++i )
        {
            nodeA.children[i] = new HMat2d<Scalar>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        block_.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *block_.data.NS;
        const NodeSymmetric& nodeB = *B.block_.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new HMat2d<Scalar>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
        block_.data.F = new LowRank<Scalar>;
        hmat_tools::Copy( *B.block_.data.F, *block_.data.F );
        break;
    case DENSE:
        block_.data.D = new Dense<Scalar>;
        hmat_tools::Copy( *B.block_.data.D, *block_.data.D );
        break;
    }
}

// A := Conj(A)
template<typename Scalar>
void
HMat2d<Scalar>::Conjugate()
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Conjugate");
#endif
    switch( block_.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            block_.data.N->children[i]->Conjugate();
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            block_.data.NS->children[i]->Conjugate();
        break;
    case LOW_RANK:
        hmat_tools::Conjugate( block_.data.F->U );
        hmat_tools::Conjugate( block_.data.F->V );
        break;
    case DENSE:
        hmat_tools::Conjugate( *block_.data.D );
        break;
    }
}

// A := Conj(B)
template<typename Scalar>
void
HMat2d<Scalar>::ConjugateFrom( const HMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::ConjugateFrom");
#endif
    HMat2d<Scalar>& A = *this;

    A.numLevels_ = B.NumLevels();
    A.maxRank_ = B.MaxRank();
    A.sourceOffset_ = B.SourceOffset();
    A.targetOffset_ = B.TargetOffset();
    A.symmetric_ = B.Symmetric();
    A.stronglyAdmissible_ = B.StronglyAdmissible();
    A.xSizeSource_ = B.XSizeSource();
    A.xSizeTarget_ = B.XSizeTarget();
    A.ySizeSource_ = B.YSizeSource();
    A.ySizeTarget_ = B.YSizeTarget();
    A.xSource_ = B.XSource();
    A.xTarget_ = B.XTarget();
    A.ySource_ = B.YSource();
    A.yTarget_ = B.YTarget();

    A.block_.Clear();
    A.block_.type = B.block_.type;
    switch( A.block_.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
        {
            A.block_.data.N->children[i]->ConjugateFrom
            ( *B.block_.data.N->children[i] );
        }
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            A.block_.data.NS->children[i]->ConjugateFrom
            ( *B.block_.data.NS->children[i] );
        }
        break;
    case LOW_RANK:
        hmat_tools::Conjugate( B.block_.data.F->U, A.block_.data.F->U );
        hmat_tools::Conjugate( B.block_.data.F->V, A.block_.data.F->V );
        break;
    case DENSE:
        hmat_tools::Conjugate( *B.block_.data.D, *A.block_.data.D );
        break;
    }
}

// A := B^T
template<typename Scalar>
void
HMat2d<Scalar>::TransposeFrom( const HMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::TransposeFrom");
#endif
    HMat2d<Scalar>& A = *this;

    A.numLevels_ = B.NumLevels();
    A.maxRank_ = B.MaxRank();
    A.sourceOffset_ = B.TargetOffset();
    A.targetOffset_ = B.SourceOffset();
    A.symmetric_ = B.Symmetric();
    A.stronglyAdmissible_ = B.StronglyAdmissible();
    A.xSizeSource_ = B.XSizeTarget();
    A.xSizeTarget_ = B.XSizeSource();
    A.ySizeSource_ = B.YSizeTarget();
    A.ySizeTarget_ = B.YSizeSource();
    A.xSource_ = B.XTarget();
    A.xTarget_ = B.XSource();
    A.ySource_ = B.YTarget();
    A.yTarget_ = B.YSource();

    A.block_.Clear();
    A.block_.type = B.block_.type;
    switch( A.block_.type )
    {
    case NODE:
    {
        A.block_.data.N = NewNode();
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new HMat2d<Scalar>;
                nodeA.Child(t,s).TransposeFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        A.block_.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *A.block_.data.NS;
        const NodeSymmetric& nodeB = *B.block_.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new HMat2d<Scalar>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        A.block_.data.F = new LowRank<Scalar>;
        hmat_tools::Transpose( *B.block_.data.F, *A.block_.data.F );
        break;
    }
    case DENSE:
    {
        A.block_.data.D = new Dense<Scalar>;
        hmat_tools::Transpose( *B.block_.data.D, *A.block_.data.D );
        break;
    }
    }
}

// A := B^H
template<typename Scalar>
void
HMat2d<Scalar>::AdjointFrom( const HMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::AdjointFrom");
#endif
    HMat2d<Scalar>& A = *this;

    A.numLevels_ = B.NumLevels();
    A.maxRank_ = B.MaxRank();
    A.sourceOffset_ = B.TargetOffset();
    A.targetOffset_ = B.SourceOffset();
    A.symmetric_ = B.Symmetric();
    A.stronglyAdmissible_ = B.StronglyAdmissible();
    A.xSizeSource_ = B.XSizeTarget();
    A.xSizeTarget_ = B.XSizeSource();
    A.ySizeSource_ = B.YSizeTarget();
    A.ySizeTarget_ = B.YSizeSource();
    A.xSource_ = B.XTarget();
    A.xTarget_ = B.XSource();
    A.ySource_ = B.YTarget();
    A.yTarget_ = B.YSource();

    A.block_.Clear();
    A.block_.type = B.block_.type;
    switch( B.block_.type )
    {
    case NODE:
    {
        A.block_.data.N = NewNode();
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new HMat2d<Scalar>;
                nodeA.Child(t,s).AdjointFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        A.block_.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *A.block_.data.NS;
        const NodeSymmetric& nodeB = *B.block_.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new HMat2d<Scalar>;
            nodeA.children[i]->ConjugateFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        A.block_.data.F = new LowRank<Scalar>;
        hmat_tools::Adjoint( *B.block_.data.F, *A.block_.data.F );
        break;
    }
    case DENSE:
    {
        A.block_.data.D = new Dense<Scalar>;
        hmat_tools::Adjoint( *B.block_.data.D, *A.block_.data.D );
        break;
    }
    }
}

// A := alpha A
template<typename Scalar>
void
HMat2d<Scalar>::Scale( Scalar alpha )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Scale");
#endif
    switch( block_.type )
    {
    case NODE:
    {
        Node& nodeA = *block_.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *block_.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case LOW_RANK:
        hmat_tools::Scale( alpha, *block_.data.F );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *block_.data.D );
        break;
    }
}

// A := I
template<typename Scalar>
void
HMat2d<Scalar>::SetToIdentity()
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::SetToIdentity");
#endif
    switch( block_.type )
    {
    case NODE:
    {
        Node& nodeA = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                if( s == t )
                    nodeA.Child(t,s).SetToIdentity();
                else
                    nodeA.Child(t,s).Scale( Scalar(0) );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *block_.data.NS;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<t; ++s )
                nodeA.Child(t,s).Scale( Scalar(0) );
            nodeA.Child(t,t).SetToIdentity();
        }
        break;
    }
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Error in SetToIdentity logic.");
#endif
        break;
    }
    case DENSE:
    {
        Dense<Scalar>& D = *block_.data.D;
        hmat_tools::Scale( Scalar(0), D );
        for( int j=0; j<D.Width(); ++j )
        {
            if( j < D.Height() )
                D.Set( j, j, Scalar(1) );
            else
                break;
        }
        break;
    }
    }
}

// A := A + alpha I
template<typename Scalar>
void
HMat2d<Scalar>::AddConstantToDiagonal( Scalar alpha )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::AddConstantToDiagonal");
#endif
    switch( block_.type )
    {
    case NODE:
    {
        Node& nodeA = *block_.data.N;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *block_.data.NS;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case DENSE:
    {
        Scalar* DBuffer = block_.data.D->Buffer();
        const int m = block_.data.D->Height();
        const int n = block_.data.D->Width();
        const int DLDim = block_.data.D->LDim();
        for( int j=0; j<std::min(m,n); ++j )
            DBuffer[j+j*DLDim] += alpha;
        break;
    }
    }
}

// A := alpha B + A
template<typename Scalar>
void
HMat2d<Scalar>::UpdateWith( Scalar alpha, const HMat2d<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::UpdateWith");
#endif
    HMat2d<Scalar>& A = *this;

    switch( A.block_.type )
    {
    case NODE:
    {
        Node& nodeA = *A.block_.data.N;
        const Node& nodeB = *B.block_.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *A.block_.data.NS;
        const NodeSymmetric& nodeB = *B.block_.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case LOW_RANK:
        hmat_tools::RoundedUpdate
        ( this->maxRank_, 
          alpha, *B.block_.data.F, Scalar(1), *A.block_.data.F );
        break;
    case DENSE:
        hmat_tools::Update
        ( alpha, *B.block_.data.D, Scalar(1), *A.block_.data.D );
        break;
    }
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar>
bool
HMat2d<Scalar>::Admissible() const
{
    return Admissible( xSource_, xTarget_, ySource_, yTarget_ );
}

template<typename Scalar>
bool
HMat2d<Scalar>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( stronglyAdmissible_ )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar>
void
HMat2d<Scalar>::ImportLowRank( const LowRank<Scalar>& F )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::ImportLowRank");
#endif
    block_.Clear();
    if( Admissible() )
    {
        block_.type = LOW_RANK;
        block_.data.F = new LowRank<Scalar>;
        hmat_tools::Copy( F.U, block_.data.F->U );
        hmat_tools::Copy( F.V, block_.data.F->V );
    }
    else if( numLevels_ > 1 )
    {
        if( symmetric_ && sourceOffset_ == targetOffset_ )
        {
            block_.type = NODE_SYMMETRIC;
            block_.data.NS = NewNodeSymmetric();
            NodeSymmetric& node = *block_.data.NS;

            int child = 0;
            const int parentOffset = targetOffset_;
            LowRank<Scalar> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s], F.Rank() );

                    node.children[child++] = 
                      new HMat2d<Scalar>
                      ( FSub, 
                        numLevels_-1, maxRank_, 
                        stronglyAdmissible_,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        2*xSource_+(s&1), 2*xTarget_+(t&1),
                        2*ySource_+(s/2), 2*yTarget_+(t/2),
                        sOffset+parentOffset, tOffset+parentOffset );
                }
            }
        }
        else
        {
            block_.type = NODE;
            block_.data.N = NewNode();
            Node& node = *block_.data.N;

            LowRank<Scalar> FSub;
            const int parentSourceOffset = sourceOffset_;
            const int parentTargetOffset = targetOffset_;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s], F.Rank() );

                    node.children[s+4*t] = 
                      new HMat2d<Scalar>
                      ( FSub,
                        numLevels_-1, maxRank_,
                        stronglyAdmissible_,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        2*xSource_+(s&1), 2*xTarget_+(t&1),
                        2*ySource_+(s/2), 2*yTarget_+(t/2),
                        sOffset+parentSourceOffset, 
                        tOffset+parentTargetOffset );
                }
            }
        }
    }
    else
    {
        block_.type = DENSE;
        block_.data.D = new Dense<Scalar>( Height(), Width() );
        const char option = 'T';
        blas::Gemm
        ( 'N', option, Height(), Width(), F.Rank(),
          1, F.U.LockedBuffer(), F.U.LDim(),
             F.V.LockedBuffer(), F.V.LDim(),
          0, block_.data.D->Buffer(), block_.data.D->LDim() );
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::UpdateWithLowRank( Scalar alpha, const LowRank<Scalar>& F )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::UpdateWithLowRank");
#endif
    if( Admissible() )
        hmat_tools::RoundedUpdate
        ( maxRank_, alpha, F, Scalar(1), *block_.data.F );
    else if( numLevels_ > 1 )
    {
        if( symmetric_ )
        {
            NodeSymmetric& node = *block_.data.NS;
            LowRank<Scalar> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRank( alpha, FSub );
                }
            }
        }
        else
        {
            Node& node = *block_.data.N;
            LowRank<Scalar> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRank( alpha, FSub );
                }
            }
        }
    }
    else
    {
        const char option = 'T';
        blas::Gemm
        ( 'N', option, Height(), Width(), F.Rank(),
          alpha, F.U.LockedBuffer(), F.U.LDim(),
                 F.V.LockedBuffer(), F.V.LDim(),
          1, block_.data.D->Buffer(), block_.data.D->LDim() );
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::ImportSparse
( const Sparse<Scalar>& S, int iOffset, int jOffset )
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::ImportSparse");
#endif
    block_.Clear();
    if( Admissible() )
    {
        block_.type = LOW_RANK;
        block_.data.F = new LowRank<Scalar>;
        hmat_tools::ConvertSubmatrix
        ( *block_.data.F, S, iOffset, jOffset, Height(), Width() );
    }
    else if( numLevels_ > 1 )
    {
        if( symmetric_ && sourceOffset_ == targetOffset_ )
        {
            block_.type = NODE_SYMMETRIC;
            block_.data.NS = NewNodeSymmetric();
            NodeSymmetric& node = *block_.data.NS;

            int child = 0;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    node.children[child++] = 
                      new HMat2d<Scalar>
                      ( S, 
                        numLevels_-1, maxRank_,
                        stronglyAdmissible_,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        2*xSource_+(s&1), 2*xTarget_+(t&1),
                        2*ySource_+(s/2), 2*yTarget_+(t/2),
                        sOffset+targetOffset_, tOffset+targetOffset_ );
                }
            }
        }
        else
        {
            block_.type = NODE;
            block_.data.N = NewNode();
            Node& node = *block_.data.N;

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                      new HMat2d<Scalar>
                      ( S,
                        numLevels_-1, maxRank_, stronglyAdmissible_,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        2*xSource_+(s&1), 2*xTarget_+(t&1),
                        2*ySource_+(s/2), 2*yTarget_+(t/2),
                        sOffset+sourceOffset_, tOffset+targetOffset_ );
                }
            }
        }
    }
    else
    {
        block_.type = DENSE;
        block_.data.D = new Dense<Scalar>;
        hmat_tools::ConvertSubmatrix
        ( *block_.data.D, S, iOffset, jOffset, Height(), Width() );
    }
}

// y += alpha A x
template<typename Scalar>
void
HMat2d<Scalar>::UpdateVectorWithNodeSymmetric
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::UpdateVectorWithNodeSymmetric");
#endif
    NodeSymmetric& node = *block_.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Vector<Scalar> ySub;
        ySub.View( y, tOffset, node.sizes[t] );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Vector<Scalar> xSub;
            xSub.LockedView( x, sOffset, node.sizes[s] );

            node.Child(t,s).Multiply( alpha, xSub, Scalar(1), ySub );
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    for( int s=0,tOffset=0; s<4; tOffset+=node.sizes[s],++s )
    {
        Vector<Scalar> ySub;
        ySub.View( y, tOffset, node.sizes[s] );

        for( int t=s+1,sOffset=tOffset+node.sizes[s]; t<4; 
             sOffset+=node.sizes[t],++t )
        {
            Vector<Scalar> xSub;
            xSub.LockedView( x, sOffset, node.sizes[t] );

            node.Child(t,s).TransposeMultiply( alpha, xSub, Scalar(1), ySub );
        }
    }
}

// C += alpha A B
template<typename Scalar>
void
HMat2d<Scalar>::UpdateWithNodeSymmetric
( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::UpdateWithNodeSymmetric");
#endif
    const HMat2d<Scalar>& A = *this;
    NodeSymmetric& node = *A.block_.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Dense<Scalar> CSub;
        CSub.View( C, tOffset, 0, node.sizes[t], C.Width() );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Dense<Scalar> BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[s], B.Width() );

            node.Child(t,s).Multiply( alpha, BSub, Scalar(1), CSub );
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    for( int s=0,tOffset=0; s<4; tOffset+=node.sizes[s],++s )
    {
        Dense<Scalar> CSub;
        CSub.View( C, tOffset, 0, node.sizes[s], C.Width() );

        for( int t=s+1,sOffset=tOffset+node.sizes[s]; t<4; 
             sOffset+=node.sizes[t],++t )
        {
            Dense<Scalar> BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[t], B.Width() );

            node.Child(t,s).TransposeMultiply( alpha, BSub, Scalar(1), CSub );
        }
    }
}

namespace {

void FillBox
( Dense<double>* matrix,
  int mStart, int nStart, int mStop, int nStop,
  double fillValue )
{
    for( int j=nStart; j<nStop; ++j )
        for( int i=mStart; i<mStop; ++i )
            matrix->Set( i, j, fillValue );
}

void FillBox
( std::ofstream& file,
  double hStart, double vStart, double hStop, double vStop,
  const std::string& fillColor )
{
    file << "\\fill[" << fillColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

void DrawBox
( Dense<double>* matrix,
  int mStart, int nStart, int mStop, int nStop,
  double borderValue )
{
    // Draw the horizontal border
    for( int j=nStart; j<nStop; ++j )
    {
        matrix->Set( mStart,  j, borderValue );     
        matrix->Set( mStop-1, j, borderValue );
    }
    // Draw the vertical border
    for( int i=mStart; i<mStop; ++i )
    {
        matrix->Set( i, nStart,  borderValue );
        matrix->Set( i, nStop-1, borderValue );
    }
}

void DrawBox
( std::ofstream& file,
  double hStart, double vStart, double hStop, double vStop,
  const std::string& drawColor )
{
    file << "\\draw[" << drawColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

} // anonymous namespace

#ifdef HAVE_QT5
template<typename Scalar>
void
HMat2d<Scalar>::DisplayRecursion
( Dense<double>* matrix, int mRatio, int nRatio ) const
{
    const int m = matrix->Height();
    const int n = matrix->Width();
    const int mBlock = Height();
    const int nBlock = Width();

    const int mStart = targetOffset_*mRatio;
    const int nStart = sourceOffset_*nRatio;
    const int mStop = (targetOffset_+mBlock)*mRatio;
    const int nStop = (sourceOffset_+nBlock)*nRatio;

    const double lowRankVal = 1;
    const double lowRankEmptyVal = 0.25;
    const double denseVal = -1;
    const double borderVal = 0;

    switch( block_.type )
    {
    case NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).DisplayRecursion( matrix, mRatio, nRatio );
        break;
    }
    case NODE_SYMMETRIC:
    {
        const NodeSymmetric& node = *block_.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->DisplayRecursion( matrix, mRatio, nRatio );
        break;
    }
    case LOW_RANK:
    {
        const int rank = block_.data.F->Rank();
        if( rank == 0 )
            FillBox( matrix, mStart, nStart, mStop, nStop, lowRankEmptyVal );
        else
            FillBox( matrix, mStart, nStart, mStop, nStop, lowRankVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;
    }
    case DENSE:
        FillBox( matrix, mStart, nStart, mStop, nStop, denseVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;
    }
}
#endif // ifdef HAVE_QT5

template<typename Scalar>
void
HMat2d<Scalar>::LatexStructureRecursion
( std::ofstream& file, int globalHeight ) const
{
    const double invScale = globalHeight; 
    const double hStart = sourceOffset_/invScale;
    const double hStop  = (sourceOffset_+Width())/invScale;
    const double vStart = (globalHeight-(targetOffset_ + Height()))/invScale;
    const double vStop  = (globalHeight-targetOffset_)/invScale;

    const std::string lowRankColor = "green";
    const std::string lowRankEmptyColor = "cyan";
    const std::string denseColor = "red";
    const std::string borderColor = "black";

    switch( block_.type )
    {
    case NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).LatexStructureRecursion( file, globalHeight );
        break;
    }
    case NODE_SYMMETRIC:
    {
        const NodeSymmetric& node = *block_.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->LatexStructureRecursion( file, globalHeight );
        break;
    }
    case LOW_RANK:
    {
        const int rank = block_.data.F->Rank();
        if( rank == 0 )
            FillBox( file, hStart, vStart, hStop, vStop, lowRankEmptyColor );
        else
            FillBox( file, hStart, vStart, hStop, vStop, lowRankColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
    case DENSE:
        FillBox( file, hStart, vStart, hStop, vStop, denseColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::MScriptStructureRecursion( std::ofstream& file ) const
{
    switch( block_.type )
    {
    case NODE:
    {
        file << "1 " 
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MScriptStructureRecursion( file );
        break;
    }
    case NODE_SYMMETRIC:
    {
        file << "1 " 
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        const NodeSymmetric& node = *block_.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->MScriptStructureRecursion( file );
        break;
    }
    case LOW_RANK:
        file << "5 " 
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        break;
    case DENSE:
        file << "20 " 
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        break;
    }
}

template class HMat2d<float>;
template class HMat2d<double>;
template class HMat2d<std::complex<float> >;
template class HMat2d<std::complex<double> >;

} // namespace dmhm
