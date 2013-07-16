/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

#include "./Add-incl.hpp"
#include "./Adjoint-incl.hpp"
#include "./Conjugate-incl.hpp"
#include "./Copy-incl.hpp"
#include "./Ghost-incl.hpp"
#include "./Invert-incl.hpp"
#include "./MultiplyDense-incl.hpp"
#include "./MultiplyHMat-incl.hpp"
#include "./MultiplyVector-incl.hpp"
#include "./Norm-incl.hpp"
#include "./RedistHMat2d-incl.hpp"
#include "./Scale-incl.hpp"
#include "./SetToRandom-incl.hpp"
#include "./Transpose-incl.hpp"
#ifdef HAVE_QT5
 #include <QApplication>
#endif

namespace dmhm {

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d
( const Teams& teams )
: numLevels_(0), maxRank_(0), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(false), 
  xSizeSource_(0), xSizeTarget_(0),
  ySizeSource_(0), ySizeTarget_(0), 
  xSource_(0), xTarget_(0), ySource_(0), yTarget_(0), 
  teams_(&teams), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
    block_.type = EMPTY;
}

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSize, int ySize, const Teams& teams )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSize), xSizeTarget_(xSize),
  ySizeSource_(ySize), ySizeTarget_(ySize), 
  xSource_(0), xTarget_(0), ySource_(0), yTarget_(0), 
  teams_(&teams), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::DistHMat2d");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
}

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSizeSource, int xSizeTarget, 
  int ySizeSource, int ySizeTarget, const Teams& teams )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget), 
  xSource_(0), xTarget_(0), ySource_(0), yTarget_(0), 
  teams_(&teams), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::DistHMat2d");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
}
    
template<typename Scalar>
DistHMat2d<Scalar>::~DistHMat2d()
{ Clear(); }

template<typename Scalar>
void
DistHMat2d<Scalar>::Clear()
{ block_.Clear(); }

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalHeight() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalHeight");
#endif
    int localHeight;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, teamSize, teamRank );
    }
    else
        localHeight = 0;
    return localHeight;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalHeightPartner() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalHeightPartner");
#endif
    int localHeightPartner;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeightPartner, xSize, ySize, teamSize, teamRank );
    }
    else
        localHeightPartner = 0;
    return localHeightPartner;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalWidth() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalWidth");
#endif
    int localWidth;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        int ySize = ySizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, teamSize, teamRank );
    }
    else
        localWidth = 0;
    return localWidth;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalWidthPartner() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalWidthPartner");
#endif
    int localWidthPartner;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        int ySize = ySizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidthPartner, xSize, ySize, teamSize, teamRank );
    }
    else
        localWidthPartner = 0;
    return localWidthPartner;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalRow() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FirstLocalRow");
#endif
    int firstLocalRow = 0;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalRow, xSizeTarget_, ySizeTarget_, teamSize, teamRank );
    }
    return firstLocalRow;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalCol() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FirstLocalCol");
#endif
    int firstLocalCol = 0;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalCol, xSizeSource_, ySizeSource_, teamSize, teamRank );
    }
    return firstLocalCol;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalXTarget() const
{ return xTarget_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalXSource() const
{ return xSource_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalYTarget() const
{ return yTarget_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat2d<Scalar>::FirstLocalYSource() const
{ return ySource_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalXTargetSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalXTargetSize");
#endif
    int xSize;
    if( inTargetTeam_ )
    {
        int localHeight;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, teamSize, teamRank );
    }
    else
        xSize = 0;
    return xSize;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalXSourceSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalXSourceSize");
#endif
    int xSize;
    if( inSourceTeam_ )
    {
        int localWidth;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        xSize = xSizeSource_;
        int ySize = ySizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, teamSize, teamRank );
    }
    else
        xSize = 0;
    return xSize;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalYTargetSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalYTargetSize");
#endif
    int ySize;
    if( inTargetTeam_ )
    {
        int localHeight;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        ySize = ySizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, teamSize, teamRank );
    }
    else
        ySize = 0;
    return ySize;
}

template<typename Scalar>
int
DistHMat2d<Scalar>::LocalYSourceSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LocalYSourceSize");
#endif
    int ySize;
    if( inSourceTeam_ )
    {
        int localWidth;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        ySize = ySizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, teamSize, teamRank );
    }
    else
        ySize = 0;
    return ySize;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::RequireRoot() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::RequireRoot");
#endif
    if( level_ != 0 )
        throw std::logic_error("Not a root H-matrix as required.");
}

template<typename Scalar>
int
DistHMat2d<Scalar>::Rank() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Rank");
#endif
    int rank = 0; // initialize to avoid compiler warnings
    switch( block_.type )
    {
    case DIST_LOW_RANK:
        rank = block_.data.DF->rank;
        break;
    case DIST_LOW_RANK_GHOST:
        rank = block_.data.DFG->rank;
        break;
    case SPLIT_LOW_RANK:
        rank = block_.data.SF->rank;
        break;
    case SPLIT_LOW_RANK_GHOST:
        rank = block_.data.SFG->rank;
        break;
    case LOW_RANK:
        rank = block_.data.F->Rank();
        break;
    case LOW_RANK_GHOST:
        rank = block_.data.FG->rank;
        break;
    default:
        throw std::logic_error("Can only request rank of low-rank blocks");
        break;
    }
    return rank;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::SetGhostRank( int rank ) 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::SetGhostRank");
#endif
    switch( block_.type )
    {
    case DIST_LOW_RANK_GHOST:
        block_.data.DFG->rank = rank;
        break;
    case SPLIT_LOW_RANK_GHOST:
        block_.data.SFG->rank = rank;
        break;
    case LOW_RANK_GHOST:
        block_.data.FG->rank = rank;
        break;
    default:
#ifndef RELEASE
        throw std::logic_error
        ("Can only set ghost rank of ghost low-rank blocks");
#endif
        break;
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//
template<typename Scalar>
bool
DistHMat2d<Scalar>::Admissible() const
{
    return Admissible( xSource_, xTarget_, ySource_, yTarget_ );
}

template<typename Scalar>
bool
DistHMat2d<Scalar>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( stronglyAdmissible_ )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

#ifdef HAVE_QT5
template<typename Scalar>
void
DistHMat2d<Scalar>::DisplayLocal( std::string title ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::DisplayLocal");
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
    DisplayLocalRecursion( A, mRatio, nRatio );

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
DistHMat2d<Scalar>::LatexLocalStructure( const std::string basename ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::LatexLocalStructure");
#endif
    mpi::Comm comm = teams_->Team( 0 );
    const int commRank = mpi::CommRank( comm );

    std::ostringstream os;
    os << basename << "-" << commRank << ".tex";
    std::ofstream file( os.str().c_str() );

    double scale = 12.8;
    file << "\\documentclass[11pt]{article}\n"
         << "\\usepackage{tikz}\n"
         << "\\begin{document}\n"
         << "\\begin{center}\n"
         << "\\begin{tikzpicture}[scale=" << scale << "]\n";
    LatexLocalStructureRecursion( file, Height() );
    file << "\\end{tikzpicture}\n"
         << "\\end{center}\n"
         << "\\end{document}" << std::endl;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MScriptLocalStructure( const std::string basename ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MScriptLocalStructure");
#endif
    mpi::Comm comm = teams_->Team( 0 );
    const int commRank = mpi::CommRank( comm );

    std::ostringstream os;
    os << basename << "-" << commRank << ".dat";
    std::ofstream file( os.str().c_str() );
    MScriptLocalStructureRecursion( file );
}
//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar>
const std::string
DistHMat2d<Scalar>::BlockTypeString( BlockType type )
{
    std::string s;
    switch( type )
    {
    case DIST_NODE:            s = "DIST_NODE";            break;
    case DIST_NODE_GHOST:      s = "DIST_NODE_GHOST";      break;
    case SPLIT_NODE:           s = "SPLIT_NODE";           break;
    case SPLIT_NODE_GHOST:     s = "SPLIT_NODE_GHOST";     break;
    case NODE:                 s = "NODE";                 break;
    case NODE_GHOST:           s = "NODE_GHOST";           break;
    case DIST_LOW_RANK:        s = "DIST_LOW_RANK";        break;
    case DIST_LOW_RANK_GHOST:  s = "DIST_LOW_RANK_GHOST";  break;
    case SPLIT_LOW_RANK:       s = "SPLIT_LOW_RANK";       break;
    case SPLIT_LOW_RANK_GHOST: s = "SPLIT_LOW_RANK_GHOST"; break;
    case LOW_RANK:             s = "LOW_RANK";             break;
    case LOW_RANK_GHOST:       s = "LOW_RANK_GHOST";       break;
    case SPLIT_DENSE:          s = "SPLIT_DENSE";          break;
    case SPLIT_DENSE_GHOST:    s = "SPLIT_DENSE_GHOST";    break;
    case DENSE:                s = "DENSE";                break;
    case DENSE_GHOST:          s = "DENSE_GHOST";          break;
    case EMPTY:                s = "EMPTY";                break;
    }
    return s;
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d()
: numLevels_(0), maxRank_(0), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(false), 
  xSizeSource_(0), xSizeTarget_(0), 
  ySizeSource_(0), ySizeTarget_(0),
  xSource_(0), xTarget_(0), 
  ySource_(0), yTarget_(0), 
  teams_(0), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
    block_.type = EMPTY;
}

template<typename Scalar>
DistHMat2d<Scalar>::DistHMat2d
( int numLevels, int maxRank, bool stronglyAdmissible,
  int sourceOffset, int targetOffset,
  int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
  int xSource, int xTarget, int ySource, int yTarget,
  const Teams& teams, int level, 
  bool inSourceTeam, bool inTargetTeam, 
  int sourceRoot, int targetRoot )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(sourceOffset), targetOffset_(targetOffset), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget), 
  xSource_(xSource), xTarget_(xTarget),
  ySource_(ySource), yTarget_(yTarget), 
  teams_(&teams), level_(level),
  inSourceTeam_(inSourceTeam), inTargetTeam_(inTargetTeam),
  sourceRoot_(sourceRoot), targetRoot_(targetRoot),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
    block_.type = EMPTY;
}

template<typename Scalar>
void
DistHMat2d<Scalar>::BuildTree()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::BuildTree");
#endif
    mpi::Comm team = teams_->Team(level_);
    const int teamSize = mpi::CommSize( team );
    const int teamRank = mpi::CommRank( team );
    if( !inSourceTeam_ && !inTargetTeam_ )
        block_.type = EMPTY;
    else if( Admissible() ) // low rank
    {
        if( teamSize > 1 )
        {
            block_.type = DIST_LOW_RANK;
            block_.data.DF = new DistLowRank;
            block_.data.DF->rank = 0;
            block_.data.DF->ULocal.Resize( LocalHeight(), 0 );
            block_.data.DF->ULocal.Init();
            block_.data.DF->VLocal.Resize( LocalWidth(),  0 );
            block_.data.DF->VLocal.Init();
        }
        else if( sourceRoot_ == targetRoot_ )
        {
            block_.type = LOW_RANK;
            block_.data.F = new LowRank<Scalar>;
            block_.data.F->U.Resize( Height(), 0 );
            block_.data.F->U.Init();
            block_.data.F->V.Resize( Width(),  0 );
            block_.data.F->V.Init();
        }
        else
        {
            block_.type = SPLIT_LOW_RANK;
            block_.data.SF = new SplitLowRank;
            block_.data.SF->rank = 0;
            if( inTargetTeam_ )
            {
                block_.data.SF->D.Resize( Height(), 0 );
                block_.data.SF->D.Init();
            }
            else
            {
                block_.data.SF->D.Resize( Width(), 0 );
                block_.data.SF->D.Init();
            }
        }
    }
    else if( numLevels_ > 1 ) // recurse
    {
        block_.data.N = NewNode();
        Node& node = *block_.data.N;        

        if( teamSize >= 4 )
        {
            block_.type = DIST_NODE;

            const int subteam = teamRank/(teamSize/4);
            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_+1,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_+1,
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else if( teamSize == 2 )
        {
            block_.type = DIST_NODE;

            const bool inUpperTeam = ( teamRank == 1 );
            const bool inLeftSourceTeam = ( !inUpperTeam && inSourceTeam_ );
            const bool inRightSourceTeam = ( inUpperTeam && inSourceTeam_ );
            const bool inTopTargetTeam = ( !inUpperTeam && inTargetTeam_ );
            const bool inBottomTargetTeam = ( inUpperTeam && inTargetTeam_ );

            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_,
                          *teams_, level_+1,
                          inLeftSourceTeam, inTopTargetTeam,
                          sourceRoot_, targetRoot_ );
                    node.Child(t,s).BuildTree();
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_,
                          *teams_, level_+1,
                          inRightSourceTeam, inTopTargetTeam,
                          sourceRoot_+1, targetRoot_ );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_, 2*yTarget_+1,
                          *teams_, level_+1,
                          inLeftSourceTeam, inBottomTargetTeam,
                          sourceRoot_, targetRoot_+1 );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+1, 2*yTarget_+1,
                          *teams_, level_+1,
                          inRightSourceTeam, inBottomTargetTeam,
                          sourceRoot_+1, targetRoot_+1 );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else // teamSize == 1 
        {
            block_.type = ( sourceRoot_==targetRoot_ ? NODE : SPLIT_NODE );

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+(s/2), 2*yTarget_+(t/2),
                          *teams_, level_+1,
                          inSourceTeam_, inTargetTeam_,
                          sourceRoot_, targetRoot_ );
                    node.Child(t,s).BuildTree();
                }
            }
        }
    }
    else // dense
    {
        if( sourceRoot_ == targetRoot_ )
        {
            block_.type = DENSE;
            block_.data.D = new Dense<Scalar>( Height(), Width() );
        }
        else
        {
            block_.type = SPLIT_DENSE;
            block_.data.SD = new SplitDense;
            if( inSourceTeam_ )
            {
                block_.data.SD->D.Resize( Height(), Width() );
                block_.data.SD->D.Init();
            }
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
DistHMat2d<Scalar>::DisplayLocalRecursion
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
    const double lowRankEmptyVal = 0.5;
    const double lowRankGhostVal = 0.25;
    const double denseVal = -1;
    const double denseGhostVal = -0.5;
    const double borderVal = 0;

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).DisplayLocalRecursion( matrix, mRatio, nRatio );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        const int rank = Rank();
        if( rank == 0 )
            FillBox( matrix, mStart, nStart, mStop, nStop, lowRankEmptyVal );
        else
            FillBox( matrix, mStart, nStart, mStop, nStop, lowRankVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;
    }

    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
        FillBox( matrix, mStart, nStart, mStop, nStop, lowRankGhostVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;

    case SPLIT_DENSE:
    case DENSE:
        FillBox( matrix, mStart, nStart, mStop, nStop, denseVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;
    
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
        FillBox( matrix, mStart, nStart, mStop, nStop, denseGhostVal );
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;

    case EMPTY:
        DrawBox( matrix, mStart, nStart, mStop, nStop, borderVal );
        break;
    }
}
#endif // ifdef HAVE_QT5

template<typename Scalar>
void
DistHMat2d<Scalar>::LatexLocalStructureRecursion
( std::ofstream& file, int globalHeight ) const
{
    const double invScale = globalHeight;
    const double hStart = sourceOffset_/invScale;
    const double hStop  = (sourceOffset_+Width())/invScale;
    const double vStart = (globalHeight-(targetOffset_ + Height()))/invScale;
    const double vStop  = (globalHeight-targetOffset_)/invScale;

    const std::string lowRankColor = "green";
    const std::string lowRankEmptyColor = "cyan";
    const std::string lowRankGhostColor = "lightgray";
    const std::string denseColor = "red";
    const std::string denseGhostColor = "gray";
    const std::string borderColor = "black";

    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).LatexLocalStructureRecursion
                ( file, globalHeight );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        const int rank = Rank();
        if( rank == 0 )
            FillBox( file, hStart, vStart, hStop, vStop, lowRankEmptyColor );
        else
            FillBox( file, hStart, vStart, hStop, vStop, lowRankColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }

    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
        FillBox( file, hStart, vStart, hStop, vStop, lowRankGhostColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;

    case SPLIT_DENSE:
    case DENSE:
        FillBox( file, hStart, vStart, hStop, vStop, denseColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
        FillBox( file, hStart, vStart, hStop, vStop, denseGhostColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;

    case EMPTY:
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MScriptLocalStructureRecursion( std::ofstream& file ) const
{
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        file << "1 " 
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MScriptLocalStructureRecursion( file );
        break;
    }

    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK:
    case LOW_RANK_GHOST:
        file << "5 "
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        break;

    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
        file << "20 "
             << targetOffset_ << " " << sourceOffset_ << " "
             << Height() << " " << Width() << "\n";
        break;

    case EMPTY:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MemoryInfo
( double& numBasic, double& numNode, double& numNodeTmp,
  double& numLowRank, double& numLowRankTmp,
  double& numDense, double& numDenseTmp )const
{
    numBasic += sizeof( numLevels_ );
    numBasic += sizeof( maxRank_ );
    numBasic += sizeof( targetOffset_ );
    numBasic += sizeof( sourceOffset_ );
    numBasic += sizeof( stronglyAdmissible_ );
    numBasic += sizeof( xSizeTarget_ );
    numBasic += sizeof( ySizeTarget_ );
    numBasic += sizeof( xSizeSource_ );
    numBasic += sizeof( ySizeSource_ );
    numBasic += sizeof( xTarget_ );
    numBasic += sizeof( yTarget_ );
    numBasic += sizeof( xSource_ );
    numBasic += sizeof( ySource_ );
    numBasic += sizeof( teams_ );
    numBasic += sizeof( level_ );
    numBasic += sizeof( inTargetTeam_ );
    numBasic += sizeof( inSourceTeam_ );
    numBasic += sizeof( targetRoot_ );
    numBasic += sizeof( sourceRoot_ );
    numBasic += sizeof( block_.type );
    numBasic += sizeof( haveDenseUpdate_ );
    numBasic += sizeof( storedDenseUpdate_ );
    numBasic += sizeof( beganRowSpaceComp_ );
    numBasic += sizeof( finishedRowSpaceComp_ );
    numBasic += sizeof( beganColSpaceComp_ );
    numBasic += sizeof( finishedColSpaceComp_ );
    
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        double& num = numNodeTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);

        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MemoryInfo
                ( numBasic, numNode, numNodeTmp,
                  numLowRank, numLowRankTmp, 
                  numDense, numDenseTmp );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *block_.data.DF;
        numLowRank += DF.ULocal.Size()*sizeof(Scalar);
        numLowRank += DF.VLocal.Size()*sizeof(Scalar);

        double& num = numLowRankTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *block_.data.SF;
        numLowRank += SF.D.Size()*sizeof(Scalar);

        double& num = numLowRankTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);

        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar> &F = *block_.data.F;
        numLowRank += F.U.Size()*sizeof(Scalar);
        numLowRank += F.V.Size()*sizeof(Scalar);

        double& num = numLowRankTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);

        break;
    }
    case SPLIT_DENSE:
    {
        SplitDense& SD = *block_.data.SD;
        numDense += SD.D.Size()*sizeof(Scalar);

        double& num = numDenseTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);

        break;
    }
    case DENSE:
    {
        Dense<Scalar>& D = *block_.data.D;
        numDense += D.Size()*sizeof(Scalar);

        double& num = numDenseTmp;
        num += UMap_.EntrySize()*sizeof(Scalar);
        num += VMap_.EntrySize()*sizeof(Scalar);
        num += ZMap_.EntrySize()*sizeof(Scalar);
        num += colXMap_.EntrySize()*sizeof(Scalar);
        num += rowXMap_.EntrySize()*sizeof(Scalar);
        num += HUMap_.EntrySize()*sizeof(Scalar);
        num += HVMap_.EntrySize()*sizeof(Scalar);
        num += HZMap_.EntrySize()*sizeof(Scalar);
        num += colSqrMap_.EntrySize()*sizeof(Scalar);
        num += rowSqrMap_.EntrySize()*sizeof(Scalar);
        num += colSqrEigMap_.EntrySize()*sizeof(Real);
        num += rowSqrEigMap_.EntrySize()*sizeof(Real);
        num += USqr_.Size()*sizeof(Scalar);
        num += VSqr_.Size()*sizeof(Scalar);
        num += USqrEig_.size()*sizeof(Real);
        num += VSqrEig_.size()*sizeof(Real);
        num += BSqrU_.Size()*sizeof(Scalar);
        num += BSqrVH_.Size()*sizeof(Scalar);
        num += BSigma_.size()*sizeof(Real);
        num += BL_.Size()*sizeof(Scalar);
        num += BR_.Size()*sizeof(Scalar);
        num += D_.Size()*sizeof(Scalar);
        num += SFD_.Size()*sizeof(Scalar);
        num += colPinvMap_.EntrySize()*sizeof(Scalar);
        num += rowPinvMap_.EntrySize()*sizeof(Scalar);
        num += BLMap_.EntrySize()*sizeof(Scalar);
        num += BRMap_.EntrySize()*sizeof(Scalar);

        break;
    }
    default:
        break;
    }

}

template<typename Scalar>
void
DistHMat2d<Scalar>::PrintMemoryInfo
( const std::string tag, std::ostream& os ) const
{
    double numBasic = 0.0, numNode = 0.0, numNodeTmp = 0.0,
           numLowRank = 0.0, numLowRankTmp = 0.0, 
           numDense = 0.0, numDenseTmp = 0.0;
    MemoryInfo
    ( numBasic, numNode, numNodeTmp, numLowRank, numLowRankTmp,
      numDense, numDenseTmp );
    mpi::Comm team = teams_->Team(0);
    const int teamRank = mpi::CommRank( team );
    os << "Process " << teamRank << ": " << tag << "\n";
    os << "Basic      Memory Usage: " << numBasic << "\n";
    os << "Node       Memory Usage: " << numNode << "\n";
    os << "NodeTmp    Memory Usage: " << numNodeTmp << "\n";
    os << "LowRank    Memory Usage: " << numLowRank << "\n";
    os << "LowRankTmp Memory Usage: " << numLowRankTmp << "\n";
    os << "Dense      Memory Usage: " << numDense << "\n";
    os << "DenseTmp   Memory Usage: " << numDenseTmp << "\n";
    os << "Matrix     Memory Usage: " << numBasic+numNode+numLowRank+numDense << "\n";
    os << "Temporary  Memory Usage: " << numNodeTmp+numLowRankTmp+numDenseTmp << "\n";
    os << "Total      Memory Usage: " << numBasic+numNode+numLowRank+numDense
                                        +numNodeTmp+numLowRankTmp+numDenseTmp << "\n";
    os << "\n";
    os.flush();
}

template class DistHMat2d<float>;
template class DistHMat2d<double>;
template class DistHMat2d<std::complex<float> >;
template class DistHMat2d<std::complex<double> >;

} // namespace dmhm
