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
#include "./RedistHMat3d-incl.hpp"
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
DistHMat3d<Scalar>::DistHMat3d
( const Teams& teams )
: numLevels_(0), maxRank_(0), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(false), 
  xSizeSource_(0), xSizeTarget_(0),
  ySizeSource_(0), ySizeTarget_(0), 
  zSizeSource_(0), zSizeTarget_(0), 
  xSource_(0), xTarget_(0),
  ySource_(0), yTarget_(0), 
  zSource_(0), zTarget_(0), 
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
DistHMat3d<Scalar>::DistHMat3d
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSize, int ySize, int zSize, const Teams& teams )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSize), xSizeTarget_(xSize),
  ySizeSource_(ySize), ySizeTarget_(ySize), 
  zSizeSource_(zSize), zSizeTarget_(zSize), 
  xSource_(0), xTarget_(0), 
  ySource_(0), yTarget_(0), 
  zSource_(0), zTarget_(0), 
  teams_(&teams), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::DistHMat3d");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
}

template<typename Scalar>
DistHMat3d<Scalar>::DistHMat3d
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSizeSource, int xSizeTarget, 
  int ySizeSource, int ySizeTarget,
  int zSizeSource, int zSizeTarget, const Teams& teams )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget), 
  zSizeSource_(zSizeSource), zSizeTarget_(zSizeTarget), 
  xSource_(0), xTarget_(0), 
  ySource_(0), yTarget_(0), 
  zSource_(0), zTarget_(0), 
  teams_(&teams), level_(0),
  inSourceTeam_(true), inTargetTeam_(true), 
  sourceRoot_(0), targetRoot_(0),
  haveDenseUpdate_(false), storedDenseUpdate_(false),
  beganRowSpaceComp_(false), finishedRowSpaceComp_(false),
  beganColSpaceComp_(false), finishedColSpaceComp_(false)
{ 
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::DistHMat3d");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
}
    
template<typename Scalar>
DistHMat3d<Scalar>::~DistHMat3d()
{ Clear(); }

template<typename Scalar>
void
DistHMat3d<Scalar>::Clear()
{ block_.Clear(); }

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalHeight() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalHeight");
#endif
    int localHeight;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        int zSize = zSizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localHeight = 0;
    return localHeight;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalHeightPartner() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalHeightPartner");
#endif
    int localHeightPartner;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        int zSize = zSizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeightPartner, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localHeightPartner = 0;
    return localHeightPartner;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalWidth() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalWidth");
#endif
    int localWidth;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        int ySize = ySizeSource_;
        int zSize = zSizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localWidth = 0;
    return localWidth;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalWidthPartner() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalWidthPartner");
#endif
    int localWidthPartner;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        int ySize = ySizeSource_;
        int zSize = zSizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidthPartner, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localWidthPartner = 0;
    return localWidthPartner;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalRow() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::FirstLocalRow");
#endif
    int firstLocalRow = 0;
    if( inTargetTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalRow, xSizeTarget_, ySizeTarget_, zSizeTarget_, 
          teamSize, teamRank );
    }
    return firstLocalRow;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalCol() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::FirstLocalCol");
#endif
    int firstLocalCol = 0;
    if( inSourceTeam_ )
    {
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalCol, xSizeSource_, ySizeSource_, zSizeSource_,
          teamSize, teamRank );
    }
    return firstLocalCol;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalXTarget() const
{ return xTarget_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalXSource() const
{ return xSource_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalYTarget() const
{ return yTarget_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalYSource() const
{ return ySource_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalZTarget() const
{ return zTarget_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::FirstLocalZSource() const
{ return zSource_ << (numLevels_-1); }

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalXTargetSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalXTargetSize");
#endif
    int xSize;
    if( inTargetTeam_ )
    {
        int localHeight;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        int zSize = zSizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        xSize = 0;
    return xSize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalXSourceSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalXSourceSize");
#endif
    int xSize;
    if( inSourceTeam_ )
    {
        int localWidth;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        xSize = xSizeSource_;
        int ySize = ySizeSource_;
        int zSize = zSizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        xSize = 0;
    return xSize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalYTargetSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalYTargetSize");
#endif
    int ySize;
    if( inTargetTeam_ )
    {
        int localHeight;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        ySize = ySizeTarget_;
        int zSize = zSizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        ySize = 0;
    return ySize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalYSourceSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalYSourceSize");
#endif
    int ySize;
    if( inSourceTeam_ )
    {
        int localWidth;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        ySize = ySizeSource_;
        int zSize = zSizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        ySize = 0;
    return ySize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalZTargetSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalZTargetSize");
#endif
    int zSize;
    if( inTargetTeam_ )
    {
        int localHeight;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeTarget_;
        int ySize = ySizeTarget_;
        zSize = zSizeTarget_;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        zSize = 0;
    return zSize;
}

template<typename Scalar>
int
DistHMat3d<Scalar>::LocalZSourceSize() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LocalZSourceSize");
#endif
    int zSize;
    if( inSourceTeam_ )
    {
        int localWidth;
        int teamSize = mpi::CommSize( teams_->Team(level_) );
        int teamRank = mpi::CommRank( teams_->Team(level_) );
        int xSize = xSizeSource_;
        int ySize = ySizeSource_;
        zSize = zSizeSource_;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        zSize = 0;
    return zSize;
}

template<typename Scalar>
void
DistHMat3d<Scalar>::RequireRoot() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::RequireRoot");
#endif
    if( level_ != 0 )
        throw std::logic_error("Not a root H-matrix as required.");
}

template<typename Scalar>
int
DistHMat3d<Scalar>::Rank() const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::Rank");
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
DistHMat3d<Scalar>::SetGhostRank( int rank ) 
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::SetGhostRank");
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
DistHMat3d<Scalar>::Admissible() const
{
    return Admissible
           ( xSource_, xTarget_, ySource_, yTarget_, zSource_, zTarget_ );
}

template<typename Scalar>
bool
DistHMat3d<Scalar>::Admissible
( int xSource, int xTarget, 
  int ySource, int yTarget,
  int zSource, int zTarget ) const
{
    if( stronglyAdmissible_ )
        return std::max(std::max(std::abs(xSource-xTarget),
            std::abs(ySource-yTarget)), std::abs(zSource-zTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget || zSource != zTarget;
}

#ifdef HAVE_QT5
template<typename Scalar>
void
DistHMat3d<Scalar>::DisplayLocal( std::string title ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::DisplayLocal");
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
DistHMat3d<Scalar>::LatexLocalStructure( const std::string basename ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::LatexLocalStructure");
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
DistHMat3d<Scalar>::MScriptLocalStructure( const std::string basename ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MScriptLocalStructure");
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
DistHMat3d<Scalar>::BlockTypeString( BlockType type )
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
DistHMat3d<Scalar>::DistHMat3d()
: numLevels_(0), maxRank_(0), 
  sourceOffset_(0), targetOffset_(0), 
  stronglyAdmissible_(false), 
  xSizeSource_(0), xSizeTarget_(0), 
  ySizeSource_(0), ySizeTarget_(0),
  zSizeSource_(0), zSizeTarget_(0),
  xSource_(0), xTarget_(0), 
  ySource_(0), yTarget_(0), 
  zSource_(0), zTarget_(0), 
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
DistHMat3d<Scalar>::DistHMat3d
( int numLevels, int maxRank, bool stronglyAdmissible,
  int sourceOffset, int targetOffset,
  int xSizeSource, int xSizeTarget, 
  int ySizeSource, int ySizeTarget,
  int zSizeSource, int zSizeTarget,
  int xSource, int xTarget, 
  int ySource, int yTarget,
  int zSource, int zTarget,
  const Teams& teams, int level, 
  bool inSourceTeam, bool inTargetTeam, 
  int sourceRoot, int targetRoot )
: numLevels_(numLevels), maxRank_(maxRank), 
  sourceOffset_(sourceOffset), targetOffset_(targetOffset), 
  stronglyAdmissible_(stronglyAdmissible), 
  xSizeSource_(xSizeSource), xSizeTarget_(xSizeTarget),
  ySizeSource_(ySizeSource), ySizeTarget_(ySizeTarget), 
  zSizeSource_(zSizeSource), zSizeTarget_(zSizeTarget), 
  xSource_(xSource), xTarget_(xTarget),
  ySource_(ySource), yTarget_(yTarget), 
  zSource_(zSource), zTarget_(zTarget), 
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
DistHMat3d<Scalar>::BuildTree()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::BuildTree");
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
            block_.data.DF->VLocal.Resize( LocalWidth(),  0 );
        }
        else if( sourceRoot_ == targetRoot_ )
        {
            block_.type = LOW_RANK;
            block_.data.F = new LowRank<Scalar>;
            block_.data.F->U.Resize( Height(), 0 );
            block_.data.F->V.Resize( Width(),  0 );
        }
        else
        {
            block_.type = SPLIT_LOW_RANK;
            block_.data.SF = new SplitLowRank;
            block_.data.SF->rank = 0;
            if( inTargetTeam_ )
                block_.data.SF->D.Resize( Height(), 0 );
            else
                block_.data.SF->D.Resize( Width(), 0 );
        }
    }
    else if( numLevels_ > 1 ) // recurse
    {
        block_.data.N = NewNode();
        Node& node = *block_.data.N;        

        if( teamSize >= 8 )
        {
            block_.type = DIST_NODE;

            const int subteam = teamRank/(teamSize/8);
            for( int t=0,tOffset=0; t<8; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = targetRoot_ + t*(teamSize/8);
                for( int s=0,sOffset=0; s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = sourceRoot_ + s*(teamSize/8);

                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[s/4], node.zTargetSizes[t/4],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && (s==subteam),
                          inTargetTeam_ && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else if( teamSize == 4 )
        {
            block_.type = DIST_NODE;

            for( int t=0,tOffset=0; t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[s/4], node.zTargetSizes[t/4],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/2) ),
                          inTargetTeam_ && ( teamRank == (t/2) ),
                          sourceRoot_+(s/2), targetRoot_+(t/2) );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else if( teamSize == 2 )
        {
            block_.type = DIST_NODE;

            for( int t=0,tOffset=0; t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[s/4], node.zTargetSizes[t/4],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
                          *teams_, level_+1,
                          inSourceTeam_ && ( teamRank == (s/4) ),
                          inTargetTeam_ && ( teamRank == (t/4) ),
                          sourceRoot_+(s/4), targetRoot_+(t/4) );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else // teamSize == 1 
        {
            block_.type = ( sourceRoot_==targetRoot_ ? NODE : SPLIT_NODE );

            for( int t=0,tOffset=0; t<8; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<8; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+8*t] =
                        new DistHMat3d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[(s/2)&1], node.yTargetSizes[(t/2)&1],
                          node.zSourceSizes[s/4], node.zTargetSizes[t/4],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+((s/2)&1), 2*yTarget_+((t/2)&1),
                          2*zSource_+(s/4), 2*zTarget_+(t/4),
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
                block_.data.SD->D.Resize( Height(), Width() );
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
DistHMat3d<Scalar>::DisplayLocalRecursion
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
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
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
DistHMat3d<Scalar>::LatexLocalStructureRecursion
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
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
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
DistHMat3d<Scalar>::MScriptLocalStructureRecursion( std::ofstream& file ) const
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
        for( int t=0; t<8; ++t )
            for( int s=0; s<8; ++s )
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

template class DistHMat3d<float>;
template class DistHMat3d<double>;
template class DistHMat3d<std::complex<float> >;
template class DistHMat3d<std::complex<double> >;

} // namespace dmhm