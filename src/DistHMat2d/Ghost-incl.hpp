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
DistHMat2d<Scalar>::FormTargetGhostNodes()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FormTargetGhostNodes");
#endif
    RequireRoot();

    // Each level will have a set of target offsets where the structure
    // is known.
    Vector<std::set<int> > targetStructure( numLevels_ );
    FillTargetStructureRecursion( targetStructure );
    
    // Fill in the local ghosted structure (but without the ghosts' ranks)
    FindTargetGhostNodesRecursion( targetStructure, 0, 0 );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::FormSourceGhostNodes()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FormSourceGhostNodes");
#endif
    RequireRoot();

    // Each level will have a set of source offsets where the structure
    // is known.
    Vector<std::set<int> > sourceStructure( numLevels_ );
    FillSourceStructureRecursion( sourceStructure );
    
    // Fill in the local ghosted structure (but without the ghosts' ranks)
    FindSourceGhostNodesRecursion( sourceStructure, 0, 0 );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::PruneGhostNodes()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::PruneGhostNodes");
#endif
    switch( block_.type )
    {
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
        block_.Clear();
        break;

    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).PruneGhostNodes();
        break;
    }

    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::FillTargetStructureRecursion
( Vector<std::set<int> >& targetStructure ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FillTargetStructureRecursion");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    {
        targetStructure[level_].insert( targetOffset_ );

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        const Node& node = *block_.data.N;
        if( teamSize >= 4 )
        {
            const int subteam = teamRank / (teamSize/4);
            for( int t=0; t<4; ++t )
                node.Child( t, subteam ).FillTargetStructureRecursion
                ( targetStructure );
            for( int s=0; s<4; ++s )
                node.Child( subteam, s ).FillTargetStructureRecursion
                ( targetStructure );
        }
        else // teamSize == 2
        {
            if( teamRank == 0 )
            {
                // Upper half
                for( int t=0; t<2; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillTargetStructureRecursion
                        ( targetStructure );
                // Bottom-left block
                for( int t=2; t<4; ++t )
                    for( int s=0; s<2; ++s )
                        node.Child(t,s).FillTargetStructureRecursion
                        ( targetStructure );
            }
            else // teamRank == 1
            {
                // Upper-right block
                for( int t=0; t<2; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).FillTargetStructureRecursion
                        ( targetStructure );
                // Bottom half
                for( int t=2; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillTargetStructureRecursion
                        ( targetStructure );
            }
        }
        break;
    }

    case SPLIT_NODE:
    case NODE:
    {
        targetStructure[level_].insert( targetOffset_ );

        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FillTargetStructureRecursion( targetStructure );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    case DENSE:
        targetStructure[level_].insert( targetOffset_ );
        break;

    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::FillSourceStructureRecursion
( Vector<std::set<int> >& sourceStructure ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FillSourceStructureRecursion");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    {
        sourceStructure[level_].insert( sourceOffset_ );

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );
        const Node& node = *block_.data.N;
        if( teamSize >= 4 )
        {
            const int subteam = teamRank / (teamSize/4);
            for( int t=0; t<4; ++t )
                node.Child( t, subteam ).FillSourceStructureRecursion
                ( sourceStructure );
            for( int s=0; s<4; ++s )
                node.Child( subteam, s ).FillSourceStructureRecursion
                ( sourceStructure );
        }
        else // teamSize == 2
        {
            if( teamRank == 0 )
            {
                // Upper half
                for( int t=0; t<2; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillSourceStructureRecursion
                        ( sourceStructure );
                // Bottom-left block
                for( int t=2; t<4; ++t )
                    for( int s=0; s<2; ++s )
                        node.Child(t,s).FillSourceStructureRecursion
                        ( sourceStructure );
            }
            else // teamRank == 1
            {
                // Upper-right block
                for( int t=0; t<2; ++t )
                    for( int s=2; s<4; ++s )
                        node.Child(t,s).FillSourceStructureRecursion
                        ( sourceStructure );
                // Bottom half
                for( int t=2; t<4; ++t )
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).FillSourceStructureRecursion
                        ( sourceStructure );
            }
        }
        break;
    }

    case SPLIT_NODE:
    case NODE:
    {
        sourceStructure[level_].insert( sourceOffset_ );

        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FillSourceStructureRecursion( sourceStructure );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    case SPLIT_DENSE:
    case DENSE:
        sourceStructure[level_].insert( sourceOffset_ );
        break;

    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::FindTargetGhostNodesRecursion
( const Vector<std::set<int> >& targetStructure,
  int sourceRoot, int targetRoot )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FindTargetGhostNodesRecursion");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        Node& node = *block_.data.N;
        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize >= 4 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindTargetGhostNodesRecursion
                    ( targetStructure, 
                      sourceRoot+s*teamSize/4, targetRoot+t*teamSize/4 );
        }
        else // teamSize == 2
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindTargetGhostNodesRecursion
                    ( targetStructure,
                      sourceRoot+s/2, targetRoot+t/2 );
        }
        break;
    }

    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FindTargetGhostNodesRecursion
                ( targetStructure, sourceRoot, targetRoot );
        break;
    }

    case EMPTY:
    {
        if( !std::binary_search
            ( targetStructure[level_].begin(),
              targetStructure[level_].end(), targetOffset_ ) )
            break;
                               
        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );

        if( Admissible() )
        {
            if( teamSize >= 2 )
            {
                block_.type = DIST_LOW_RANK_GHOST;
                block_.data.DFG = new DistLowRankGhost;
                block_.data.DFG->rank = -1;
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                {
                    block_.type = LOW_RANK_GHOST;
                    block_.data.FG = new LowRankGhost;
                    block_.data.FG->rank = -1;
                }
                else
                {
                    block_.type = SPLIT_LOW_RANK_GHOST;
                    block_.data.SFG = new SplitLowRankGhost;
                    block_.data.SFG->rank = -1;
                }
            }
        }
        else if( numLevels_ > 1 )
        {
            block_.data.N = NewNode();
            Node& node = *block_.data.N;

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    int newSourceRoot, newTargetRoot;
                    if( teamSize >= 4 )
                    {
                        block_.type = DIST_NODE_GHOST;
                        newSourceRoot = sourceRoot + s*teamSize/4;
                        newTargetRoot = targetRoot + t*teamSize/4;
                    }
                    else if( teamSize == 2 )
                    {
                        block_.type = DIST_NODE_GHOST;
                        newSourceRoot = sourceRoot + s/2;
                        newTargetRoot = targetRoot + t/2;
                    }
                    else
                    {
                        block_.type = 
                            ( sourceRoot==targetRoot ? 
                              NODE_GHOST : SPLIT_NODE_GHOST );
                        newSourceRoot = sourceRoot;
                        newTargetRoot = targetRoot;
                    }
                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+(s/2), 2*yTarget_+(t/2),
                          *teams_, level_+1, false, false, 
                          newSourceRoot, newTargetRoot );
                    node.Child(t,s).FindTargetGhostNodesRecursion
                    ( targetStructure, newSourceRoot, newTargetRoot );
                }
            }
        }
        else
        {
            if( sourceRoot == targetRoot )
                block_.type = DENSE_GHOST;
            else
                block_.type = SPLIT_DENSE_GHOST;
        }
        break;
    }

    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::FindSourceGhostNodesRecursion
( const Vector<std::set<int> >& sourceStructure,
  int sourceRoot, int targetRoot )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::FindSourceGhostNodesRecursion");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    {
        Node& node = *block_.data.N;
        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize >= 4 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindSourceGhostNodesRecursion
                    ( sourceStructure, 
                      sourceRoot+s*teamSize/4, targetRoot+t*teamSize/4 );
        }
        else // teamSize == 2
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).FindSourceGhostNodesRecursion
                    ( sourceStructure,
                      sourceRoot+s/2, targetRoot+t/2 );
        }
        break;
    }

    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).FindSourceGhostNodesRecursion
                ( sourceStructure, sourceRoot, targetRoot );
        break;
    }

    case EMPTY:
    {
        if( !std::binary_search
            ( sourceStructure[level_].begin(),
              sourceStructure[level_].end(), sourceOffset_ ) )
            break;
                               
        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );

        if( Admissible() )
        {
            if( teamSize >= 2 )
            {
                block_.type = DIST_LOW_RANK_GHOST;
                block_.data.DFG = new DistLowRankGhost;
                block_.data.DFG->rank = -1;
            }
            else // teamSize == 1
            {
                if( sourceRoot == targetRoot )
                {
                    block_.type = LOW_RANK_GHOST;
                    block_.data.FG = new LowRankGhost;
                    block_.data.FG->rank = -1;
                }
                else
                {
                    block_.type = SPLIT_LOW_RANK_GHOST;
                    block_.data.SFG = new SplitLowRankGhost;
                    block_.data.SFG->rank = -1;
                }
            }
        }
        else if( numLevels_ > 1 )
        {
            block_.data.N = NewNode();
            Node& node = *block_.data.N;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    int newSourceRoot, newTargetRoot;
                    if( teamSize >= 4 )
                    {
                        block_.type = DIST_NODE_GHOST;
                        newSourceRoot = sourceRoot + s*teamSize/4;
                        newTargetRoot = targetRoot + t*teamSize/4;
                    }
                    else if( teamSize == 2 )
                    {
                        block_.type = DIST_NODE_GHOST;
                        newSourceRoot = sourceRoot + s/2;
                        newTargetRoot = targetRoot + t/2;
                    }
                    else
                    {
                        block_.type = 
                            ( sourceRoot==targetRoot ? 
                              NODE_GHOST : SPLIT_NODE_GHOST );
                        newSourceRoot = sourceRoot;
                        newTargetRoot = targetRoot;
                    }
                    node.children[s+4*t] = 
                        new DistHMat2d<Scalar>
                        ( numLevels_-1, maxRank_, stronglyAdmissible_,
                          sourceOffset_+sOffset, targetOffset_+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                          2*xSource_+(s&1), 2*xTarget_+(t&1),
                          2*ySource_+(s/2), 2*yTarget_+(t/2),
                          *teams_, level_+1, false, false, 
                          newSourceRoot, newTargetRoot );
                    node.Child(t,s).FindSourceGhostNodesRecursion
                    ( sourceStructure, newSourceRoot, newTargetRoot );
                }
            }
        }
        else
        {
            if( sourceRoot == targetRoot )
                block_.type = DENSE_GHOST;
            else
                block_.type = SPLIT_DENSE_GHOST;
        }
        break;
    }
    
    default:
        break;
    }
}

} // namespace dmhm
