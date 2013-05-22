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
DistHMat2d<Scalar>::Multiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Multiply");
#endif
    yLocal.Resize( LocalHeight() );
    Multiply( alpha, xLocal, Scalar(0), yLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiply");
#endif
    yLocal.Resize( LocalWidth() );
    TransposeMultiply( alpha, xLocal, Scalar(0), yLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiply");
#endif
    yLocal.Resize( LocalWidth() );
    AdjointMultiply( alpha, xLocal, Scalar(0), yLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::Multiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Multiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );
    MultiplyVectorContext context;
    MultiplyVectorInitialize( context );
    MultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    MultiplyVectorSums( context );
    MultiplyVectorPassData( context );
    MultiplyVectorBroadcasts( context );

    MultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );

    MultiplyVectorContext context;
    TransposeMultiplyVectorInitialize( context );
    TransposeMultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    TransposeMultiplyVectorSums( context );
    TransposeMultiplyVectorPassData( context, xLocal );
    TransposeMultiplyVectorBroadcasts( context );

    TransposeMultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& xLocal, 
  Scalar beta,        Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiply");
#endif
    RequireRoot();
    hmat_tools::Scale( beta, yLocal );

    MultiplyVectorContext context;
    AdjointMultiplyVectorInitialize( context );
    AdjointMultiplyVectorPrecompute( context, alpha, xLocal, yLocal );

    AdjointMultiplyVectorSums( context );
    AdjointMultiplyVectorPassData( context, xLocal );
    AdjointMultiplyVectorBroadcasts( context );

    AdjointMultiplyVectorPostcompute( context, alpha, xLocal, yLocal );
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorInitialize");
#endif
    context.Clear();
    switch( block_.type )
    {
    case DIST_NODE:
    {
        context.block.type = DIST_NODE;
        context.block.data.DN = new typename MultiplyVectorContext::DistNode;

        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case SPLIT_NODE:
    {
        context.block.type = SPLIT_NODE;
        context.block.data.SN = new typename MultiplyVectorContext::SplitNode;

        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorInitialize
                ( nodeContext.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
        context.block.type = DIST_LOW_RANK;
        context.block.data.z = new Vector<Scalar>;
        break;
    case SPLIT_LOW_RANK:
        context.block.type = SPLIT_LOW_RANK;
        context.block.data.z = new Vector<Scalar>;
        break;
    case SPLIT_DENSE:
        context.block.type = SPLIT_DENSE;
        context.block.data.z = new Vector<Scalar>;
        break;

    default:
        context.block.type = EMPTY;
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyVectorInitialize( context );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorInitialize
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorInitialize");
#endif
    // The non-transposed initialization is identical
    MultiplyVectorInitialize( context );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPrecompute");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *block_.data.N;

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyVectorPrecompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inTargetTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block 
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=0,sOffset=0; s<2;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            xLocalSub.LockedView
                            ( xLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the lower-left block
                    yLocalSub.View( yLocal, 0, 0 );
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        xLocalSub.LockedView
                        ( xLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2; t<4; ++t )
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }

                }
                else // teamRank == 1
                {
                    // Bottom-right block
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=2,sOffset=0; s<4;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            xLocalSub.LockedView
                            ( xLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Upper-right block
                    yLocalSub.View( yLocal, 0, 0 );
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        xLocalSub.LockedView
                        ( xLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<2; ++t )
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else
            {
                // Only split xLocal
                Vector<Scalar> xLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the left half
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        xLocalSub.LockedView
                        ( xLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the right half
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        xLocalSub.LockedView
                        ( xLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).MultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
            for( int t=0; t<4; ++t )
                node.Child(t,s).MultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocal );
        }
        break;
    }
    case NODE:
    {
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MultiplyVectorPrecompute
                ( context, alpha, xLocalSub, yLocalSub );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // Form z := alpha VLocal^[T/H] xLocal
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        const char option = 'T';
        blas::Gemv
        ( option, DF.VLocal.Height(), DF.rank, 
          alpha,     DF.VLocal.LockedBuffer(), DF.VLocal.LDim(), 
                     xLocal.LockedBuffer(),    1,
          Scalar(0), z.Buffer(),               1 );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::TransposeMultiply( alpha, SF.D, xLocal, z );
        break;
    }
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const LowRank<Scalar>& F = *block_.data.F;
        hmat_tools::Multiply( alpha, F, xLocal, Scalar(1), yLocal );
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDense& SD = *block_.data.SD;
        Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::Multiply( alpha, SD.D, xLocal, z );
        break;
    }
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *block_.data.D;
        hmat_tools::Multiply( alpha, D, xLocal, Scalar(1), yLocal );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPrecompute");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *block_.data.N;

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPrecompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inSourceTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0,sOffset=0; s<2;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            yLocalSub.View
                            ( yLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).TransposeMultiplyVectorPrecompute  
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the upper-right block
                    yLocalSub.View( yLocal, 0, 0 );
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=2; s<4; ++s )
                            node.Child(t,s).TransposeMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Bottom-right block
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2,tOffset=0; t<4;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).TransposeMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Bottom-left block
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<2; ++s )
                            node.Child(t,s).TransposeMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else // !inSourceTeam_
            {
                // Only split xLocal
                Vector<Scalar> xLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper half 
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).TransposeMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the bottom half
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).TransposeMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocal );
        }
        break;
    }
    case NODE:
    {
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).TransposeMultiplyVectorPrecompute
                ( context, alpha, xLocalSub, yLocalSub );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // Form z := alpha ULocal^T xLocal
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        blas::Gemv
        ( 'T', DF.ULocal.Height(), DF.rank, 
          alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                     xLocal.LockedBuffer(),    1,
          Scalar(0), z.Buffer(),               1 );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::TransposeMultiply( alpha, SF.D, xLocal, z );
        break;
    }
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const LowRank<Scalar>& F = *block_.data.F;
        hmat_tools::TransposeMultiply( alpha, F, xLocal, Scalar(1), yLocal );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *block_.data.D;
        hmat_tools::TransposeMultiply( alpha, D, xLocal, Scalar(1), yLocal );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorPrecompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorPrecompute");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        const Node& node = *block_.data.N;

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).AdjointMultiplyVectorPrecompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inSourceTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0,sOffset=0; s<2;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            yLocalSub.View
                            ( yLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).AdjointMultiplyVectorPrecompute  
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the upper-right block
                    yLocalSub.View( yLocal, 0, 0 );
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=2; s<4; ++s )
                            node.Child(t,s).AdjointMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Bottom-right block
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2,tOffset=0; t<4;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).AdjointMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Bottom-left block
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<2; ++s )
                            node.Child(t,s).AdjointMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else // !inSourceTeam_
            {
                // Only split xLocal
                Vector<Scalar> xLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper half 
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).AdjointMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the bottom half
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        xLocalSub.LockedView
                        ( xLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).AdjointMultiplyVectorPrecompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocal );
                    }
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).AdjointMultiplyVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocal );
        }
        break;
    }
    case NODE:
    {
        const Node& node = *block_.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).AdjointMultiplyVectorPrecompute
                ( context, alpha, xLocalSub, yLocalSub );
            }
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // Form z := alpha ULocal^H xLocal
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        blas::Gemv
        ( 'C', DF.ULocal.Height(), DF.rank, 
          alpha,     DF.ULocal.LockedBuffer(), DF.ULocal.LDim(), 
                     xLocal.LockedBuffer(),    1,
          Scalar(0), z.Buffer(),               1 );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::AdjointMultiply( alpha, SF.D, xLocal, z );
        break;
    }
    case LOW_RANK:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const LowRank<Scalar>& F = *block_.data.F;
        hmat_tools::AdjointMultiply( alpha, F, xLocal, Scalar(1), yLocal );
        break;
    }
    case SPLIT_DENSE:
        break;
    case DENSE:
    {
        // There is no communication required for this piece, so simply perform
        // the entire update.
        const Dense<Scalar>& D = *block_.data.D;
        hmat_tools::AdjointMultiply( alpha, D, xLocal, Scalar(1), yLocal );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces, 0 );
    MultiplyVectorSumsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    MultiplyVectorSumsPack( context, buffer, offsetsCopy );

    // Perform the reduces with log2(p) messages
    teams_->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of teamunicators have data)
    MultiplyVectorSumsUnpack( context, buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorSums");
#endif
    // Compute the message sizes for each reduce 
    const int numLevels = teams_->NumLevels();
    const int numReduces = numLevels-1;
    std::vector<int> sizes( numReduces, 0 );
    TransposeMultiplyVectorSumsCount( sizes );

    // Pack all of the data to be reduced into a single buffer
    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numReduces );
    for( int i=0,offset=0; i<numReduces; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    TransposeMultiplyVectorSumsPack( context, buffer, offsetsCopy );

    // Perform the reduces with log2(p) messages
    teams_->TreeSumToRoots( buffer, sizes );

    // Unpack the reduced buffers (only roots of teamunicators have data)
    TransposeMultiplyVectorSumsUnpack( context, buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorSums
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorSums");
#endif
    // This unconjugated version is identical
    TransposeMultiplyVectorSums( context );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorSumsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorSumsCount");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        sizes[level_] += block_.data.DF->rank;
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorSumsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorSumsCount");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        sizes[level_] += block_.data.DF->rank;
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorSumsPack
( const MultiplyVectorContext& context, 
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorSumsPack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        const Vector<Scalar>& z = *context.block.data.z;
        MemCopy( &buffer[offsets[level_]], z.LockedBuffer(), DF.rank );
        offsets[level_] += DF.rank;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorSumsPack
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorSumsPack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        const Vector<Scalar>& z = *context.block.data.z;
        MemCopy( &buffer[offsets[level_]], z.LockedBuffer(), DF.rank );
        offsets[level_] += DF.rank;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorSumsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorSumsUnpack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            MemCopy( z.Buffer(), &buffer[offsets[level_]], DF.rank );
            offsets[level_] += DF.rank;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorSumsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorSumsUnpack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorSumsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            MemCopy( z.Buffer(), &buffer[offsets[level_]], DF.rank );
            offsets[level_] += DF.rank;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPassData
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    MultiplyVectorPassDataCount( sendSizes, recvSizes );

    // Fill the offset vectors defined by the sizes
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    std::vector<Scalar> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    MultiplyVectorPassDataPack( context, sendBuffer, offsets );

    // Start the non-blocking recvs
    mpi::Comm comm = teams_->Team( 0 );
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
    int offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    mpi::Barrier( comm );

    // Start the non-blocking sends
    const int numSends = sendSizes.size();
    std::vector<mpi::Request> sendRequests( numSends );
    offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    mpi::WaitAll( numRecvs, &recvRequests[0] );
    MultiplyVectorPassDataUnpack( context, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPassDataCount");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataCount
                ( sendSizes, recvSizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( inSourceTeam_ )
                AddToMap( sendSizes, targetRoot_, DF.rank );
            else
                AddToMap( recvSizes, sourceRoot_, DF.rank );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( inSourceTeam_ )
            AddToMap( sendSizes, targetRoot_, SF.rank );
        else
            AddToMap( recvSizes, sourceRoot_, SF.rank );
        break;
    }
    case SPLIT_DENSE:
    {
        if( inSourceTeam_ )
            AddToMap( sendSizes, targetRoot_, Height() );
        else
            AddToMap( recvSizes, sourceRoot_, Height() );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPassDataPack
( MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPassDataPack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        if( DF.rank != 0 )
        {
            mpi::Comm team = teams_->Team( level_ );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                MemCopy
                ( &buffer[offsets[targetRoot_]], z.LockedBuffer(), DF.rank );
                offsets[targetRoot_] += DF.rank;
                z.Clear();
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( SF.rank != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            MemCopy( &buffer[offsets[targetRoot_]], z.LockedBuffer(), SF.rank );
            offsets[targetRoot_] += SF.rank;
            z.Clear();
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        if( height != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            MemCopy( &buffer[offsets[targetRoot_]], z.LockedBuffer(), height );
            offsets[targetRoot_] += height;
            z.Clear();
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPassDataUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPassDataUnpack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        if( DF.rank != 0 )
        {
            mpi::Comm team = teams_->Team( level_ );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( DF.rank );
                MemCopy( z.Buffer(), &buffer[offsets[sourceRoot_]], DF.rank );
                offsets[sourceRoot_] += DF.rank;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( SF.rank != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( SF.rank );
            MemCopy( z.Buffer(), &buffer[offsets[sourceRoot_]], SF.rank );
            offsets[sourceRoot_] += SF.rank;
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        if( height != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( height );
            MemCopy( z.Buffer(), &buffer[offsets[sourceRoot_]], height );
            offsets[sourceRoot_] += height;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPassData
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPassData");
#endif
    // Constuct maps of the send/recv processes to the send/recv sizes
    std::map<int,int> sendSizes, recvSizes;
    TransposeMultiplyVectorPassDataCount( sendSizes, recvSizes );

    // Fill the offset vectors defined by the sizes
    int totalSendSize=0, totalRecvSize=0;
    std::map<int,int> sendOffsets, recvOffsets;
    std::map<int,int>::iterator it;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        sendOffsets[it->first] = totalSendSize;
        totalSendSize += it->second;
    }
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        recvOffsets[it->first] = totalRecvSize;
        totalRecvSize += it->second;
    }

    // Fill the send buffer
    std::vector<Scalar> sendBuffer( totalSendSize );
    std::map<int,int> offsets = sendOffsets;
    TransposeMultiplyVectorPassDataPack( context, xLocal, sendBuffer, offsets );

    // Start the non-blocking sends
    mpi::Comm comm = teams_->Team( 0 );
    const int numSends = sendSizes.size();
    std::vector<mpi::Request> sendRequests( numSends );
    int offset = 0;
    for( it=sendSizes.begin(); it!=sendSizes.end(); ++it )
    {
        const int dest = it->first;
        mpi::ISend
        ( &sendBuffer[sendOffsets[dest]], sendSizes[dest], dest, 0,
          comm, sendRequests[offset++] );
    }

    // Start the non-blocking recvs
    const int numRecvs = recvSizes.size();
    std::vector<mpi::Request> recvRequests( numRecvs );
    std::vector<Scalar> recvBuffer( totalRecvSize );
      offset = 0;
    for( it=recvSizes.begin(); it!=recvSizes.end(); ++it )
    {
        const int source = it->first;
        mpi::IRecv
        ( &recvBuffer[recvOffsets[source]], recvSizes[source], source, 0,
          comm, recvRequests[offset++] );
    }

    // Unpack as soon as we have received our data
    mpi::WaitAll( numRecvs, &recvRequests[0] );
    TransposeMultiplyVectorPassDataUnpack( context, recvBuffer, recvOffsets );

    // Don't exit until we know that the data was sent
    mpi::WaitAll( numSends, &sendRequests[0] );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPassDataCount
( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPassDataCount");
#endif
    switch( block_.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataCount
                ( sendSizes, recvSizes );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ && inTargetTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            if( inTargetTeam_ )
                AddToMap( sendSizes, sourceRoot_, DF.rank );
            else
                AddToMap( recvSizes, targetRoot_, DF.rank );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( inTargetTeam_ )
            AddToMap( sendSizes, sourceRoot_, SF.rank );
        else
            AddToMap( recvSizes, targetRoot_, SF.rank );
        break;
    }
    case SPLIT_DENSE:
    {
        if( inTargetTeam_ )
            AddToMap( sendSizes, sourceRoot_, Height() );
        else
            AddToMap( recvSizes, targetRoot_, Height() );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPassDataPack
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal,
  std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPassDataPack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), xLocal, buffer, offsets );

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPassDataPack
                    ( nodeContext.Child(t,s), xLocal, buffer, offsets );
        }
        else // teamSize == 2
        {
            Vector<Scalar> xLocalSub;
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                // Take care of the upper half 
                for( int t=0,tOffset=0; t<2;
                     tOffset+=node.targetSizes[t],++t )
                {
                    xLocalSub.LockedView
                    ( xLocal, tOffset, node.targetSizes[t] );
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).TransposeMultiplyVectorPassDataPack
                        ( nodeContext.Child(t,s), xLocalSub, buffer, offsets );
                }
            }
            else // teamRank == 1
            {
                // Take care of the bottom half
                for( int t=2,tOffset=0; t<4;
                     tOffset+=node.targetSizes[t],++t )
                {
                    xLocalSub.LockedView
                    ( xLocal, tOffset, node.targetSizes[t] );
                    for( int s=0; s<4; ++s )
                        node.Child(t,s).TransposeMultiplyVectorPassDataPack
                        ( nodeContext.Child(t,s), xLocalSub, buffer, offsets );
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        Vector<Scalar> xLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataPack
                ( nodeContext.Child(t,s), xLocalSub, buffer, offsets );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inSourceTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        if( DF.rank != 0 )
        {
            mpi::Comm team = teams_->Team( level_ );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                MemCopy
                ( &buffer[offsets[sourceRoot_]], z.LockedBuffer(), DF.rank );
                offsets[sourceRoot_] += DF.rank;
                z.Clear();
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( SF.rank != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            MemCopy
            ( &buffer[offsets[sourceRoot_]], z.LockedBuffer(), SF.rank );
            offsets[sourceRoot_] += SF.rank;
            z.Clear();
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        if( height != 0 )
        {
            MemCopy
            ( &buffer[offsets[sourceRoot_]], xLocal.LockedBuffer(), height );
            offsets[sourceRoot_] += height;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPassDataUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPassDataUnpack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext =
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext =
            *context.block.data.SN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorPassDataUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        if( inTargetTeam_ )
            break;
        const DistLowRank& DF = *block_.data.DF;
        if( DF.rank != 0 )
        {
            mpi::Comm team = teams_->Team( level_ );
            const int teamRank = mpi::CommRank( team );
            if( teamRank == 0 )
            {
                Vector<Scalar>& z = *context.block.data.z;
                z.Resize( DF.rank );
                MemCopy( z.Buffer(), &buffer[offsets[targetRoot_]], DF.rank );
                offsets[targetRoot_] += DF.rank;
            }
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        if( SF.rank != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( SF.rank );
            MemCopy( z.Buffer(), &buffer[offsets[targetRoot_]], SF.rank );
            offsets[targetRoot_] += SF.rank;
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const int height = Height();
        if( height != 0 )
        {
            Vector<Scalar>& z = *context.block.data.z;
            z.Resize( height );
            MemCopy( z.Buffer(), &buffer[offsets[targetRoot_]], height );
            offsets[targetRoot_] += height;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorPassData
( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorPassData");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorPassData( context, xLocal );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts, 0 );
    MultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of teamunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    MultiplyVectorBroadcastsPack( context, buffer, offsetsCopy );

    // Perform the broadcasts with log2(p) messages
    teams_->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers 
    MultiplyVectorBroadcastsUnpack( context, buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorBroadcasts");
#endif
    // Compute the message sizes for each broadcast
    const int numLevels = teams_->NumLevels();
    const int numBroadcasts = numLevels-1;
    std::vector<int> sizes( numBroadcasts, 0 );
    TransposeMultiplyVectorBroadcastsCount( sizes );

    // Pack all of the data to be broadcasted into a single buffer
    // (only roots of teamunicators contribute)
    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];
    std::vector<Scalar> buffer( totalSize );
    std::vector<int> offsets( numBroadcasts );
    for( int i=0,offset=0; i<numBroadcasts; offset+=sizes[i],++i )
        offsets[i] = offset;
    std::vector<int> offsetsCopy = offsets;
    TransposeMultiplyVectorBroadcastsPack( context, buffer, offsetsCopy );

    // Perform the broadcasts with log2(p) messages
    teams_->TreeBroadcasts( buffer, sizes );

    // Unpack the broadcasted buffers 
    TransposeMultiplyVectorBroadcastsUnpack( context, buffer, offsets );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorBroadcasts
( MultiplyVectorContext& context ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorBroadcasts");
#endif
    // The unconjugated version should be identical
    TransposeMultiplyVectorBroadcasts( context );
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorBroadcastsCount");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        sizes[level_] += block_.data.DF->rank;
        break;
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorBroadcastsCount
( std::vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorBroadcastsCount");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsCount( sizes );
        break;
    }
    case DIST_LOW_RANK:
        sizes[level_] += block_.data.DF->rank;
        break;

    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorBroadcastsPack
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorBroadcastsPack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        const Vector<Scalar>& z = *context.block.data.z;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            MemCopy( &buffer[offsets[level_]], z.LockedBuffer(), DF.rank );
            offsets[level_] += DF.rank;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorBroadcastsPack
( const MultiplyVectorContext& context,
  std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorBroadcastsPack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        const typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsPack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        const Vector<Scalar>& z = *context.block.data.z;
        mpi::Comm team = teams_->Team( level_ );
        const int teamRank = mpi::CommRank( team );
        if( teamRank == 0 )
        {
            MemCopy( &buffer[offsets[level_]], z.LockedBuffer(), DF.rank );
            offsets[level_] += DF.rank;
        }
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorBroadcastsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorBroadcastsPack");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorBroadcastsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        MemCopy( z.Buffer(), &buffer[offsets[level_]], DF.rank );
        offsets[level_] += DF.rank;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorBroadcastsUnpack
( MultiplyVectorContext& context,
  const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorBroadcastsPack");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMultiplyVectorBroadcastsUnpack
                ( nodeContext.Child(t,s), buffer, offsets );
        break;
    }
    case DIST_LOW_RANK:
    {
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        z.Resize( DF.rank );
        MemCopy( z.Buffer(), &buffer[offsets[level_]], DF.rank );
        offsets[level_] += DF.rank;
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::MultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorPostcompute");
#endif
    if( !inTargetTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).MultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inSourceTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block 
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=0,sOffset=0; s<2;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            xLocalSub.LockedView
                            ( xLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the upper-right block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=2; s<4; ++s )
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Bottom-right block
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=2,sOffset=0; s<4;
                             sOffset+=node.sourceSizes[s],++s )
                        {
                            xLocalSub.LockedView
                            ( xLocal, sOffset, node.sourceSizes[s] );
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Bottom-left block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<2; ++s )
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else
            {
                // Only split yLocal
                Vector<Scalar> yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper half
                    for( int t=0,tOffset=0; t<2;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the bottom half
                    for( int t=2,tOffset=0; t<4;
                         tOffset+=node.targetSizes[t],++t )
                    {
                        yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
                        for( int s=0; s<4; ++s )
                            node.Child(t,s).MultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        Vector<Scalar> yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).MultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocalSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // yLocal += ULocal z
        const DistLowRank& DF = *block_.data.DF;
        const Vector<Scalar>& z = *context.block.data.z;
        blas::Gemv
        ( 'N', DF.ULocal.Height(), DF.rank,
          Scalar(1), DF.ULocal.LockedBuffer(), DF.ULocal.LDim(),
                     z.LockedBuffer(),         1,
          Scalar(1), yLocal.Buffer(),          1 );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        const Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::Multiply( Scalar(1), SF.D, z, Scalar(1), yLocal );
        break;
    }
    case SPLIT_DENSE:
    {
        const Vector<Scalar>& z = *context.block.data.z;
        const int localHeight = Height();
        const Scalar* zBuffer = z.LockedBuffer();
        Scalar* yLocalBuffer = yLocal.Buffer();
        for( int i=0; i<localHeight; ++i )
            yLocalBuffer[i] += zBuffer[i];
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::TransposeMultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal, 
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::TransposeMultiplyVectorPostcompute");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;
        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).TransposeMultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inTargetTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block 
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0,tOffset=0; t<2;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the lower left block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2; t<4; ++t )
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Bottom right block
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2,tOffset=0; t<4;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Top right block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<2; ++t )
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else
            {
                // Only split yLocal
                Vector<Scalar> yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the left half
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the right half
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).TransposeMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
            }
        }
        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        Vector<Scalar> yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                node.Child(t,s).TransposeMultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocalSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // yLocal += (VLocal^[T/H])^T z
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        // yLocal += VLocal z
        blas::Gemv
        ( 'N', DF.VLocal.Height(), DF.rank,
          Scalar(1), DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                     z.LockedBuffer(),         1,
          Scalar(1), yLocal.Buffer(),          1 );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::Multiply( Scalar(1), SF.D, z, Scalar(1), yLocal );
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDense& SD = *block_.data.SD;
        const Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::TransposeMultiply( alpha, SD.D, z, Scalar(1), yLocal );
        break;
    }
    default:
        break;
    }
}

template<typename Scalar>
void
DistHMat2d<Scalar>::AdjointMultiplyVectorPostcompute
( MultiplyVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::AdjointMultiplyVectorPostcompute");
#endif
    if( !inSourceTeam_ )
        return;

    switch( block_.type )
    {
    case DIST_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::DistNode& nodeContext = 
            *context.block.data.DN;

        mpi::Comm team = teams_->Team( level_ );
        const int teamSize = mpi::CommSize( team );
        if( teamSize > 2 )
        {
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    node.Child(t,s).AdjointMultiplyVectorPostcompute
                    ( nodeContext.Child(t,s), alpha, xLocal, yLocal );
        }
        else // teamSize == 2
        {
            const int teamRank = mpi::CommRank( team );
            if( inTargetTeam_ )
            {
                // Split xLocal and yLocal
                Vector<Scalar> xLocalSub, yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the upper left block 
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0,tOffset=0; t<2;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Take care of the lower left block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2; t<4; ++t )
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Bottom right block
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=2,tOffset=0; t<4;
                             tOffset+=node.targetSizes[t],++t )
                        {
                            xLocalSub.LockedView
                            ( xLocal, tOffset, node.targetSizes[t] );
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                        }
                    }
                    // Top right block
                    xLocalSub.LockedView( xLocal, 0, 0 );
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<2; ++t )
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocalSub, yLocalSub );
                    }
                }
            }
            else
            {
                // Only split yLocal
                Vector<Scalar> yLocalSub;
                if( teamRank == 0 )
                {
                    // Take care of the left half
                    for( int s=0,sOffset=0; s<2;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
                else // teamRank == 1
                {
                    // Take care of the right half
                    for( int s=2,sOffset=0; s<4;
                         sOffset+=node.sourceSizes[s],++s )
                    {
                        yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                        for( int t=0; t<4; ++t )
                            node.Child(t,s).AdjointMultiplyVectorPostcompute
                            ( nodeContext.Child(t,s),
                              alpha, xLocal, yLocalSub );
                    }
                }
            }
        }

        break;
    }
    case SPLIT_NODE:
    {
        const Node& node = *block_.data.N;
        typename MultiplyVectorContext::SplitNode& nodeContext = 
            *context.block.data.SN;
        Vector<Scalar> yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
                node.Child(t,s).AdjointMultiplyVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocal, yLocalSub );
        }
        break;
    }
    case DIST_LOW_RANK:
    {
        // yLocal += (VLocal^[T/H])^H z
        const DistLowRank& DF = *block_.data.DF;
        Vector<Scalar>& z = *context.block.data.z;
        // yLocal += conj(VLocal) z
        hmat_tools::Conjugate( z );
        hmat_tools::Conjugate( yLocal );
        blas::Gemv
        ( 'N', DF.VLocal.Height(), DF.rank,
          Scalar(1), DF.VLocal.LockedBuffer(), DF.VLocal.LDim(),
                     z.LockedBuffer(),         1,
          Scalar(1), yLocal.Buffer(),          1 );
        hmat_tools::Conjugate( yLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        const SplitLowRank& SF = *block_.data.SF;
        Vector<Scalar>& z = *context.block.data.z;
        // yLocal += conj(V) z
        hmat_tools::Conjugate( z );
        hmat_tools::Conjugate( yLocal );
        hmat_tools::Multiply( Scalar(1), SF.D, z, Scalar(1), yLocal );
        hmat_tools::Conjugate( yLocal );
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDense& SD = *block_.data.SD;
        const Vector<Scalar>& z = *context.block.data.z;
        hmat_tools::AdjointMultiply( alpha, SD.D, z, Scalar(1), yLocal );
        break;
    }
    default:
        break;
    }
}

} // namespace dmhm
