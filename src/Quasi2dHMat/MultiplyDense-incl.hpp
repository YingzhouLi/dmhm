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
Quasi2dHMat<Scalar>::Multiply
( Scalar alpha, const Dense<Scalar>& B, 
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (D := H D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.targetSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.sourceSizes[s], B.Width() );

                node.Child(t,s).Multiply( alpha, BSub, Scalar(1), CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmat_tools::Multiply( alpha, *_block.data.F, B, Scalar(1), C );
        break;
    case DENSE:
        hmat_tools::Multiply( alpha, *_block.data.D, B, Scalar(1), C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
Quasi2dHMat<Scalar>::Multiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (D := H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Height(), B.Width() );
    Multiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
Quasi2dHMat<Scalar>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (D := H^T D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).TransposeMultiply
                ( alpha, BSub, Scalar(1), CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmat_tools::TransposeMultiply
        ( alpha, *_block.data.F, B, Scalar(1), C );
        break;
    case DENSE:
        hmat_tools::TransposeMultiply
        ( alpha, *_block.data.D, B, Scalar(1), C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
Quasi2dHMat<Scalar>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (D := H^T D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    TransposeMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
Quasi2dHMat<Scalar>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).AdjointMultiply
                ( alpha, BSub, Scalar(1), CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Dense<Scalar> BConj;
        hmat_tools::Conjugate( B, BConj );
        hmat_tools::Conjugate( C );
        UpdateWithNodeSymmetric( alpha, B, C );
        hmat_tools::Conjugate( C );
        break;
    }
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *_block.data.F, B, Scalar(1), C );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *_block.data.D, B, Scalar(1), C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar>
void
Quasi2dHMat<Scalar>::AdjointMultiply
( Scalar alpha, Dense<Scalar>& B,
  Scalar beta,  Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D + D, non-const)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).AdjointMultiply
                ( alpha, BSub, Scalar(1), CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmat_tools::Conjugate( B );
        hmat_tools::Conjugate( C );
        UpdateWithNodeSymmetric( alpha, B, C );
        hmat_tools::Conjugate( B );
        hmat_tools::Conjugate( C );
        break;
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *_block.data.F, B, Scalar(1), C );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *_block.data.D, B, Scalar(1), C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
void
Quasi2dHMat<Scalar>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    AdjointMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar>
void
Quasi2dHMat<Scalar>::AdjointMultiply
( Scalar alpha, Dense<Scalar>& B,
                Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D, non-const)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    AdjointMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm
