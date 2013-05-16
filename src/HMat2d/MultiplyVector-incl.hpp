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
HMat2d<Scalar>::Multiply
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Multiply (y := H x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( block_.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *block_.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.targetSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.sourceSizes[s] );

                node.Child(t,s).Multiply( alpha, xSub, Scalar(1), ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmat_tools::Multiply( alpha, *block_.data.F, x, Scalar(1), y );
        break;
    case DENSE:
        hmat_tools::Multiply( alpha, *block_.data.D, x, Scalar(1), y );
        break;
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::Multiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::Multiply (y := H x)");
#endif
    y.Resize( Height() );
    Multiply( alpha, x, 0, y );
}

template<typename Scalar>
void
HMat2d<Scalar>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::TransposeMultiply (y := H^T x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( block_.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *block_.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).TransposeMultiply
                ( alpha, xSub, Scalar(1), ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmat_tools::TransposeMultiply( alpha, *block_.data.F, x, Scalar(1), y );
        break;
    case DENSE:
        hmat_tools::TransposeMultiply( alpha, *block_.data.D, x, Scalar(1), y );
        break;
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::TransposeMultiply (y := H^T x)");
#endif
    y.Resize( Width() );
    TransposeMultiply( alpha, x, 0, y );
}

template<typename Scalar>
void
HMat2d<Scalar>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::AdjointMultiply (y := H^H x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( block_.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *block_.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).AdjointMultiply
                ( alpha, xSub, Scalar(1), ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Vector<Scalar> xConj;
        hmat_tools::Conjugate( x, xConj );
        hmat_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), xConj, y ); 
        hmat_tools::Conjugate( y );
        break;
    }
    case LOW_RANK:
        hmat_tools::AdjointMultiply
        ( alpha, *block_.data.F, x, Scalar(1), y );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply
        ( alpha, *block_.data.D, x, Scalar(1), y );
        break;
    }
}

// Having a non-const x allows us to conjugate x in place for the 
// NODE_SYMMETRIC updates.
template<typename Scalar>
void
HMat2d<Scalar>::AdjointMultiply
( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry
    ("HMat2d::AdjointMultiply (y := H^H x + y, non-const)");
#endif
    hmat_tools::Scale( beta, y );
    switch( block_.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *block_.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).AdjointMultiply
                ( alpha, xSub, Scalar(1), ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmat_tools::Conjugate( x );
        hmat_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), x, y ); 
        hmat_tools::Conjugate( x );
        hmat_tools::Conjugate( y );
        break;
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *block_.data.F, x, Scalar(1), y );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *block_.data.D, x, Scalar(1), y );
        break;
    }
}

template<typename Scalar>
void
HMat2d<Scalar>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::AdjointMultiply (y := H^H x)");
#endif
    y.Resize( Width() );
    AdjointMultiply( alpha, x, 0, y );
}

// This version allows for temporary in-place conjugation of x
template<typename Scalar>
void
HMat2d<Scalar>::AdjointMultiply
( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    CallStackEntry entry("HMat2d::AdjointMultiply (y := H^H x, non-const)");
#endif
    y.Resize( Width() );
    AdjointMultiply( alpha, x, 0, y );
}

} // namespace dmhm
