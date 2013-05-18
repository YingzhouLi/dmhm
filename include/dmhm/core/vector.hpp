/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_VECTOR_HPP
#define DMHM_VECTOR_HPP 1

#include <cstring>
#include <stdexcept>
#include <vector>

namespace dmhm {

// A vector implementation that allows O(1) creation of subvectors. 
// The tradeoff versus std::vector is that introducing (locked) views makes 
// operator[] usage impractical, so we instead require Set() and Get().
template<typename Scalar>
class Vector
{
    int height_;
    bool viewing_;
    bool lockedView_;
    std::vector<Scalar> memory_;
    Scalar* buffer_;
    const Scalar* lockedBuffer_;

public:
    Vector();
    Vector( int height );
    Vector( int height, Scalar* buffer );
    Vector( int height, const Scalar* lockedBuffer );
    Vector( const Vector<Scalar>& x );
    ~Vector();

    int Height() const;
    void Resize( int height );
    void Clear();

    void Set( int i, Scalar value );
    Scalar Get( int i ) const;
    void Print( const std::string tag ) const;

    Scalar* Buffer( int i=0 );
    const Scalar* LockedBuffer( int i=0 ) const;

    void View( Vector<Scalar>& x );
    void View( Vector<Scalar>& x, int i, int height );

    void LockedView( const Vector<Scalar>& x );
    void LockedView( const Vector<Scalar>& x, int i, int height );
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline
Vector<Scalar>::Vector()
: height_(0), viewing_(false), lockedView_(false),
  memory_(), buffer_(0), lockedBuffer_(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height )
: height_(height), viewing_(false), lockedView_(false),
  memory_(height), buffer_(&memory_[0]), lockedBuffer_(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height, Scalar* buffer )
: height_(height), viewing_(true), lockedView_(false),
  memory_(), buffer_(buffer), lockedBuffer_(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height, const Scalar* lockedBuffer )
: height_(height), viewing_(true), lockedView_(true),
  memory_(), buffer_(0), lockedBuffer_(lockedBuffer)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( const Vector<Scalar>& x )
: height_(x.Height()), viewing_(false), lockedView_(false),
  memory_(x.Height()), buffer_(&memory_[0]), lockedBuffer_(0)
{ std::memcpy( buffer_, x.LockedBuffer(), x.Height()*sizeof(Scalar) ); }

template<typename Scalar>
inline
Vector<Scalar>::~Vector()
{ }

template<typename Scalar>
inline int
Vector<Scalar>::Height() const
{ return height_; }

template<typename Scalar>
inline void
Vector<Scalar>::Resize( int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Resize");
    if( viewing_ || lockedView_ )
        throw std::logic_error("Cannot resize a Vector that is a view.");
#endif
    height_ = height;
    memory_.resize( height );
    buffer_ = &memory_[0];
}

template<typename Scalar>
inline void
Vector<Scalar>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Clear");
#endif
    height_ = 0;
    viewing_ = false;
    lockedView_ = false;
    memory_.clear();
    buffer_ = 0;
    lockedBuffer_ = 0;
}

template<typename Scalar>
inline void
Vector<Scalar>::Set( int i, Scalar value )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Set");
    if( lockedView_ )
        throw std::logic_error("Cannot modify locked views");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= height_ )
        throw std::logic_error("Vector::Set is out of bounds");
#endif
    buffer_[i] = value;
}

template<typename Scalar>
inline Scalar
Vector<Scalar>::Get( int i ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Get");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= height_ )
        throw std::logic_error("Vector::Get is out of bounds");
#endif
    if( lockedView_ )
        return lockedBuffer_[i];
    else
        return buffer_[i];
}

template<typename Scalar>
inline void
Vector<Scalar>::Print( const std::string tag ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Print");
#endif
    std::cout << tag << "\n";
    if( lockedView_ )
    {
        for( int i=0; i<height_; ++i )
            std::cout << WrapScalar(lockedBuffer_[i]) << "\n";
    }
    else
    {
        for( int i=0; i<height_; ++i )
            std::cout << WrapScalar(buffer_[i]) << "\n";
    }
    std::cout << std::endl;
}

template<typename Scalar>
inline Scalar*
Vector<Scalar>::Buffer( int i )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Buffer");
    if( lockedView_ )
        throw std::logic_error("Cannot get modifiable buffer from locked view");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > height_ )
        throw std::logic_error("Out of bounds of buffer");
#endif
    return &buffer_[i];
}

template<typename Scalar>
inline const Scalar*
Vector<Scalar>::LockedBuffer( int i ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedBuffer");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > height_ )
        throw std::logic_error("Out of bounds of buffer");
#endif
    if( lockedView_ )
        return &lockedBuffer_[i];
    else
        return &buffer_[i];
}

template<typename Scalar>
inline void
Vector<Scalar>::View( Vector<Scalar>& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::View");
#endif
    viewing_ = true;
    lockedView_ = false;
    buffer_ = x.Buffer();
    height_ = x.Height();
}

template<typename Scalar>
inline void
Vector<Scalar>::View( Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::View");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    viewing_ = true;
    lockedView_ = false;
    buffer_ = x.Buffer( i );
    height_ = height;
}

template<typename Scalar>
inline void
Vector<Scalar>::LockedView( const Vector<Scalar>& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedView");
#endif
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = x.Buffer();
    height_ = x.Height();
}

template<typename Scalar>
inline void
Vector<Scalar>::LockedView( const Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedView");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = x.LockedBuffer( i );
    height_ = height;
}

} // namespace dmhm

#endif // ifndef DMHM_VECTOR_HPP
