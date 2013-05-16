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
    int _height;
    bool _viewing;
    bool _lockedView;
    std::vector<Scalar> _memory;
    Scalar* _buffer;
    const Scalar* _lockedBuffer;

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
: _height(0), _viewing(false), _lockedView(false),
  _memory(), _buffer(0), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height )
: _height(height), _viewing(false), _lockedView(false),
  _memory(height), _buffer(&_memory[0]), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height, Scalar* buffer )
: _height(height), _viewing(true), _lockedView(false),
  _memory(), _buffer(buffer), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( int height, const Scalar* lockedBuffer )
: _height(height), _viewing(true), _lockedView(true),
  _memory(), _buffer(0), _lockedBuffer(lockedBuffer)
{ }

template<typename Scalar>
inline
Vector<Scalar>::Vector( const Vector<Scalar>& x )
: _height(x.Height()), _viewing(false), _lockedView(false),
  _memory(x.Height()), _buffer(&_memory[0]), _lockedBuffer(0)
{ std::memcpy( _buffer, x.LockedBuffer(), x.Height()*sizeof(Scalar) ); }

template<typename Scalar>
inline
Vector<Scalar>::~Vector()
{ }

template<typename Scalar>
inline int
Vector<Scalar>::Height() const
{ return _height; }

template<typename Scalar>
inline void
Vector<Scalar>::Resize( int height )
{
#ifndef RELEASE
    PushCallStack("Vector::Resize");
    if( _viewing || _lockedView )
        throw std::logic_error("Cannot resize a Vector that is a view.");
#endif
    _height = height;
    _memory.resize( height );
    _buffer = &_memory[0];
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
Vector<Scalar>::Clear()
{
#ifndef RELEASE
    PushCallStack("Vector::Clear");
#endif
    _height = 0;
    _viewing = false;
    _lockedView = false;
    _memory.clear();
    _buffer = 0;
    _lockedBuffer = 0;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
Vector<Scalar>::Set( int i, Scalar value )
{
#ifndef RELEASE
    PushCallStack("Vector::Set");
    if( _lockedView )
        throw std::logic_error("Cannot modify locked views");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= _height )
        throw std::logic_error("Vector::Set is out of bounds");
    PopCallStack();
#endif
    _buffer[i] = value;
}

template<typename Scalar>
inline Scalar
Vector<Scalar>::Get( int i ) const
{
#ifndef RELEASE
    PushCallStack("Vector::Get");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= _height )
        throw std::logic_error("Vector::Get is out of bounds");
    PopCallStack();
#endif
    if( _lockedView )
        return _lockedBuffer[i];
    else
        return _buffer[i];
}

template<typename Scalar>
inline void
Vector<Scalar>::Print( const std::string tag ) const
{
#ifndef RELEASE
    PushCallStack("Vector::Print");
#endif
    std::cout << tag << "\n";
    if( _lockedView )
    {
        for( int i=0; i<_height; ++i )
            std::cout << WrapScalar(_lockedBuffer[i]) << "\n";
    }
    else
    {
        for( int i=0; i<_height; ++i )
            std::cout << WrapScalar(_buffer[i]) << "\n";
    }
    std::cout << std::endl;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline Scalar*
Vector<Scalar>::Buffer( int i )
{
#ifndef RELEASE
    PushCallStack("Vector::Buffer");
    if( _lockedView )
        throw std::logic_error("Cannot get modifiable buffer from locked view");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > _height )
        throw std::logic_error("Out of bounds of buffer");
    PopCallStack();
#endif
    return &_buffer[i];
}

template<typename Scalar>
inline const Scalar*
Vector<Scalar>::LockedBuffer( int i ) const
{
#ifndef RELEASE
    PushCallStack("Vector::LockedBuffer");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > _height )
        throw std::logic_error("Out of bounds of buffer");
    PopCallStack();
#endif
    if( _lockedView )
        return &_lockedBuffer[i];
    else
        return &_buffer[i];
}

template<typename Scalar>
inline void
Vector<Scalar>::View( Vector<Scalar>& x )
{
#ifndef RELEASE
    PushCallStack("Vector::View");
#endif
    _viewing = true;
    _lockedView = false;
    _buffer = x.Buffer();
    _height = x.Height();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
Vector<Scalar>::View( Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    PushCallStack("Vector::View");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    _viewing = true;
    _lockedView = false;
    _buffer = x.Buffer( i );
    _height = height;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
Vector<Scalar>::LockedView( const Vector<Scalar>& x )
{
#ifndef RELEASE
    PushCallStack("Vector::LockedView");
#endif
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = x.Buffer();
    _height = x.Height();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
Vector<Scalar>::LockedView( const Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    PushCallStack("Vector::LockedView");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = x.LockedBuffer( i );
    _height = height;
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace dmhm

#endif // ifndef DMHM_VECTOR_HPP
