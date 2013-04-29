/*
   Distributed-Memory Hierarchical Matrices (DMHM): a prototype implementation
   of distributed-memory H-matrix arithmetic. 

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef DMHM_DENSE_HPP
#define DMHM_DENSE_HPP 1

#include <stdexcept>
#include <string>
#include <vector>

namespace dmhm {

enum MatrixType { GENERAL, SYMMETRIC /*, HERMITIAN*/ };

// A basic dense matrix representation that is used for storing blocks 
// whose sources and targets are too close to represent as low rank
template<typename Scalar>
class Dense
{
private:
    /*
     * Private member data
     */
    int _height, _width;
    int _ldim; // leading dimension of matrix
    bool _viewing;
    bool _lockedView;
    std::vector<Scalar> _memory;
    Scalar* _buffer;
    const Scalar* _lockedBuffer;
    MatrixType _type;

public:
    /*
     * Public non-static member functions
     */
    Dense
    ( MatrixType type=GENERAL );
    Dense
    ( int height, int width, MatrixType type=GENERAL );
    Dense
    ( int height, int width, int ldim, MatrixType type=GENERAL );
    Dense
    ( Scalar* buffer, int height, int width, int ldim, 
      MatrixType type=GENERAL );
    Dense
    ( const Scalar* lockedBuffer, int height, int width, int ldim, 
      MatrixType type=GENERAL );
    ~Dense();

    void SetType( MatrixType type );
    MatrixType Type() const;
    bool General() const;
    bool Symmetric() const;
    /* bool Hermitian() const; */

    int Height() const;
    int Width() const;
    int LDim() const;
    void Resize( int height, int width );
    void Resize( int height, int width, int ldim );
    void EraseCol( int first, int last );
    void EraseRow( int first, int last );
    void Erase( int colfirst, int collast, int rowfirst, int rowlast );
    void Clear();

    void Set( int i, int j, Scalar value );
    Scalar Get( int i, int j ) const;
    void Print( std::ostream& os, const std::string tag ) const;
    void Print( const std::string tag ) const;

    Scalar* Buffer( int i=0, int j=0 );
    const Scalar* LockedBuffer( int i=0, int j=0 ) const;

    void View( Dense<Scalar>& A );
    void View( Dense<Scalar>& A, int i, int j, int height, int width );

    void LockedView( const Dense<Scalar>& A );
    void LockedView
    ( const Dense<Scalar>& A, int i, int j, int height, int width );
};

} // namespace dmhm 

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline
dmhm::Dense<Scalar>::Dense
( MatrixType type )
: _height(0), _width(0), _ldim(1), 
  _viewing(false), _lockedView(false),
  _memory(), _buffer(0), _lockedBuffer(0),
  _type(type)
{ }

template<typename Scalar>
inline
dmhm::Dense<Scalar>::Dense
( int height, int width, MatrixType type )
: _height(height), _width(width), _ldim(std::max(height,1)),
  _viewing(false), _lockedView(false),
  _memory(_ldim*_width), _buffer(&_memory[0]), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    PushCallStack("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    PopCallStack();
#endif
}

template<typename Scalar>
inline 
dmhm::Dense<Scalar>::Dense
( int height, int width, int ldim, MatrixType type )
: _height(height), _width(width), _ldim(ldim), 
  _viewing(false), _lockedView(false),
  _memory(_ldim*_width), _buffer(&_memory[0]), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    PushCallStack("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
    PopCallStack();
#endif
}

template<typename Scalar>
inline 
dmhm::Dense<Scalar>::Dense
( Scalar* buffer, int height, int width, int ldim, MatrixType type )
: _height(height), _width(width), _ldim(ldim), 
  _viewing(true), _lockedView(false),
  _memory(), _buffer(buffer), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    PushCallStack("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
    PopCallStack();
#endif
}

template<typename Scalar>
inline 
dmhm::Dense<Scalar>::Dense
( const Scalar* lockedBuffer, int height, int width, int ldim, MatrixType type )
: _height(height), _width(width), _ldim(ldim),
  _viewing(true), _lockedView(true),
  _memory(), _buffer(0), _lockedBuffer(lockedBuffer),
  _type(type)
{
#ifndef RELEASE
    PushCallStack("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
    PopCallStack();
#endif
}

template<typename Scalar>
inline 
dmhm::Dense<Scalar>::~Dense()
{ }

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::SetType( MatrixType type )
{
#ifndef RELEASE
    PushCallStack("Dense::SetType");
    if( type == SYMMETRIC && _height != _width )
        throw std::logic_error("Symmetric matrices must be square");
    PopCallStack();
#endif
    _type = type;
}

template<typename Scalar>
inline dmhm::MatrixType
dmhm::Dense<Scalar>::Type() const
{
    return _type;
}

template<typename Scalar>
inline bool
dmhm::Dense<Scalar>::General() const
{
    return _type == GENERAL;
}

template<typename Scalar>
inline bool
dmhm::Dense<Scalar>::Symmetric() const
{
    return _type == SYMMETRIC;
}

/*
template<typename Scalar>
inline bool
dmhm::Dense<Scalar>::Hermitian() const
{
    return _type == HERMITIAN;
}
*/

template<typename Scalar>
inline int
dmhm::Dense<Scalar>::Height() const
{
    return _height;
}

template<typename Scalar>
inline int
dmhm::Dense<Scalar>::Width() const
{
    return _width;
}

template<typename Scalar>
inline int
dmhm::Dense<Scalar>::LDim() const
{
    return _ldim;
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Resize( int height, int width )
{
#ifndef RELEASE
    PushCallStack("Dense::Resize");
    if( _viewing )
        throw std::logic_error("Cannot resize views");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( _type == SYMMETRIC && height != width )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    if( height > _ldim )
    {
        // We cannot trivially preserve the old contents
        _ldim = std::max( height, 1 );
    }
    _height = height;
    _width = width;
    _memory.resize( _ldim*width );
    _buffer = &_memory[0];
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Resize( int height, int width, int ldim )
{
#ifndef RELEASE
    PushCallStack("Dense::Resize");
    if( _viewing )
        throw std::logic_error("Cannot resize views");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( ldim < height || ldim < 0 )
        throw std::logic_error("LDim must be positive and >= the height");
    if( _type == SYMMETRIC && height != width )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    _height = height;
    _width = width;
    _ldim = ldim;
    _memory.resize( ldim*width );
    _buffer = &_memory[0];
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::EraseCol( int first, int last )
{
#ifndef RELEASE
    PushCallStack("Dense::EraseCol");
    if( _viewing )
        throw std::logic_error("Cannot erase views");
    if( first < 0 || last > _width+1 )
        throw std::logic_error("First and last must be in the range of matrix");
    if( _type == SYMMETRIC )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    if( first <= last )
    {
        _width = _width-last+first-1;                                                 
        _memory.erase( _memory.begin()+first*_ldim, _memory.begin()+(last+1)*_ldim );
        _buffer = &_memory[0];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::EraseRow( int first, int last )
{
#ifndef RELEASE
    PushCallStack("Dense::EraseRow");
    if( _viewing )
        throw std::logic_error("Cannot erase views");
    if( first < 0 || last > _height+1 )
        throw std::logic_error("First and last must be in the range of matrix");
    if( _type == SYMMETRIC )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    if(first <= last)
    {
        _height = _height-last+first-1;                                                     
        for( int i=_width-1; i>=0; --i)
            _memory.erase
            ( _memory.begin()+i*_ldim+first, _memory.begin()+i*_ldim+last+1 );
        _buffer = &_memory[0];
        _ldim = _ldim-last+first-1;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Erase( int colfirst, int collast, int rowfirst, int rowlast )
{
    MatrixType typetmp = _type;
#ifndef RELEASE
    PushCallStack("Dense::Erase");
    if( _viewing )
        throw std::logic_error("Cannot erase views");
    if( rowfirst < 0 || rowlast > _height+1 || colfirst<0 || collast > _width+1 )
        throw std::logic_error("First and last must be in the range of matrix");
    if( _type == SYMMETRIC && ( colfirst != rowfirst || collast != rowlast ) )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
    if( _type == SYMMETRIC )
        _type = GENERAL;
#endif
        
    EraseCol( colfirst, collast );
    EraseRow( rowfirst, rowlast );
#ifndef RELEASE
    _type = typetmp;
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Clear()
{
#ifndef RELEASE
    PushCallStack("Dense::Clear");
#endif
    _height = 0;
    _width = 0;
    _ldim = 1;
    _viewing = false;
    _lockedView = false;
    _memory.clear();
    _buffer = 0;
    _lockedBuffer = 0;
    _type = GENERAL;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Set( int i, int j, Scalar value )
{
#ifndef RELEASE
    PushCallStack("Dense::Set");
    if( _lockedView )
        throw std::logic_error("Cannot change data in a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= _height || j >= _width )
        throw std::logic_error("Indices are out of bound");
    if( _type == SYMMETRIC && j > i )
        throw std::logic_error("Setting upper entry from symmetric matrix");
    PopCallStack();
#endif
    _buffer[i+j*_ldim] = value;
}

template<typename Scalar>
inline Scalar
dmhm::Dense<Scalar>::Get( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("Dense::Get");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= _height || j >= _width )
        throw std::logic_error("Indices are out of bound");
    if( _type == SYMMETRIC && j > i )
        throw std::logic_error("Retrieving upper entry from symmetric matrix");
    PopCallStack();
#endif
    if( _lockedView )
        return _lockedBuffer[i+j*_ldim];
    else
        return _buffer[i+j*_ldim];
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Print( std::ostream& os, const std::string tag ) const
{
#ifndef RELEASE
    PushCallStack("Dense::Print");
#endif
    os << tag << "\n";
    if( _type == SYMMETRIC )
    {
        for( int i=0; i<_height; ++i )
        {
            for( int j=0; j<=i; ++j )
                os << WrapScalar(Get(i,j)) << " ";
            for( int j=i+1; j<_width; ++j )
                os << WrapScalar(Get(j,i)) << " ";
            os << "\n";
        }
    }
    else
    {
        for( int i=0; i<_height; ++i )
        {
            for( int j=0; j<_width; ++j )
                os << WrapScalar(Get(i,j)) << " ";
            os << "\n";
        }
    }
    os.flush();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::Print( const std::string tag ) const
{
    Print( std::cout, tag );
}

template<typename Scalar>
inline Scalar*
dmhm::Dense<Scalar>::Buffer( int i, int j )
{
#ifndef RELEASE
    PushCallStack("Dense::Buffer");
    if( _lockedView )
        throw std::logic_error("Cannot modify the buffer from a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > _height || j > _width )
        throw std::logic_error("Indices are out of bound");
    PopCallStack();
#endif
    return &_buffer[i+j*_ldim];
}

template<typename Scalar>
inline const Scalar*
dmhm::Dense<Scalar>::LockedBuffer( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("Dense::LockedBuffer");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > _height || j > _width )
        throw std::logic_error("Indices are out of bound");
    PopCallStack();
#endif
    if( _lockedView )
        return &_lockedBuffer[i+j*_ldim];
    else
        return &_buffer[i+j*_ldim];
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::View( Dense<Scalar>& A )
{
#ifndef RELEASE
    PushCallStack("Dense::View");
#endif
    _height = A.Height();
    _width = A.Width();
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = false;
    _buffer = A.Buffer();
    _type = A.Type();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::View
( Dense<Scalar>& A, int i, int j, int height, int width )
{
#ifndef RELEASE
    PushCallStack("Dense::View");
    if( A.Type() == SYMMETRIC && (i != j || height != width) )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+height > A.Height() || j+width > A.Width() )
    {
        std::ostringstream s;
        s << "Submatrix out of bounds: attempted to grab ["
          << i << ":" << i+height-1 << "," << j << ":" << j+width-1 
          << "] from " << A.Height() << " x " << A.Width() << " matrix.";
        throw std::logic_error( s.str().c_str() );
    }
#endif
    _height = height;
    _width = width;
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = false;
    _buffer = A.Buffer(i,j);
    _type = A.Type();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::LockedView( const Dense<Scalar>& A )
{
#ifndef RELEASE
    PushCallStack("Dense::LockedView");
#endif
    _height = A.Height();
    _width = A.Width();
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = A.LockedBuffer();
    _type = A.Type();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
dmhm::Dense<Scalar>::LockedView
( const Dense<Scalar>& A, int i, int j, int height, int width )
{
#ifndef RELEASE
    PushCallStack("Dense::LockedView");
    if( A.Type() == SYMMETRIC && (i != j || height != width) )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+height > A.Height() || j+width > A.Width() )
    {
        std::ostringstream s;
        s << "Submatrix out of bounds: attempted to grab ["
          << i << ":" << i+height << "," << j << ":" << j+width 
          << "] from " << A.Height() << " x " << A.Width() << " matrix.";
        throw std::logic_error( s.str().c_str() );
    }
#endif
    _height = height;
    _width = width;
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = A.LockedBuffer(i,j);
    _type = A.Type();
#ifndef RELEASE
    PopCallStack();
#endif
}

#endif // DMHM_DENSE_HPP
