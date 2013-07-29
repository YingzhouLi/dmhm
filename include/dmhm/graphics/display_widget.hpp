/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_GRAPHICS_DISPLAYWIDGET_HPP
#define DMHM_GRAPHICS_DISPLAYWIDGET_HPP 1

#ifdef HAVE_QT5

#include <QPainter>
#include <QPixmap>
#include <QStylePainter>
#include <QWidget>

#include "dmhm/hmat_tools.hpp"

namespace dmhm {

template<typename T>
class DisplayWidget : public QWidget
{
public:
    DisplayWidget( QWidget* parent=0 );
    ~DisplayWidget();

    void DisplayReal( const Dense<T>* A );
    void DisplayImag( const Dense<T>* A );

protected:
    void paintEvent( QPaintEvent* event );

private:
    QPixmap pixmap_;
};

// Implementation

template<typename R>
inline R
RealPart( R alpha )
{ return alpha; }

template<typename R>
inline R
RealPart( std::complex<R> alpha )
{ return std::real(alpha); }

template<typename R>
inline R
ImagPart( R alpha )
{ return 0; }

template<typename R>
inline R
ImagPart( std::complex<R> alpha )
{ return std::imag(alpha); }

template<typename T>
inline
DisplayWidget<T>::DisplayWidget( QWidget* parent )
: QWidget(parent)
{ }

template<typename T>
inline
DisplayWidget<T>::~DisplayWidget()
{ }

template<typename T>
inline void
DisplayWidget<T>::paintEvent( QPaintEvent* event )
{
#ifndef RELEASE
    CallStackEntry entry("DisplayWidget::paintEvent");
#endif
    QStylePainter painter( this );
    painter.drawPixmap( 0, 0, pixmap_ );
}

template<typename T>
inline void
DisplayWidget<T>::DisplayReal( const Dense<T>* A )
{
#ifndef RELEASE
    CallStackEntry entry("DisplayWidget::DisplayReal");
#endif
    typedef BASE(T) R;
    const int m = A->Height();
    const int n = A->Width();

    // Compute the range of the real values in A
    R minVal=0, maxVal=0;
    if( m != 0 && n != 0 )
    {
        minVal = maxVal = RealPart(A->Get( 0, 0 ));
        for( int j=0; j<n; ++j )
        {
            for( int i=0; i<m; ++i )
            {
                minVal = std::min( minVal, RealPart(A->Get(i,j)) );
                maxVal = std::max( maxVal, RealPart(A->Get(i,j)) );
            }
        }
    }

    // TODO: Parameterize these instead
    const int mPix = std::max( 500, 2*m );
    const int nPix = std::max( 500, 2*n );
    const double mRatio = double(m) / double(mPix);
    const double nRatio = double(n) / double(nPix);
    pixmap_ = QPixmap( nPix, mPix );
    resize( nPix, mPix );

    // Paint the matrix
    QPainter painter( &pixmap_ );
    painter.initFrom( this );
    for( int jPix=0; jPix<nPix; ++jPix )
    {
        const int j = nRatio*jPix;
        for( int iPix=0; iPix<mPix; ++iPix )
        {
            const int i = mRatio*iPix;
            QRgb color = ColorMap( RealPart(A->Get(i,j)), minVal, maxVal );
            painter.setPen( color );
            painter.drawPoint( jPix, iPix );
        }
    }

    // Refresh the widget
    update();
}

template<typename T>
inline void
DisplayWidget<T>::DisplayImag( const Dense<T>* A )
{
#ifndef RELEASE
    CallStackEntry entry("DisplayWidget::DisplayImag");
#endif
    typedef BASE(T) R;
    const int m = A->Height();
    const int n = A->Width();

    // Compute the range of the real values in A
    R minVal=0, maxVal=0;
    if( m != 0 && n != 0 )
    {
        minVal = maxVal = ImagPart(A->Get( 0, 0 ));
        for( int j=0; j<n; ++j )
        {
            for( int i=0; i<m; ++i )
            {
                minVal = std::min( minVal, ImagPart(A->Get(i,j)) );
                maxVal = std::max( maxVal, ImagPart(A->Get(i,j)) );
            }
        }
    }

    // TODO: Parameterize these instead
    const int mPix = std::max( 500, 2*m );
    const int nPix = std::max( 500, 2*n );
    const double mRatio = double(m) / double(mPix);
    const double nRatio = double(n) / double(nPix);
    pixmap_ = QPixmap( nPix, mPix );
    resize( nPix, mPix );

    // Paint the matrix
    QPainter painter( &pixmap_ );
    painter.initFrom( this );
    for( int jPix=0; jPix<nPix; ++jPix )
    {
        const int j = nRatio*jPix;
        for( int iPix=0; iPix<mPix; ++iPix )
        {
            const int i = mRatio*iPix;
            QRgb color = ColorMap( ImagPart(A->Get(i,j)), minVal, maxVal );
            painter.setPen( color );
            painter.drawPoint( jPix, iPix );
        }
    }

    // Refresh the widget
    update();
}

} // namespace dmhm

#endif // ifdef HAVE_QT5

#endif // ifndef DMHM_GRAPHICS_DISPLAYWIDGET_HPP
