/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

#ifdef HAVE_QT5

namespace dmhm {

ComplexDisplayWindow::ComplexDisplayWindow( QWidget* parent )
: QWidget(parent)
{
#ifndef RELEASE
    CallStackEntry entry("ComplexDisplayWindow::DisplayWindow");
#endif
    matrix_ = 0;
 
    // For the real matrix
    QHBoxLayout* matrixLayout = new QHBoxLayout(); 
    realDisplay_ = new DisplayWidget<std::complex<double> >();
    realScroll_ = new QScrollArea();
    realScroll_->setWidget( realDisplay_ );
    matrixLayout->addWidget( realScroll_ );

    // For the imaginary matrix
    imagDisplay_ = new DisplayWidget<std::complex<double> >(); 
    imagScroll_ = new QScrollArea();
    imagScroll_->setWidget( imagDisplay_ );
    matrixLayout->addWidget( imagScroll_ );

    setLayout( matrixLayout );
    setAttribute( Qt::WA_DeleteOnClose );

    // DMHM needs to know if a window was opened for cleanup purposes
    OpenedWindow();
}

ComplexDisplayWindow::~ComplexDisplayWindow()
{ delete matrix_; }

void 
ComplexDisplayWindow::Display
( const Dense<std::complex<double> >* matrix, QString title )
{
#ifndef RELEASE
    CallStackEntry entry("ComplexDisplayWindow::Display");
#endif
    if( matrix_ != 0 )
        delete matrix_;
    matrix_ = matrix;

    setWindowTitle( title );
    realDisplay_->DisplayReal( matrix );
    imagDisplay_->DisplayImag( matrix );
}

} // namespace dmhm

#endif // ifndef HAVE_QT5
