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

} // namespace dmhm

#endif // ifdef HAVE_QT5

#endif // ifndef DMHM_GRAPHICS_DISPLAYWIDGET_HPP
