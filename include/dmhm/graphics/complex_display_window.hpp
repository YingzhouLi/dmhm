/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_GRAPHICS_COMPLEXDISPLAYWINDOW_HPP
#define DMHM_GRAPHICS_COMPLEXDISPLAYWINDOW_HPP 1

#include <QBoxLayout>
#include <QScrollArea>
#include <QWidget>

#include "dmhm/hmat_tools.hpp"
#include "dmhm/graphics/display_widget.hpp"

namespace dmhm {

class ComplexDisplayWindow : public QWidget
{
    // This isn't needed until we add slots
    // Q_OBJECT
public:
    ComplexDisplayWindow( QWidget* parent=0 );
    ~ComplexDisplayWindow();

    void Display
    ( const Dense<std::complex<double> >* A,
      QString title=QString("Default title") );

private:
    QScrollArea *realScroll_, *imagScroll_;
    DisplayWidget<std::complex<double> > *realDisplay_, *imagDisplay_;
    const Dense<std::complex<double> > *matrix_;
};

} // namespace dmhm

#endif // ifndef DMHM_GRAPHICS_COMPLEXDISPLAYWINDOW_HPP
