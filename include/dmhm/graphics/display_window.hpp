/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_GRAPHICS_DISPLAYWINDOW_HPP
#define DMHM_GRAPHICS_DISPLAYWINDOW_HPP

#include <QBoxLayout>
#include <QScrollArea>
#include <QWidget>

#include "dmhm/hmat_tools.hpp"
#include "dmhm/graphics/display_widget.hpp"

namespace dmhm {

class DisplayWindow : public QWidget
{
     Q_OBJECT
public:
    DisplayWindow( QWidget* parent=0 );
    ~DisplayWindow();

    void Display
    ( const Dense<double>* A, QString title=QString("Default title") );

private:
    QScrollArea *scroll_;
    DisplayWidget<double> *display_;
    const Dense<double> *matrix_;

public slots:
    void Save();
};

} // namespace dmhm

#endif // ifndef DMHM_GRAPHICS_DISPLAYWINDOW_HPP
