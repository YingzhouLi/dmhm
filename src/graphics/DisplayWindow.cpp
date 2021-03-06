/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

#ifdef HAVE_QT5

#include <QPushButton>

namespace dmhm {

DisplayWindow::DisplayWindow( QWidget* parent )
: QWidget(parent)
{
#ifndef RELEASE
    CallStackEntry entry("DisplayWindow::DisplayWindow");
#endif
    matrix_ = 0;
    QVBoxLayout* mainLayout = new QVBoxLayout();

    QHBoxLayout* matrixLayout = new QHBoxLayout();
    display_ = new DisplayWidget<double>();
    scroll_ = new QScrollArea();
    scroll_->setWidget( display_ );
    matrixLayout->addWidget( scroll_ );
    mainLayout->addLayout( matrixLayout ); 

    // Add a save button
    QHBoxLayout* optionsLayout = new QHBoxLayout();
    QPushButton* saveButton = new QPushButton("Save");
    optionsLayout->addWidget( saveButton );
    mainLayout->addLayout( optionsLayout );

    setLayout( mainLayout );
    connect( saveButton, SIGNAL(clicked()), this, SLOT(Save()) );
    setAttribute( Qt::WA_DeleteOnClose );

    // DMHM needs to know if a window was opened for cleanup purposes
    OpenedWindow();
}

DisplayWindow::~DisplayWindow()
{ delete matrix_; }

void
DisplayWindow::Display( const Dense<double>* matrix, QString title )
{
#ifndef RELEASE
    CallStackEntry entry("DisplayWindow::Display");
#endif
    if( matrix_ != 0 )
        delete matrix_;
    matrix_ = matrix;

    setWindowTitle( title );
    display_->DisplayReal( matrix );
}

void
DisplayWindow::Save()
{
    display_->SavePng( windowTitle().toStdString() );
}

} // namespace dmhm

#endif // ifndef HAVE_QT5
