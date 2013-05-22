/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_GRAPHICS_COLORMAP_HPP
#define DMHM_GRAPHICS_COLORMAP_HPP 1

#ifdef HAVE_QT5

#include <QWidget>

namespace dmhm {

inline QRgb
ColorMap( double value, double minVal, double maxVal )
{
#ifndef RELEASE
    CallStackEntry entry("ColorMap");
#endif
    const double percent = (value-minVal) / (maxVal-minVal);

    // Grey-scale
    /*
    const int red = 255*percent;
    const int green = 255*percent;
    const int blue = 255*percent;
    const int alpha = 255;
    */

    // 0: Red, 0.5: Black, 1: Green
    const int red = ( percent<=0.5 ? 255*(1.-2*percent) : 0 );
    const int green = ( percent>=0.5 ? 255*(2*(percent-0.5)) : 0 );
    const int blue = 0;
    const int alpha = 255;

    // Red and blue mixture
    /*
    const int red = 255*percent;
    const int green = 0;
    const int blue = 255*(R(1)-percent/2);
    const int alpha = 255;
    */

    return qRgba( red, green, blue, alpha );
}

} // namespace dmhm

#endif // ifdef HAVE_QT5

#endif // ifndef DMHM_GRAPHICS_COLORMAP_HPP
