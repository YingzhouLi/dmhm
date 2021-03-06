/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_CONFIG_H
#define DMHM_CONFIG_H 1

#define DMHM_VERSION_MAJOR @DMHM_VERSION_MAJOR@
#define DMHM_VERSION_MINOR @DMHM_VERSION_MINOR@
#cmakedefine RELEASE
#cmakedefine TIME_MULTIPLY
#cmakedefine MEMORY_INFO
#cmakedefine BLAS_POST
#cmakedefine LAPACK_POST
#cmakedefine HAVE_QT5
#cmakedefine AVOID_COMPLEX_MPI
#cmakedefine HAVE_MPI_IN_PLACE

#define RESTRICT @RESTRICT@

#endif /* ifndef DMHM_CONFIG_H */
