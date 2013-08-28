{\rtf1\ansi\ansicpg1252\cocoartf1187\cocoasubrtf390
{\fonttbl\f0\fmodern\fcharset0 Courier-Oblique;\f1\fmodern\fcharset0 Courier;\f2\fmodern\fcharset0 Courier-Bold;
}
{\colortbl;\red255\green255\blue255;\red135\green136\blue117;\red38\green38\blue38;\red14\green114\blue164;
\red210\green0\blue53;\red14\green110\blue109;}
{\info
{\title Elemental/cmake/toolchains/Stampede-intel-mvapich2-mkl.cmake at master \'b7 elemental/Elemental \'b7 GitHub}
{\doccomm Elemental - Distributed-memory dense linear algebra}}\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl360

\f0\i\fs24 \cf2 # The serial Intel compilers
\f1\i0 \cf3 \
\pard\pardeftab720\sl360
\cf4 set\cf3 (\cf5 COMPILER_DIR\cf3  \cf5 /opt/apps/intel/13/composer_xe_2013.2.146/bin/intel64\cf3 )\
\cf4 set\cf3 (\cf5 CMAKE_C_COMPILER\cf3        
\f2\b $\{
\f1\b0 \cf6 COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /icc\cf3 )\
\cf4 set\cf3 (\cf5 CMAKE_CXX_COMPILER\cf3      
\f2\b $\{
\f1\b0 \cf6 COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /icpc\cf3 )\
\cf4 set\cf3 (\cf5 CMAKE_Fortran_COMPILER\cf3  
\f2\b $\{
\f1\b0 \cf6 COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /ifort\cf3 )\
\
\pard\pardeftab720\sl360

\f0\i \cf2 # The MPI wrappers for the C and C++ compilers
\f1\i0 \cf3 \
\pard\pardeftab720\sl360
\cf4 set\cf3 (\cf5 MPI_COMPILER_DIR\cf3  \cf5 /opt/apps/intel13/mvapich2/1.9/bin\cf3 )\
\cf4 set\cf3 (\cf5 MPI_C_COMPILER\cf3        
\f2\b $\{
\f1\b0 \cf6 MPI_COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /mpicc\cf3 )\
\cf4 set\cf3 (\cf5 MPI_CXX_COMPILER\cf3      
\f2\b $\{
\f1\b0 \cf6 MPI_COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /mpicxx\cf3 )\
\cf4 set\cf3 (\cf5 MPI_Fortran_COMPILER\cf3  
\f2\b $\{
\f1\b0 \cf6 MPI_COMPILER_DIR
\f2\b \cf3 \}
\f1\b0 \cf5 /mpif90\cf3 )\
\
\cf4 set\cf3 (\cf5 MPI_C_COMPILE_FLAGS\cf3  \cf5 ""\cf3 )\
\cf4 set\cf3 (\cf5 MPI_CXX_COMPILE_FLAGS\cf3  \cf5 ""\cf3 )\
\cf4 set\cf3 (\cf5 MPI_Fortran_COMPILE_FLAGS\cf3  \cf5 ""\cf3 )\
\cf4 set\cf3 (\cf5 MPI_C_INCLUDE_PATH\cf3        \cf5 /opt/apps/intel13/mvapich2/1.9/include\cf3 )\
\cf4 set\cf3 (\cf5 MPI_CXX_INCLUDE_PATH\cf3      
\f2\b $\{
\f1\b0 \cf6 MPI_C_INCLUDE_PATH
\f2\b \cf3 \}
\f1\b0 )\
\cf4 set\cf3 (\cf5 MPI_Fortran_INCLUDE_PATH\cf3  
\f2\b $\{
\f1\b0 \cf6 MPI_C_INCLUDE_PATH
\f2\b \cf3 \}
\f1\b0 )\
\cf4 set\cf3 (\cf5 MPI_C_LINK_FLAGS\cf3  \cf5 "-Wl,-rpath,/opt/apps/limic2/0.5.5/lib -L/opt/apps/limic2/0.5.5/lib -L/opt/apps/intel13/mvapich2/1.9/lib -L/opt/ofed/lib64/"\cf3 )\
\cf4 set\cf3 (\cf5 MPI_CXX_LINK_FLAGS\cf3  
\f2\b $\{
\f1\b0 \cf6 MPI_C_LINK_FLAGS
\f2\b \cf3 \}
\f1\b0 )\
\cf4 set\cf3 (\cf5 MPI_Fortran_LINK_FLAGS\cf3  
\f2\b $\{
\f1\b0 \cf6 MPI_C_LINK_FLAGS
\f2\b \cf3 \}
\f1\b0 )\
\cf4 set\cf3 (\cf5 MPI_BASE_LIBS\cf3  \
\'a0\'a0\'a0\'a0\cf5 "-lmpich -lopa -llimic2 -lpthread -lrdmacm -libverbs -libumad -ldl -lrt"\cf3 )\
\cf4 set\cf3 (\cf5 MPI_C_LIBRARIES\cf3  \cf5 "-limf $\{MPI_BASE_LIBS\}"\cf3 )\
\cf4 set\cf3 (\cf5 MPI_CXX_LIBRARIES\cf3  \cf5 "-limf -lmpichcxx $\{MPI_BASE_LIBS\}"\cf3 )\
\cf4 set\cf3 (\cf5 MPI_Fortran_LIBRARIES\cf3  \cf5 "-limf -lmpichf90 $\{MPI_BASE_LIBS\}"\cf3 )\
\
\cf4 if\cf3 (\cf5 CMAKE_BUILD_TYPE\cf3  \cf5 MATCHES\cf3  \cf5 PureDebug\cf3  \cf5 OR\cf3 \
\'a0\'a0\'a0\cf5 CMAKE_BUILD_TYPE\cf3  \cf5 MATCHES\cf3  \cf5 HybridDebug\cf3 )\
\'a0\'a0\cf4 set\cf3 (\cf5 CXX_FLAGS\cf3  \cf5 "-g -std=c++11"\cf3 )\
\cf4 else\cf3 ()\
\'a0\'a0\cf4 set\cf3 (\cf5 CXX_FLAGS\cf3  \cf5 "-O3 -std=c++11"\cf3 )\
\cf4 endif\cf3 ()\
\
\cf4 set\cf3 (\cf5 OpenMP_CXX_FLAGS\cf3  \cf5 "-openmp"\cf3 )\
\
\cf4 set\cf3 (\cf5 MATH_LIBS\cf3  \cf5 "-mkl"\cf3 )}