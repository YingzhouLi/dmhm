# The serial Intel compilers
set(COMPILER_DIR /opt/intel/compilers_and_libraries_2018.2.199/linux/bin/intel64/)
set(CMAKE_C_COMPILER       ${COMPILER_DIR}/icc)
set(CMAKE_CXX_COMPILER     ${COMPILER_DIR}/icpc)

# The MPI wrappers for the C and C++ compilers
set(MPI_COMPILER_DIR /opt/apps/intel18/impi/18.0.2/bin/)
set(MPI_C_COMPILER       ${MPI_COMPILER_DIR}/mpicc)
set(MPI_CXX_COMPILER     ${MPI_COMPILER_DIR}/mpicxx)

SET(MPI_CXX_COMPILE_FLAGS "-xCORE-AVX2 -axCORE-AVX512,MIC-AVX512 -O3")
SET(CMAKE_CXX_FLAGS "-xCORE-AVX2 -axCORE-AVX512,MIC-AVX512 -O3")

set(MATH_LIBS "-mkl")
