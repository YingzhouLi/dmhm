/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

void Usage()
{
    std::cout << "Pack <xSize> <ySize> <numLevels> "
                 "<strongly admissible?> <r> <print?>" << std::endl;
}

template<typename Real>
void
FormRow
( int x, int y, int xSize, int ySize,
  std::vector<std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y;

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y );
        row.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y );
        row.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) );
        row.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) );
        row.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int rank = dmhm::mpi::CommRank( MPI_COMM_WORLD );

    if( argc < 7 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    int arg=1;
    const int xSize = atoi( argv[arg++] );
    const int ySize = atoi( argv[arg++] );
    const int numLevels = atoi( argv[arg++] );
    const bool stronglyAdmissible = atoi( argv[arg++] );
    const int r = atoi( argv[arg++] );
    const bool print = atoi( argv[arg++] );

    const int m = xSize*ySize;
    const int n = xSize*ySize;

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing complex double HMat2d packing/unpacking\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef dmhm::HMat2d<Scalar> HMat;

        dmhm::Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        std::vector<int> map;
        HMat::BuildNaturalToHierarchicalMap( map, xSize, ySize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( rank == 0 )
        {
            std::cout << "Filling sparse matrix...";
            std::cout.flush();
        }
        double fillStartTime = dmhm::mpi::WallTime();
        std::vector<Scalar> row;
        std::vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;
            const int z = iNatural/(xSize*ySize);

            FormRow( x, y, xSize, ySize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        double fillStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
            if( print )
                S.Print("S");
        }

        // Convert to H-matrix form
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrix...";
            std::cout.flush();
        }
        double constructStartTime = dmhm::mpi::WallTime();
        HMat H( S, numLevels, r, stronglyAdmissible, xSize, ySize );
        double constructStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
        }

        // Test against a vector of all 1's
        dmhm::Vector<Scalar> x;
        x.Resize( m );
        Scalar* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector of all ones...";
            std::cout.flush();
        }
        dmhm::Vector<Scalar> y;
        double matVecStartTime = dmhm::mpi::WallTime();
        H.Multiply( 1.0, x, y );
        double matVecStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime 
                      << " seconds." << std::endl;
            if( print )
                y.Print("y := H x ~= S x");
        }

        // Pack the H-matrix
        std::vector<dmhm::byte> packedHMat;
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix...";
            std::cout.flush();
        }
        double packStartTime = dmhm::mpi::WallTime();
        H.Pack( packedHMat );
        double packStopTime = dmhm::mpi::WallTime();
        double sizeInMB = ((double)packedHMat.size())/(1024.*1024.);
        if( rank == 0 )
        {
            std::cout << "done: " << packStopTime-packStartTime << " seconds.\n"
                      << "Packed size: " << sizeInMB << " MB." << std::endl;
        }

        // Unpack the H-matrix
        if( rank == 0 )
        {
            std::cout << "Unpacking H-matrix...";
            std::cout.flush();
        }
        double unpackStartTime = dmhm::mpi::WallTime();
        HMat HCopy( packedHMat );
        double unpackStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime 
                      << " seconds." << std::endl;
            if( print )
                HCopy.Print("Unpacked copy of H-matrix");
        }

        // Check that the copied H-matrix has the same action on our vector of 
        // all 1's
        dmhm::Vector<Scalar> z;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector...";
            std::cout.flush();
        }
        matVecStartTime = dmhm::mpi::WallTime();
        HCopy.Multiply( 1.0, x, z );
        matVecStopTime = dmhm::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime 
                      << " seconds." << std::endl;
            if( print )
                z.Print("z := HCopy x ~= S x");
            for( int i=0; i<z.Height(); ++i )
            {
                if( z.Get(i) != y.Get(i) )
                {
                    std::ostringstream s;
                    s << "Action of copied H-matrix differed at index " << i;
                    throw std::logic_error( s.str().c_str() );
                }
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        dmhm::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}
