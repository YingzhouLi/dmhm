/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
using namespace dmhm;

template<typename Real>
void
FormRow
( int x, int y, int xSize, int ySize,
  Vector<std::complex<Real> >& row, Vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y;

    row.Resize( 0 );
    colIndices.Resize( 0 );

    // Set up the diagonal entry
    colIndices.Push_back( rowIdx );
    row.Push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.Push_back( (x-1) + xSize*y );
        row.Push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.Push_back( (x+1) + xSize*y );
        row.Push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.Push_back( x + xSize*(y-1) );
        row.Push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.Push_back( x + xSize*(y+1) );
        row.Push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    Initialize( argc, argv );
    const int rank = mpi::CommRank( mpi::COMM_WORLD );
    typedef std::complex<double> Scalar;
    typedef HMat2d<Scalar> HMat;

    try
    {
        const int xSize = Input("--xSize","size of x dimension",20);
        const int ySize = Input("--ySize","size of y dimension",20);
        const int numLevels = Input("--numLevels","depth of H-matrix tree",4);
        const bool strong = Input("--strong","strongly admissible?",false);
        const int maxRank = Input("--maxRank","maximum rank of block",5);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();

        const int m = xSize*ySize;
        const int n = xSize*ySize;

        Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        Vector<int> map;
        HMat::BuildNaturalToHierarchicalMap( map, xSize, ySize, numLevels );

        Vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( rank == 0 )
        {
            std::cout << "Filling sparse matrix...";
            std::cout.flush();
        }
        double fillStartTime = mpi::Time();
        Vector<Scalar> row;
        Vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.Push_back( S.nonzeros.Size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;

            FormRow( x, y, xSize, ySize, row, colIndices );

            for( unsigned j=0; j<row.Size(); ++j )
            {
                S.nonzeros.Push_back( row[j] );
                S.columnIndices.Push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.Push_back( S.nonzeros.Size() );
        double fillStopTime = mpi::Time();
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
        double constructStartTime = mpi::Time();
        HMat H( S, numLevels, maxRank, strong, xSize, ySize );
        double constructStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
        }

        // Test against a vector of all 1's
        Vector<Scalar> x;
        x.Resize( m );
        Scalar* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector of all ones...";
            std::cout.flush();
        }
        Vector<Scalar> y;
        double matVecStartTime = mpi::Time();
        H.Multiply( 1.0, x, y );
        double matVecStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime
                      << " seconds." << std::endl;
            if( print )
                y.Print("y := H x ~= S x");
        }

        // Pack the H-matrix
        Vector<byte> packedHMat;
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix...";
            std::cout.flush();
        }
        double packStartTime = mpi::Time();
        H.Pack( packedHMat );
        double packStopTime = mpi::Time();
        double sizeInMB = ((double)packedHMat.Size())/(1024.*1024.);
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
        double unpackStartTime = mpi::Time();
        HMat HCopy( packedHMat );
        double unpackStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
            if( print )
                HCopy.Print("Unpacked copy of H-matrix");
        }

        // Check that the copied H-matrix has the same action on our vector of
        // all 1's
        Vector<Scalar> z;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector...";
            std::cout.flush();
        }
        matVecStartTime = mpi::Time();
        HCopy.Multiply( 1.0, x, z );
        matVecStopTime = mpi::Time();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime
                      << " seconds." << std::endl;
            if( print )
                z.Print("z := HCopy x ~= S x");
            for( unsigned i=0; i<z.Height(); ++i )
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
    catch( ArgException& e ) { }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what()
                  << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    Finalize();
    return 0;
}
