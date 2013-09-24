/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_DISTHMAT2D_HPP
#define DMHM_DISTHMAT2D_HPP 1

#include "dmhm/core/mpi.hpp"
#include "dmhm/core/memory_map.hpp"
#include "dmhm/hmat2d.hpp"

namespace dmhm {

extern Timer timerGlobal;
// A distributed H-matrix class that assumes a 2d box domain and requires
// a power of two number of processes. It does not yet support implicit
// symmetry.
template<typename Scalar>
class DistHMat2d
{
public:
    typedef BASE(Scalar) Real;

    /*
     * Public data structures
     */
    class Teams
    {
    private:
        Vector<mpi::Comm> teams_, crossTeams_;
    public:
        Teams( mpi::Comm comm );
        ~Teams();

        int NumLevels() const;
        int TeamLevel( int level ) const;
        mpi::Comm Team( int level ) const;
        mpi::Comm CrossTeam( int inverseLevel ) const;

        void TreeSums
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeSumToRoots
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeBroadcasts
        ( Vector<Scalar>& buffer, const Vector<int>& sizes ) const;

        void TreeBroadcasts
        ( Vector<int>& buffer, const Vector<int>& sizes ) const;
    };

    /*
     * Public static member functions
     */
    static int SampleRank( int approxRank ) { return approxRank + Oversample(); }

    static std::size_t PackedSizes
    ( Vector<std::size_t>& packedSizes,
      const HMat2d<Scalar>& H, const Teams& teams );

    static std::size_t Pack
    ( Vector<byte*>& packedPieces,
      const HMat2d<Scalar>& H, const Teams& teams );

    static int ComputeLocalHeight
    ( int p, int rank, const HMat2d<Scalar>& H );

    static int ComputeLocalWidth
    ( int p, int rank, const HMat2d<Scalar>& H );

    static int ComputeFirstLocalRow
    ( int p, int rank, const HMat2d<Scalar>& H );

    static int ComputeFirstLocalCol
    ( int p, int rank, const HMat2d<Scalar>& H );

    static void ComputeLocalSizes
    ( Vector<int>& localSizes,
      const HMat2d<Scalar>& H );

    /*
     * Public non-static member functions
     */

    // Generate an empty H-matrix.
    DistHMat2d( const Teams& teams );

    // Generate an uninitialized H-matrix tree.
    DistHMat2d
    ( int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, const Teams& teams );

    // Generate an uninitialized H-matrix tree.
    DistHMat2d
    ( int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget, const Teams& teams );

    // Generate an H-matrix from sparse matrix.
    DistHMat2d
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, const Teams& teams );

    // Unpack our portion of a distributed H-matrix. The buffer should have been
    // generated from the above 'Pack' routine.
    DistHMat2d( const byte* packedPiece, const Teams& teams );

    ~DistHMat2d();

    void Clear();

    int Height() const;
    int Width() const;
    int MaxRank() const;
    int NumLevels() const;

    int LocalHeight() const;
    int LocalWidth() const;

    int LocalHeightPartner() const;
    int LocalWidthPartner() const;

    int FirstLocalRow() const;
    int FirstLocalCol() const;

    int FirstLocalXTarget() const;
    int FirstLocalXSource() const;
    int FirstLocalYTarget() const;
    int FirstLocalYSource() const;
    int FirstLocalZSource() const;
    int FirstLocalZTarget() const;

    int LocalXTargetSize() const;
    int LocalXSourceSize() const;
    int LocalYTargetSize() const;
    int LocalYSourceSize() const;
    int LocalZTargetSize() const;
    int LocalZSourceSize() const;

    void RequireRoot() const;

    // If this block is not low rank, throw an error
    int Rank() const;
    void SetGhostRank( int rank ); // this block must also be a ghost

    /*
     * Routines for visualizing the locally known H-matrix structure
     */
#ifdef HAVE_QT5
    void DisplayLocal( const std::string title="" ) const;
#endif
    // Compile this output with pdflatex+TikZ
    void LatexLocalStructure( const std::string basename ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptLocalStructure( const std::string basename ) const;

    void MemoryInfo
    ( double& numBasic, double& numNode, double& numNodeTmp,
      double& numLowRank, double& numLowRankTmp,
      double& numDense, double& numDenseTmp ) const;

    void PrintMemoryInfo
    ( const std::string tag = "", std::ostream& os = std::cout ) const;

    void PrintGlobalMemoryInfo
    ( const std::string tag = "", std::ostream& os = std::cout ) const;

    // Unpack this process's portion of the DistHMat2d
    std::size_t Unpack
    ( const byte* packedDistHMat, const Teams& teams );

    // Union the structure known in each block row and column at each level.
    void FormTargetGhostNodes();
    void FormSourceGhostNodes();

    // Return to the minimal local structure
    void PruneGhostNodes();

    // Set every admissible block to a random 'maxRank' matrix, and every dense
    // matrix to a random matrix.
    void SetToRandom();

    // A := alpha A
    void Scale( Scalar alpha );

    // A := alpha I + A
    void AddConstantToDiagonal( Scalar alpha );

    // estimate ||A||_2
    Real ParallelEstimateTwoNorm( Real theta, Real confidence);

    // A := inv(A)
    void SchulzInvert
    ( int numIterations, int multType=2, Real theta=1.5, Real confidence=6 );

    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const DistHMat2d<Scalar>& B );

    // A := B
    void CopyFrom( const DistHMat2d<Scalar>& B );

    // A := A^T
    void Transpose();

    // A := B^T
    void TransposeFrom( const DistHMat2d<Scalar>& B );

    // A := A^H
    void Adjoint();

    // A := B^H
    void AdjointFrom( const DistHMat2d<Scalar>& B );
    void AdjointCopy( const DistHMat2d<Scalar>& B );
    void AdjointPassData( const DistHMat2d<Scalar>& B );
    void AdjointPassDataCount
    ( const DistHMat2d<Scalar>& B,
      std::map<int,int>& sendSizes,
      std::map<int,int>& recvSizes ) const;
    void AdjointPassDataPack
    ( const DistHMat2d<Scalar>& B,
      Vector<Scalar>& buffer,
      std::map<int,int>& offsets ) const;
    void AdjointPassDataUnpack
    ( const DistHMat2d<Scalar>& B,
      const Vector<Scalar>& buffer,
      std::map<int,int>& offsets );

    // y := alpha H x
    void Multiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H x + beta y
    void Multiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H X
    void Multiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H X + beta Y
    void Multiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A B
    void Multiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C,
      int multType=0 );

    // C := alpha A B + beta C
    void Multiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
      Scalar beta,  DistHMat2d<Scalar>& C,
      int multType=0 );

    // y := alpha H^T x
    void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H^T x + beta y
    void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H^T X
    void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H^T X + beta Y
    void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A^T B
    void TransposeMultiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C );

    // C := alpha A^T B + beta C
    void TransposeMultiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
      Scalar beta,  DistHMat2d<Scalar>& C );

    // y := alpha H' x
    void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H' x + beta y
    void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H' X
    void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H' X + beta Y
    void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A' B
    void AdjointMultiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C );

    // C := alpha A' B + beta C
    void AdjointMultiply
    ( Scalar alpha, DistHMat2d<Scalar>& B,
      Scalar beta,  DistHMat2d<Scalar>& C );

private:
    /*
     * Private data structures
     */

    struct DistLowRank
    {
        int rank;
        Dense<Scalar> ULocal, VLocal;
    };

    struct DistLowRankGhost
    {
        int rank;
    };

    struct SplitLowRank
    {
        int rank;
        Dense<Scalar> D;
    };

    struct SplitLowRankGhost
    {
        int rank;
    };

    struct LowRankGhost
    {
        int rank;
    };

    struct SplitDense
    {
        Dense<Scalar> D;
    };

    struct Node
    {
        Vector<DistHMat2d*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
        Node
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget );
        ~Node();
        DistHMat2d& Child( int t, int s );
        const DistHMat2d& Child( int t, int s ) const;
    };
    Node* NewNode() const;

    enum BlockType
    {
        DIST_NODE,        // each side is distributed
        DIST_NODE_GHOST,  //
        SPLIT_NODE,       // each side is owned by a single process
        SPLIT_NODE_GHOST, //
        NODE,             // serial
        NODE_GHOST,       //

        DIST_LOW_RANK,        // each side is distributed
        DIST_LOW_RANK_GHOST,  //
        SPLIT_LOW_RANK,       // each side is given to a different process
        SPLIT_LOW_RANK_GHOST, //
        LOW_RANK,             // serial
        LOW_RANK_GHOST,       //

        SPLIT_DENSE,       // split between two processes
        SPLIT_DENSE_GHOST, //
        DENSE,             // serial
        DENSE_GHOST,       //

        EMPTY
    };

    struct Block
    {
        BlockType type;
        union Data
        {
            Node* N;

            DistLowRank* DF;
            SplitLowRank* SF;
            LowRank<Scalar>* F;

            SplitDense* SD;
            Dense<Scalar>* D;

            DistLowRankGhost* DFG;
            SplitLowRankGhost* SFG;
            LowRankGhost* FG;

            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Block();
        ~Block();
        void Clear();
    };

    struct BlockId
    {
        int level;
        int sourceOffset, targetOffset;
    };

    // TODO: Merge this with all of the MultiplyVector routines and create
    //       a full-fledged class.
    struct MultiplyVectorContext
    {
        struct DistNode
        {
            Vector<MultiplyVectorContext*> children;
            DistNode();
            ~DistNode();
            MultiplyVectorContext& Child( int t, int s );
            const MultiplyVectorContext& Child( int t, int s ) const;
        };
        typedef DistNode SplitNode;

        struct Block
        {
            BlockType type;
            union Data
            {
                DistNode* DN;
                SplitNode* SN;
                Vector<Scalar>* z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            Block();
            ~Block();
            void Clear();
            double TotalSize();
        };
        Block block;
        void Clear();
        double TotalSize();
    };

    // TODO: Merge this with all of the MultiplyDense routines and create
    //       a full-fledged class.
    struct MultiplyDenseContext
    {
        struct DistNode
        {
            Vector<MultiplyDenseContext*> children;
            DistNode();
            ~DistNode();
            MultiplyDenseContext& Child( int t, int s );
            const MultiplyDenseContext& Child( int t, int s ) const;
        };
        typedef DistNode SplitNode;

        struct Block
        {
            BlockType type;
            union Data
            {
                DistNode* DN;
                SplitNode* SN;
                Dense<Scalar>* Z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            Block();
            ~Block();
            void Clear();
            double TotalSize();
        };
        int numRhs;
        Block block;
        void Clear();
        double TotalSize();
    };

    /*
     * Private static functions
     */
    static const std::string BlockTypeString( BlockType type );

    static void PackedSizesRecursion
    ( Vector<std::size_t>& packedSizes,
      const Vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const HMat2d<Scalar>& H );

    static void PackRecursion
    ( Vector<byte**>& headPointers,
      const Vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const HMat2d<Scalar>& H );

    static void ComputeLocalDimensionRecursion
    ( int& localDim, int& xSize, int& ySize, int p, int rank );

    static void ComputeFirstLocalIndexRecursion
    ( int& firstLocalIndex, int xSize, int ySize, int p, int rank );

    static void ComputeLocalSizesRecursion
    ( int* localSizes, int teamSize, int xSize, int ySize );

    /*
     * Private non-static member functions
     */

    // This default constructure is purposely not publically accessible
    // because many routines are not functional without teams_ set.
    // This only constructs one level of the H-matrix.
    DistHMat2d();

    // This only constructs one level of the H-matrix
    DistHMat2d
    ( int numLevels, int maxRank, bool stronglyAdmissible,
      int sourceOffset, int targetOffset,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int xSource, int xTarget, int ySource, int yTarget,
      const Teams& teams, int level,
      bool inSourceTeam, bool inTargetTeam,
      int sourceRoot, int targetRoot );

    // This only constructs one level of the H-matrix from Sparse matrix
    DistHMat2d
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int sourceOffset, int targetOffset,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int xSource, int xTarget, int ySource, int yTarget,
      const Teams& teams, int level,
      bool inSourceTeam, bool inTargetTeam,
      int sourceRoot, int targetRoot );

    // Continue to fill the H-matrix tree
    void BuildTree();

    // Import Sparse matrix into H-matrix
    void ImportSparse
    ( const Sparse<Scalar>& S, int iOffset=0, int jOffset=0 );

    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

#ifdef HAVE_QT5
    void DisplayLocalRecursion
    ( Dense<double>* matrix, int mRatio, int nRatio ) const;
#endif
    void LatexLocalStructureRecursion
    ( std::ofstream& file, int globalheight ) const;
    void MScriptLocalStructureRecursion( std::ofstream& file ) const;

    void UnpackRecursion( const byte*& head );

    void FillTargetStructureRecursion
    ( Vector<std::set<int> >& targetStructure ) const;

    void FillSourceStructureRecursion
    ( Vector<std::set<int> >& sourceStructure ) const;

    void FindTargetGhostNodesRecursion
    ( const Vector<std::set<int> >& targetStructure,
      int sourceRoot, int targetRoot );

    void FindSourceGhostNodesRecursion
    ( const Vector<std::set<int> >& sourceStructure,
      int sourceRoot, int targetRoot );

    //
    // H-matrix/vector multiplication
    //
    void MultiplyVectorInitialize( MultiplyVectorContext& context ) const;

    void MultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void MultiplyVectorSums( MultiplyVectorContext& context ) const;
    void MultiplyVectorSumsCount( Vector<int>& sizes ) const;
    void MultiplyVectorSumsPack
    ( const MultiplyVectorContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyVectorSumsUnpack
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void MultiplyVectorPassData( MultiplyVectorContext& context ) const;
    void MultiplyVectorPassDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyVectorPassDataPack
    ( MultiplyVectorContext& context, Vector<Scalar>& sendBuffer,
      std::map<int,int>& offsets ) const;
    void MultiplyVectorPassDataUnpack
    ( MultiplyVectorContext& context, const Vector<Scalar>& recvBuffer,
      std::map<int,int>& recvOffsets ) const;

    void MultiplyVectorBroadcasts( MultiplyVectorContext& context ) const;
    void MultiplyVectorBroadcastsCount( Vector<int>& sizes ) const;
    void MultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void MultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // H-matrix/dense-matrix multiplication
    //
    void MultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;

    void MultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    void MultiplyDenseSums( MultiplyDenseContext& context ) const;
    void MultiplyDenseSumsCount( Vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseSumsPack
    ( const MultiplyDenseContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyDenseSumsUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void MultiplyDensePassData( MultiplyDenseContext& context ) const;
    void MultiplyDensePassDataCount
    ( std::map<int,int>& sendSizes,
      std::map<int,int>& recvSizes, int numRhs ) const;
    void MultiplyDensePassDataPack
    ( MultiplyDenseContext& context,
      Vector<Scalar>& buffer, std::map<int,int>& offsets ) const;
    void MultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, std::map<int,int>& offsets ) const;

    void MultiplyDenseBroadcasts( MultiplyDenseContext& context ) const;
    void MultiplyDenseBroadcastsCount
    ( Vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void MultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    //
    // Transpose H-matrix/vector multiplication
    //
    void TransposeMultiplyVectorInitialize
    ( MultiplyVectorContext& context ) const;

    void TransposeMultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void TransposeMultiplyVectorSums( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorSumsCount
    ( Vector<int>& sizes ) const;
    void TransposeMultiplyVectorSumsPack
    ( const MultiplyVectorContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void TransposeMultiplyVectorSumsUnpack
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void TransposeMultiplyVectorPassData
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const;
     void TransposeMultiplyVectorPassDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void TransposeMultiplyVectorPassDataPack
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal,
      Vector<Scalar>& sendBuffer, std::map<int,int>& offsets ) const;
    void TransposeMultiplyVectorPassDataUnpack
    ( MultiplyVectorContext& context, const Vector<Scalar>& recvBuffer,
      std::map<int,int>& recvOffsets ) const;

    void TransposeMultiplyVectorBroadcasts
    ( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorBroadcastsCount
    ( Vector<int>& sizes ) const;
    void TransposeMultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void TransposeMultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void TransposeMultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // Transpose H-matrix/dense-matrix multiplication
    //
    void TransposeMultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;

    void TransposeMultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    void TransposeMultiplyDenseSums( MultiplyDenseContext& context ) const;
    void TransposeMultiplyDenseSumsCount
    ( Vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseSumsPack
    ( const MultiplyDenseContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void TransposeMultiplyDenseSumsUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void TransposeMultiplyDensePassData
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const;
    void TransposeMultiplyDensePassDataCount
    ( std::map<int,int>& sendSizes,
      std::map<int,int>& recvSizes, int numRhs ) const;
    void TransposeMultiplyDensePassDataPack
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal,
      Vector<Scalar>& buffer, std::map<int,int>& offsets ) const;
    void TransposeMultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, std::map<int,int>& offsets ) const;

    void TransposeMultiplyDenseBroadcasts
    ( MultiplyDenseContext& context ) const;
    void TransposeMultiplyDenseBroadcastsCount
    ( Vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context,
      Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void TransposeMultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context,
      const Vector<Scalar>& buffer, Vector<int>& offsets ) const;

    void TransposeMultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    //
    // Adjoint H-matrix/vector multiplication
    //
    void AdjointMultiplyVectorInitialize
    ( MultiplyVectorContext& context ) const;

    void AdjointMultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void AdjointMultiplyVectorSums( MultiplyVectorContext& context ) const;

    void AdjointMultiplyVectorPassData
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const;

    void AdjointMultiplyVectorBroadcasts
    ( MultiplyVectorContext& context ) const;

    void AdjointMultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // Adjoint H-matrix/dense-matrix multiplication
    //
    void AdjointMultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;

    void AdjointMultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    void AdjointMultiplyDenseSums( MultiplyDenseContext& context ) const;

    void AdjointMultiplyDensePassData
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const;

    void AdjointMultiplyDenseBroadcasts( MultiplyDenseContext& context ) const;

    void AdjointMultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    //
    // H-matrix/H-matrix multiplication
    //
    void MultiplyHMatFullAccumulate
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C, Real twoNorm );
    void MultiplyHMatSingleLevelAccumulate
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C, Real twoNorm );
    void MultiplyHMatSingleUpdateAccumulate
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C, Real twoNorm );

    void MultiplyHMatFormGhostRanks( DistHMat2d<Scalar>& B );
    void MultiplyHMatFormGhostRanksCount
    ( const DistHMat2d<Scalar>& B,
      std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyHMatFormGhostRanksPack
    ( const DistHMat2d<Scalar>& B,
      Vector<int>& sendBuffer, std::map<int,int>& offsets ) const;
    void MultiplyHMatFormGhostRanksUnpack
    ( DistHMat2d<Scalar>& B,
      const Vector<int>& recvBuffer, std::map<int,int>& offsets );
    void MultiplyHMatMainSetUp
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C ) const;

    void MultiplyHMatMainPrecompute
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatMainSums
    ( DistHMat2d<Scalar>& B,
      DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainSumsCountA
    ( Vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsPackA
    ( Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsUnpackA
    ( const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainSumsCountB
    ( Vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsPackB
    ( Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsUnpackB
    ( const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainSumsCountC
    ( const DistHMat2d<Scalar>& B,
      const DistHMat2d<Scalar>& C,
      Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainSumsPackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainSumsUnpackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainPassData
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainPassDataCountA
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainPassDataPackA
    ( Vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatMainPassDataUnpackA
    ( const Vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainPassDataCountB
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainPassDataPackB
    ( Vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatMainPassDataUnpackB
    ( const Vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainPassDataCountC
    ( const DistHMat2d<Scalar>& B,
      const DistHMat2d<Scalar>& C,
      std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPassDataPackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      Vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPassDataUnpackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      const Vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainBroadcasts
    ( DistHMat2d<Scalar>& B,
      DistHMat2d<Scalar>& C,
      int startLevel, int endLevel, int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainBroadcastsCountA
    ( Vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsPackA
    ( Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsUnpackA
    ( const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainBroadcastsCountB
    ( Vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsPackB
    ( Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsUnpackB
    ( const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainBroadcastsCountC
    ( const DistHMat2d<Scalar>& B,
      const DistHMat2d<Scalar>& C,
      Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainBroadcastsPackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainBroadcastsUnpackC
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainPostcompute
    ( Scalar alpha, DistHMat2d<Scalar>& B,
                    DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatMainPostcomputeA // To be called from A
    ( int startLevel, int endLevel );
    void MultiplyHMatMainPostcomputeB // To be called from B
    ( int startLevel, int endLevel );
    void MultiplyHMatMainPostcomputeC // To be called from A
    ( Scalar alpha, const DistHMat2d<Scalar>& B,
                          DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPostcomputeCCleanup // To be called from C
    ( int startLevel, int endLevel );

    // TODO: Think of how to switch to the lockstep approach
    void MultiplyHMatParallelQR
    ( const Vector<int>& numQRs,
      const Vector<Dense<Scalar>*>& Xs,
      const Vector<int>& XOffsets,
            Vector<int>& halfHeights,
      const Vector<int>& halfHeightOffsets,
            Vector<Scalar>& qrBuffer,
      const Vector<int>& qrOffsets,
            Vector<Scalar>& tauBuffer,
      const Vector<int>& tauOffsets,
            Vector<Scalar>& qrWork ) const;

    void MultiplyHMatFHHPrecompute
    ( Scalar alpha, DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHSums
    ( Scalar alpha, DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel, int startUpdate, int endUpdate );
    void MultiplyHMatFHHSumsCount
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
            Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHSumsPack
    ( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHSumsUnpack
    ( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHPassData
    ( Scalar alpha, DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHPassDataCount
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
            std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHPassDataPack
    ( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHPassDataUnpack
    ( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      const Vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHBroadcasts
    ( Scalar alpha, DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHBroadcastsCount
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
            Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHBroadcastsPack
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHBroadcastsUnpack
    ( DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHPostcompute
    ( Scalar alpha, DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHPostcomputeC
    ( Scalar alpha, const DistHMat2d<Scalar>& B,
                          DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHPostcomputeCCleanup
    ( int startLevel, int endLevel ); // To be called from C

    void MultiplyHMatFHHFinalize
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate ) const;
    void MultiplyHMatFHHFinalizeCounts // To be called from C
    ( Vector<int>& numQrs,
      Vector<int>& numTargetFHH, Vector<int>& numSourceFHH,
      int startLevel, int endLevel );
    void MultiplyHMatFHHFinalizeMiddleUpdates
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
            Vector<Scalar>& allReduceBuffer,
            Vector<int>& middleOffsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHFinalizeLocalQR
    ( Vector<Dense<Scalar>*>& Xs, Vector<int>& XOffsets,
      Vector<Scalar>& tauBuffer, Vector<int>& tauOffsets,
      Vector<Scalar>& work,
      int startLevel, int endLevel );
    void MultiplyHMatFHHFinalizeOuterUpdates
    ( const DistHMat2d<Scalar>& B,
            DistHMat2d<Scalar>& C,
            Vector<Scalar>& allReduceBuffer,
            Vector<int>& leftOffsets,
            Vector<int>& rightOffsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHFinalizeFormLowRank
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
            Vector<Scalar>& allReduceBuffer,
            Vector<int>& leftOffsets,
            Vector<int>& middleOffsets,
            Vector<int>& rightOffsets,
            Vector<Real>& singularValues,
            Vector<Scalar>& U,
            Vector<Scalar>& VH,
            Vector<Scalar>& svdWork,
            Vector<Real>& svdRealWork,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void EVDTrunc( Dense<Scalar>& Q, Vector<Real>& w, Real epsilon );
    void SVDTrunc
    ( Dense<Scalar>& U, Vector<Real>& w, Dense<Scalar>& VH,
      Real epsilon, Real twoNorm );

    void MultiplyHMatCompress( Real twoNorm );
    void MultiplyHMatCompressLowRankCountAndResize( int rank );
    void MultiplyHMatCompressLowRankImport( int rank );
    void MultiplyHMatCompressImportU
    ( int rank, const Dense<Scalar>& U );
    void MultiplyHMatCompressImportV
    ( int rank, const Dense<Scalar>& V );
    void MultiplyHMatCompressFPrecompute();
    void MultiplyHMatCompressFReduces();
    void MultiplyHMatCompressFReducesCount
    ( Vector<int>& sizes ) const;
    void MultiplyHMatCompressFReducesPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatCompressFTreeReduces
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatCompressFReducesUnpack
    ( const Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatCompressFEigenDecomp();
    void MultiplyHMatCompressFPassMatrix();
    void MultiplyHMatCompressFPassMatrixCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyHMatCompressFPassMatrixPack
    ( Vector<Scalar>& buffer, std::map<int,int>& offsets ) const;
    void MultiplyHMatCompressFPassMatrixUnpack
    ( const Vector<Scalar>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatCompressFPassVector();
    void MultiplyHMatCompressFPassVectorCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyHMatCompressFPassVectorPack
    ( Vector<Real>& buffer, std::map<int,int>& offsets ) const;
    void MultiplyHMatCompressFPassVectorUnpack
    ( const Vector<Real>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatCompressFMidcompute( Real epsilon, Real twoNorm );
    void MultiplyHMatCompressFPassbackNum();
    void MultiplyHMatCompressFPassbackNumCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyHMatCompressFPassbackNumPack
    ( Vector<int>& buffer, std::map<int,int>& offsets ) const;
    void MultiplyHMatCompressFPassbackNumUnpack
    ( const Vector<int>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatCompressFPassbackData();
    void MultiplyHMatCompressFPassbackDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes );
    void MultiplyHMatCompressFPassbackDataPack
    ( Vector<Scalar>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatCompressFPassbackDataUnpack
    ( const Vector<Scalar>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatCompressFPostcompute( Real epsilon );
    void MultiplyHMatCompressFBroadcastsNum();
    void MultiplyHMatCompressFBroadcastsNumCount( Vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsNumPack
    ( Vector<int>& buffer, Vector<int>& offsets ) const;
    void MultiplyHMatCompressFTreeBroadcastsNum
    ( Vector<int>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsNumUnpack
    ( Vector<int>& buffer, Vector<int>& offsets );
    void MultiplyHMatCompressFBroadcasts();
    void MultiplyHMatCompressFBroadcastsCount( Vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyHMatCompressFTreeBroadcasts
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsUnpack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatCompressFFinalcompute();
    void MultiplyHMatCompressFCleanup();


    void MultiplyHMatRandomCompress( Real twoNorm );
    void MultiplyHMatRandomCompressPrecompute();
    void MultiplyHMatRandomCompressReducesOmegaTUV();
    void MultiplyHMatRandomCompressReducesOmegaTUVCount
    ( Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressReducesOmegaTUVPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressTreeReducesOmegaTUV
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressReducesOmegaTUVUnpack
    ( const Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressPassOmegaTUV();
    void MultiplyHMatRandomCompressPassOmegaTUVCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes );
    void MultiplyHMatRandomCompressPassOmegaTUVPack
    ( Vector<Scalar>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatRandomCompressPassOmegaTUVUnpack
    ( const Vector<Scalar>& buffer, std::map<int,int>& offsets );
    void MultiplyHMatRandomCompressBroadcastsOmegaTUV();
    void MultiplyHMatRandomCompressBroadcastsOmegaTUVCount
    ( Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsOmegaTUVPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyHMatRandomCompressTreeBroadcastsOmegaTUV
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsOmegaTUVUnpack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressMidcompute();
    void MultiplyHMatRandomCompressReducesTSqr();
    void MultiplyHMatRandomCompressReducesTSqrCount
    ( Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressReducesTSqrPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressTreeReducesTSqr
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressReducesTSqrUnpack
    ( const Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressPostcompute
    ( Real epsilon, Real relTol, Real twoNorm );
    void MultiplyHMatRandomCompressBroadcastsNum();
    void MultiplyHMatRandomCompressBroadcastsNumCount( Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsNumPack
    ( Vector<int>& buffer, Vector<int>& offsets ) const;
    void MultiplyHMatRandomCompressTreeBroadcastsNum
    ( Vector<int>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsNumUnpack
    ( Vector<int>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressBroadcasts();
    void MultiplyHMatRandomCompressBroadcastsCount( Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsPack
    ( Vector<Scalar>& buffer, Vector<int>& offsets ) const;
    void MultiplyHMatRandomCompressTreeBroadcasts
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatRandomCompressBroadcastsUnpack
    ( Vector<Scalar>& buffer, Vector<int>& offsets );
    void MultiplyHMatRandomCompressFinalcompute();


    // The following group of routines are used for HH compress.
    void MultiplyHMatFHHCompress
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate);
    void MultiplyHMatFHHCompressPrecompute
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressReduces
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressReducesCount
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHCompressReducesPack
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHCompressTreeReduces
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatFHHCompressReducesUnpack
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressMidcompute
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Real epsilon,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressBroadcasts
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressBroadcastsCount
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<int>& sizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHCompressBroadcastsPack
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHCompressTreeBroadcasts
    ( Vector<Scalar>& buffer, Vector<int>& sizes ) const;
    void MultiplyHMatFHHCompressBroadcastsUnpack
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      const Vector<Scalar>& buffer, Vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressPostcompute
    ( const DistHMat2d<Scalar>& B, DistHMat2d<Scalar>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressCleanup
    ( int startLevel, int endLevel );
    // TODO: The rest of the Compress routines

    /*
     * Private data
     */
    int numLevels_;
    int maxRank_;
    int sourceOffset_, targetOffset_;
    bool stronglyAdmissible_;

    int xSizeSource_, xSizeTarget_;
    int ySizeSource_, ySizeTarget_;
    int xSource_, xTarget_;
    int ySource_, yTarget_;
    Block block_;

    const Teams* teams_;
    int level_;
    bool inSourceTeam_, inTargetTeam_;
    int sourceRoot_, targetRoot_;

    // For temporary products in an H-matrix/H-matrix multiplication.
    // These are only needed for the C in C += alpha A B
    MemoryMap<int,MultiplyDenseContext>
        mainContextMap_, colFHHContextMap_, rowFHHContextMap_;
    MemoryMap<int,Dense<Scalar> > UMap_, VMap_, ZMap_, colXMap_, rowXMap_;
    // Tmp space for F compression
    Dense<Scalar> USqr_, VSqr_;
    Vector<Real> USqrEig_, VSqrEig_;
    Dense<Scalar> BSqrU_, BSqrVH_;
    Vector<Real> BSigma_;
    Dense<Scalar> BL_, BR_;
    bool haveDenseUpdate_, storedDenseUpdate_;
    Dense<Scalar> D_, SFD_;

    //Tmp space for random compression
    Dense<Scalar> OmegaTU_, OmegaTV_;
    Dense<Scalar> colTSqr_, rowTSqr_;

    // For the reuse of the computation of T1 = H Omega1 and T2 = H' Omega2 in
    // order to capture the column and row space, respectively, of H. These
    // variables could be mutable since they do not effect the usage of the
    // logical state of the class and simply help avoid redundant computation.
    bool beganRowSpaceComp_, finishedRowSpaceComp_,
         beganColSpaceComp_, finishedColSpaceComp_;
    Dense<Scalar> colOmega_, rowOmega_, colT_, rowT_;

    MemoryMap<int,Dense<Scalar> > colPinvMap_, rowPinvMap_;
    MemoryMap<int,Dense<Scalar> > colUSqrMap_, rowUSqrMap_;
    MemoryMap<int,Dense<Scalar> > BLMap_, BRMap_;
    MultiplyDenseContext colContext_, rowContext_;
};

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

/*
 * Private structure member functions
 */
template<typename Scalar>
inline
DistHMat2d<Scalar>::Node::Node
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget )
: children(16)
{
    xSourceSizes[0] = xSizeSource/2;
    xSourceSizes[1] = xSizeSource - xSourceSizes[0];
    ySourceSizes[0] = ySizeSource/2;
    ySourceSizes[1] = ySizeSource - ySourceSizes[0];

    sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0];
    sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0];
    sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1];
    sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1];

    xTargetSizes[0] = xSizeTarget/2;
    xTargetSizes[1] = xSizeTarget - xTargetSizes[0];
    yTargetSizes[0] = ySizeTarget/2;
    yTargetSizes[1] = ySizeTarget - yTargetSizes[0];

    targetSizes[0] = xTargetSizes[0]*yTargetSizes[0];
    targetSizes[1] = xTargetSizes[1]*yTargetSizes[0];
    targetSizes[2] = xTargetSizes[0]*yTargetSizes[1];
    targetSizes[3] = xTargetSizes[1]*yTargetSizes[1];
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::Node::~Node()
{
    for( int i=0; i<children.Size(); ++i )
        delete children[i];
    children.Clear();
}

template<typename Scalar>
inline DistHMat2d<Scalar>&
DistHMat2d<Scalar>::Node::Child( int t, int s )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline const DistHMat2d<Scalar>&
DistHMat2d<Scalar>::Node::Child( int t, int s ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline typename DistHMat2d<Scalar>::Node*
DistHMat2d<Scalar>::NewNode() const
{
    return
        new Node
        ( xSizeSource_, xSizeTarget_, ySizeSource_, ySizeTarget_ );
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar>
inline
DistHMat2d<Scalar>::Block::~Block()
{ Clear(); }

template<typename Scalar>
inline void
DistHMat2d<Scalar>::Block::Clear()
{
    switch( type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        delete data.N; break;

    case DIST_LOW_RANK:  delete data.DF; break;
    case SPLIT_LOW_RANK: delete data.SF; break;
    case LOW_RANK:       delete data.F;  break;

    case DIST_LOW_RANK_GHOST:  delete data.DFG; break;
    case SPLIT_LOW_RANK_GHOST: delete data.SFG; break;
    case LOW_RANK_GHOST:       delete data.FG;  break;

    case SPLIT_DENSE: delete data.SD; break;
    case DENSE:       delete data.D;  break;

    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case EMPTY:
        break;
    }
    type = EMPTY;
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyVectorContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyVectorContext;
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyVectorContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.Clear();
}

template<typename Scalar>
inline typename DistHMat2d<Scalar>::MultiplyVectorContext&
DistHMat2d<Scalar>::MultiplyVectorContext::DistNode::Child( int t, int s )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline const typename DistHMat2d<Scalar>::MultiplyVectorContext&
DistHMat2d<Scalar>::MultiplyVectorContext::DistNode::Child( int t, int s ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyVectorContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyVectorContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyVectorContext::Block::~Block()
{ Clear(); }

template<typename Scalar>
inline void
DistHMat2d<Scalar>::MultiplyVectorContext::Block::Clear()
{
    switch( type )
    {
    case DIST_NODE:
        delete data.DN; break;

    case SPLIT_NODE:
        delete data.SN; break;

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.z; break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
    type = EMPTY;
}

template<typename Scalar>
inline double
DistHMat2d<Scalar>::MultiplyVectorContext::Block::TotalSize()
{
    switch( type )
    {
    case DIST_NODE:
    {
        MultiplyVectorContext::DistNode& node = *data.DN;
        double totalSize = 0.0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                totalSize += node.Child(t,s).TotalSize();
        return totalSize;
    }
    case SPLIT_NODE:
    {
        MultiplyVectorContext::SplitNode& node = *data.SN;
        double totalSize = 0.0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                totalSize += node.Child(t,s).TotalSize();
        return totalSize;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    {
        Vector<Scalar>& ZVector = *data.z;
        return ZVector.Size();
    }
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
    return 0.0;
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::MultiplyVectorContext::Clear()
{ block.Clear(); }

template<typename Scalar>
inline double
DistHMat2d<Scalar>::MultiplyVectorContext::TotalSize()
{ return block.TotalSize(); }

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyDenseContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyDenseContext;
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyDenseContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.Clear();
}

template<typename Scalar>
inline typename DistHMat2d<Scalar>::MultiplyDenseContext&
DistHMat2d<Scalar>::MultiplyDenseContext::DistNode::Child( int t, int s )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyDenseContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline const typename
DistHMat2d<Scalar>::MultiplyDenseContext&
DistHMat2d<Scalar>::MultiplyDenseContext::DistNode::Child( int t, int s )
const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::MultiplyDenseContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.Size() != 16 )
        throw std::logic_error("children array not yet set up");
#endif
    return *children[s+4*t];
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyDenseContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar>
inline
DistHMat2d<Scalar>::MultiplyDenseContext::Block::~Block()
{ Clear(); }

template<typename Scalar>
inline void
DistHMat2d<Scalar>::MultiplyDenseContext::Block::Clear()
{
    switch( type )
    {
    case DIST_NODE:
        delete data.DN; break;

    case SPLIT_NODE:
        delete data.SN; break;

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.Z; break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
    type = EMPTY;
}

template<typename Scalar>
inline double
DistHMat2d<Scalar>::MultiplyDenseContext::Block::TotalSize()
{
    switch( type )
    {
    case DIST_NODE:
    {
        MultiplyDenseContext::DistNode& node = *data.DN;
        double totalSize = 0.0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                totalSize += node.Child(t,s).TotalSize();
        return totalSize;
    }
    case SPLIT_NODE:
    {
        MultiplyDenseContext::SplitNode& node = *data.SN;
        double totalSize = 0.0;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                totalSize += node.Child(t,s).TotalSize();
        return totalSize;
    }
    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
    {
        Dense<Scalar>& ZDense = *data.Z;
        return ZDense.Size();
    }
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY:
        break;
    }
    return 0.0;
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::MultiplyDenseContext::Clear()
{ block.Clear(); }

template<typename Scalar>
inline double
DistHMat2d<Scalar>::MultiplyDenseContext::TotalSize()
{ return block.TotalSize(); }

/*
 * Public member functions
 */

template<typename Scalar>
inline int
DistHMat2d<Scalar>::Height() const
{ return xSizeTarget_*ySizeTarget_; }

template<typename Scalar>
inline int
DistHMat2d<Scalar>::Width() const
{ return xSizeSource_*ySizeSource_; }

template<typename Scalar>
inline int
DistHMat2d<Scalar>::MaxRank() const
{ return maxRank_; }

template<typename Scalar>
inline int
DistHMat2d<Scalar>::NumLevels() const
{ return numLevels_; }

/*
 * Public structures member functions
 */

template<typename Scalar>
inline
DistHMat2d<Scalar>::Teams::Teams( mpi::Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::Teams");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing the number of teams
    // (and how many we're the root of)
    int numLevels=1;
    unsigned teamSize=p;
    while( teamSize != 1 )
    {
        ++numLevels;
        if( teamSize >= 4 )
            teamSize >>= 2;
        else // teamSize == 2
            teamSize = 1;
    }

    teams_.Resize( numLevels );
    mpi::CommDup( comm, teams_[0] );
    teamSize = p;
    for( int level=1; level<numLevels; ++level )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, teams_[level] );
    }

    crossTeams_.Resize( numLevels );
    mpi::CommDup( teams_[numLevels-1], crossTeams_[0] );
    for( int inverseLevel=1; inverseLevel<numLevels; ++inverseLevel )
    {
        const int level = numLevels-1-inverseLevel;
        teamSize = mpi::CommSize( teams_[level] );
        const int teamSizePrev = mpi::CommSize( teams_[level+1] );

        int color, key;
        if( teamSize == 2 )
        {
            color = rank / 2;
            key = rank % 2;
        }
        else
        {
            const int mod = rank % teamSizePrev;
            color = (rank/teamSize)*teamSizePrev + mod;
            key = (rank/teamSizePrev) % 4;
        }

        mpi::CommSplit( comm, color, key, crossTeams_[inverseLevel] );
    }
}

template<typename Scalar>
inline
DistHMat2d<Scalar>::Teams::~Teams()
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::~Teams");
#endif
    for( int i=0; i<teams_.Size(); ++i )
        mpi::CommFree( teams_[i] );
    for( int i=0; i<crossTeams_.Size(); ++i )
        mpi::CommFree( crossTeams_[i] );
}

template<typename Scalar>
inline int
DistHMat2d<Scalar>::Teams::NumLevels() const
{ return teams_.Size(); }

template<typename Scalar>
inline int
DistHMat2d<Scalar>::Teams::TeamLevel( int level ) const
{ return std::min(level,teams_.Size()-1); }

template<typename Scalar>
inline mpi::Comm
DistHMat2d<Scalar>::Teams::Team( int level ) const
{ return teams_[TeamLevel(level)]; }

template<typename Scalar>
inline mpi::Comm
DistHMat2d<Scalar>::Teams::CrossTeam( int inverseLevel ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::CrossTeam");
    if( inverseLevel >= crossTeams_.Size() )
        throw std::logic_error("Invalid cross team request");
#endif
    return crossTeams_[inverseLevel];
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::Teams::TreeSums
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::TreeSums");
#endif
    const int numLevels = NumLevels();
    const int numAllReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numAllReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - AllReduce over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numAllReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::AllReduce
        ( (const Scalar*)MPI_IN_PLACE, &buffer[0], partialSize, mpi::SUM,
          crossTeam );
        partialSize -= sizes[numAllReduces-1-i];
    }
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::Teams::TreeSumToRoots
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::TreeSumToRoots");
#endif
    const int numLevels = NumLevels();
    const int numReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Reduce to the root of each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        const int crossTeamRank = mpi::CommRank( crossTeam );
        if( crossTeamRank == 0 )
            mpi::Reduce
            ( (const Scalar*)MPI_IN_PLACE, &buffer[0],
              partialSize, mpi::SUM, 0, crossTeam );
        else
            mpi::Reduce
            ( &buffer[0], (Scalar*)0, partialSize, mpi::SUM, 0, crossTeam );
        partialSize -= sizes[numReduces-1-i];
    }
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::Teams::TreeBroadcasts
( Vector<Scalar>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
}

template<typename Scalar>
inline void
DistHMat2d<Scalar>::Teams::TreeBroadcasts
( Vector<int>& buffer, const Vector<int>& sizes ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat2d::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
        return;

    // Use O(log(p)) custom method:
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        mpi::Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
}

} // namespace dmhm

#endif // ifndef DMHM_DISTHMAT2D_HPP
