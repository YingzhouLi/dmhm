/*
   Distributed-Memory Hierarchical Matrices (DMHM): a prototype implementation
   of distributed-memory H-matrix arithmetic. 

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef DMHM_DIST_QUASI2D_HMAT_HPP
#define DMHM_DIST_QUASI2D_HMAT_HPP 1

#include "dmhm/building_blocks/mpi.hpp"
#include "dmhm/building_blocks/memory_map.hpp"
#include "dmhm/quasi2d_hmat.hpp"

namespace dmhm {

// A distributed H-matrix class that assumes a quasi2d box domain and requires
// a power of two number of processes. It does not yet support implicit 
// symmetry.
template<typename Scalar,bool Conjugated=true>
class DistQuasi2dHMat
{
public:
    typedef typename RealBase<Scalar>::type Real;

    /*
     * Public data structures
     */
    class Teams
    {
    private:
        std::vector<MPI_Comm> _teams, _crossTeams;
    public:
        Teams( MPI_Comm comm );
        ~Teams();

        unsigned NumLevels() const;
        unsigned TeamLevel( unsigned level ) const;
        MPI_Comm Team( unsigned level ) const;
        MPI_Comm CrossTeam( unsigned inverseLevel ) const;

        void TreeSums
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeSumToRoots
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeBroadcasts
        ( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const;

        void TreeBroadcasts
        ( std::vector<int>& buffer, const std::vector<int>& sizes ) const;
    };

    /*
     * Public static member functions
     */
    static int SampleRank( int approxRank ) { return approxRank + 4; }

    static std::size_t PackedSizes
    ( std::vector<std::size_t>& packedSizes,
      const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams );

    static std::size_t Pack
    ( std::vector<byte*>& packedPieces, 
      const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams );

    static int ComputeLocalHeight
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeLocalWidth
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeFirstLocalRow
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeFirstLocalCol
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static void ComputeLocalSizes
    ( std::vector<int>& localSizes, 
      const Quasi2dHMat<Scalar,Conjugated>& H );

    /*
     * Public non-static member functions
     */

    // Generate an empty H-matrix.
    DistQuasi2dHMat( const Teams& teams );

    // Generate an uninitialized H-matrix tree.
    DistQuasi2dHMat
    ( int numLevels, int maxRank, bool stronglyAdmissible, 
      int xSize, int ySize, int zSize,
      const Teams& teams );

    // Generate an uninitialized H-matrix tree.
    DistQuasi2dHMat
    ( int numLevels, int maxRank, bool stronglyAdmissible, 
      int xSizeSource, int xSizeTarget, 
      int ySizeSource, int ySizeTarget, int zSize, 
      const Teams& teams );

    // Unpack our portion of a distributed H-matrix. The buffer should have been
    // generated from the above 'Pack' routine.
    DistQuasi2dHMat( const byte* packedPiece, const Teams& teams );

    ~DistQuasi2dHMat();

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
    // Compile this output with pdflatex+TikZ
    void LatexWriteLocalStructure( const std::string basename ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptWriteLocalStructure( const std::string basename ) const;

    // Unpack this process's portion of the DistQuasi2dHMat
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

    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const DistQuasi2dHMat<Scalar,Conjugated>& B );

    // A := B
    void CopyFrom( const DistQuasi2dHMat<Scalar,Conjugated>& B );

    // A := A^T
    void Transpose();

    // A := B^T
    void TransposeFrom( const DistQuasi2dHMat<Scalar,Conjugated>& B );

    // A := A^H
    void Adjoint();

    // A := B^H
    void AdjointFrom( const DistQuasi2dHMat<Scalar,Conjugated>& B );

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
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int multType=0 );

    // C := alpha A B + beta C
    void Multiply
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,  DistQuasi2dHMat<Scalar,Conjugated>& C,
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
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C );

    // C := alpha A^T B + beta C
    void TransposeMultiply
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,  DistQuasi2dHMat<Scalar,Conjugated>& C );

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
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C );

    // C := alpha A' B + beta C
    void AdjointMultiply
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,  DistQuasi2dHMat<Scalar,Conjugated>& C );

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
        std::vector<DistQuasi2dHMat*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
        Node
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget,
          int zSize );
        ~Node();
        DistQuasi2dHMat& Child( int t, int s );
        const DistQuasi2dHMat& Child( int t, int s ) const;
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
            LowRank<Scalar,Conjugated>* F;

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
            std::vector<MultiplyVectorContext*> children;
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
        };
        Block block;
        void Clear();
    };

    // TODO: Merge this with all of the MultiplyDense routines and create 
    //       a full-fledged class.
    struct MultiplyDenseContext
    {
        struct DistNode
        {
            std::vector<MultiplyDenseContext*> children;
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
        };
        int numRhs;
        Block block;
        void Clear();
    };

    /*
     * Private static functions
     */
    static const std::string BlockTypeString( BlockType type );

    static void PackedSizesRecursion
    ( std::vector<std::size_t>& packedSizes,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMat<Scalar,Conjugated>& H );

    static void PackRecursion
    ( std::vector<byte**>& headPointers,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMat<Scalar,Conjugated>& H );

    static void ComputeLocalDimensionRecursion
    ( int& localDim, int& xSize, int& ySize, int& zSize, int p, int rank );

    static void ComputeFirstLocalIndexRecursion
    ( int& firstLocalIndex, int xSize, int ySize, int zSize, int p, int rank );

    static void ComputeLocalSizesRecursion
    ( int* localSizes, int teamSize, int xSize, int ySize, int zSize );

    /*
     * Private non-static member functions
     */
    
    // This default constructure is purposely not publically accessible
    // because many routines are not functional without _teams set.
    // This only constructs one level of the H-matrix.
    DistQuasi2dHMat();
   
    // This only constructs one level of the H-matrix
    DistQuasi2dHMat
    ( int numLevels, int maxRank, bool stronglyAdmissible, 
      int sourceOffset, int targetOffset,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int zSize, int xSource, int xTarget, int ySource, int yTarget,
      const Teams& teams, int level, 
      bool inSourceTeam, bool inTargetTeam, 
      int sourceRoot, int targetRoot );

    // Continue to fill the H-matrix tree
    void BuildTree(); 

    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void LatexWriteLocalStructureRecursion
    ( std::ofstream& file, int globalheight ) const;
    void MScriptWriteLocalStructureRecursion( std::ofstream& file ) const;
    
    void UnpackRecursion( const byte*& head );

    void FillTargetStructureRecursion
    ( std::vector<std::set<int> >& targetStructure ) const;

    void FillSourceStructureRecursion
    ( std::vector<std::set<int> >& sourceStructure ) const;

    void FindTargetGhostNodesRecursion
    ( const std::vector<std::set<int> >& targetStructure,
      int sourceRoot, int targetRoot );

    void FindSourceGhostNodesRecursion
    ( const std::vector<std::set<int> >& sourceStructure,
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
    void MultiplyVectorSumsCount( std::vector<int>& sizes ) const;
    void MultiplyVectorSumsPack
    ( const MultiplyVectorContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorSumsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void MultiplyVectorPassData( MultiplyVectorContext& context ) const;
    void MultiplyVectorPassDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyVectorPassDataPack
    ( MultiplyVectorContext& context, std::vector<Scalar>& sendBuffer, 
      std::map<int,int>& offsets ) const;
    void MultiplyVectorPassDataUnpack
    ( MultiplyVectorContext& context, const std::vector<Scalar>& recvBuffer,
      std::map<int,int>& recvOffsets ) const;

    void MultiplyVectorBroadcasts( MultiplyVectorContext& context ) const;
    void MultiplyVectorBroadcastsCount( std::vector<int>& sizes ) const;
    void MultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

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
    void MultiplyDenseSumsCount( std::vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseSumsPack
    ( const MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseSumsUnpack
    ( MultiplyDenseContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void MultiplyDensePassData( MultiplyDenseContext& context ) const;
    void MultiplyDensePassDataCount
    ( std::map<int,int>& sendSizes, 
      std::map<int,int>& recvSizes, int numRhs ) const;
    void MultiplyDensePassDataPack
    ( MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const;
    void MultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const;

    void MultiplyDenseBroadcasts( MultiplyDenseContext& context ) const;
    void MultiplyDenseBroadcastsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

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
    ( std::vector<int>& sizes ) const;
    void TransposeMultiplyVectorSumsPack
    ( const MultiplyVectorContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorSumsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void TransposeMultiplyVectorPassData
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const;
     void TransposeMultiplyVectorPassDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void TransposeMultiplyVectorPassDataPack
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal,
      std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets ) const;
    void TransposeMultiplyVectorPassDataUnpack
    ( MultiplyVectorContext& context, const std::vector<Scalar>& recvBuffer,
      std::map<int,int>& recvOffsets ) const;

    void TransposeMultiplyVectorBroadcasts
    ( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorBroadcastsCount
    ( std::vector<int>& sizes ) const;
    void TransposeMultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

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
    ( std::vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseSumsPack
    ( const MultiplyDenseContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseSumsUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void TransposeMultiplyDensePassData
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const;
    void TransposeMultiplyDensePassDataCount
    ( std::map<int,int>& sendSizes, 
      std::map<int,int>& recvSizes, int numRhs ) const;
    void TransposeMultiplyDensePassDataPack
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal,
      std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const;
    void TransposeMultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::map<int,int>& offsets ) const;

    void TransposeMultiplyDenseBroadcasts
    ( MultiplyDenseContext& context ) const;
    void TransposeMultiplyDenseBroadcastsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

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
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C );
    void MultiplyHMatSingleLevelAccumulate
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C );
    void MultiplyHMatSingleUpdateAccumulate
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C );

    void MultiplyHMatFormGhostRanks( DistQuasi2dHMat<Scalar,Conjugated>& B );   
    void MultiplyHMatFormGhostRanksCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      std::map<int,int>& sendSizes, std::map<int,int>& recvSizes ) const;
    void MultiplyHMatFormGhostRanksPack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      std::vector<int>& sendBuffer, std::map<int,int>& offsets ) const;
    void MultiplyHMatFormGhostRanksUnpack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      const std::vector<int>& recvBuffer, std::map<int,int>& offsets );
    void MultiplyHMatMainSetUp
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    void MultiplyHMatMainPrecompute
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatMainSums
    ( DistQuasi2dHMat<Scalar,Conjugated>& B, 
      DistQuasi2dHMat<Scalar,Conjugated>& C, 
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainSumsCountA
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsPackA
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
      int startLevel, int endLevel ) const; 
    void MultiplyHMatMainSumsUnpackA
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainSumsCountB
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainSumsPackB
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const; 
    void MultiplyHMatMainSumsUnpackB
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainSumsCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<int>& sizes, 
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainSumsPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const; 
    void MultiplyHMatMainSumsUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainPassData
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainPassDataCountA
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainPassDataPackA
    ( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatMainPassDataUnpackA
    ( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainPassDataCountB
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainPassDataPackB
    ( std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatMainPassDataUnpackB
    ( const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainPassDataCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPassDataPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPassDataUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainBroadcasts
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C, 
      int startLevel, int endLevel, int startUpdate, int endUpdate );
    // To be called from A
    void MultiplyHMatMainBroadcastsCountA
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsPackA
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsUnpackA
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from B
    void MultiplyHMatMainBroadcastsCountB
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsPackB
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatMainBroadcastsUnpackB
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    // To be called from A
    void MultiplyHMatMainBroadcastsCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C, 
      std::vector<int>& sizes, 
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainBroadcastsPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainBroadcastsUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void MultiplyHMatMainPostcompute
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate );
    void MultiplyHMatMainPostcomputeA // To be called from A
    ( int startLevel, int endLevel ); 
    void MultiplyHMatMainPostcomputeB // To be called from B
    ( int startLevel, int endLevel ); 
    void MultiplyHMatMainPostcomputeC // To be called from A
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatMainPostcomputeCCleanup // To be called from C
    ( int startLevel, int endLevel ); 

    // TODO: Think of how to switch to the lockstep approach
    void MultiplyHMatParallelQR
    ( const std::vector<int>& numQRs,
      const std::vector<Dense<Scalar>*>& Xs,      
      const std::vector<int>& XOffsets,
            std::vector<int>& halfHeights,
      const std::vector<int>& halfHeightOffsets,
            std::vector<Scalar>& qrBuffer,
      const std::vector<int>& qrOffsets,
            std::vector<Scalar>& tauBuffer,
      const std::vector<int>& tauOffsets,
            std::vector<Scalar>& qrWork ) const;

    void MultiplyHMatFHHPrecompute
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHSums
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHSumsCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<int>& sizes, 
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHSumsPack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHSumsUnpack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHPassData
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHPassDataCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHPassDataPack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& sendBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHPassDataUnpack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& recvBuffer, std::map<int,int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHBroadcasts
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHBroadcastsCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<int>& sizes, 
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHBroadcastsPack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHBroadcastsUnpack
    ( DistQuasi2dHMat<Scalar,Conjugated>& B,
      DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update );

    void MultiplyHMatFHHPostcompute
    ( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                    DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate );
    void MultiplyHMatFHHPostcomputeC
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHPostcomputeCCleanup
    ( int startLevel, int endLevel ); // To be called from C

    void MultiplyHMatFHHFinalize
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate ) const;
    void MultiplyHMatFHHFinalizeCounts // To be called from C
    ( std::vector<int>& numQrs, 
      std::vector<int>& numTargetFHH, std::vector<int>& numSourceFHH,
      int startLevel, int endLevel );
    void MultiplyHMatFHHFinalizeMiddleUpdates
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& allReduceBuffer,
            std::vector<int>& middleOffsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHFinalizeLocalQR
    ( std::vector<Dense<Scalar>*>& Xs, std::vector<int>& XOffsets,
      std::vector<Scalar>& tauBuffer, std::vector<int>& tauOffsets,
      std::vector<Scalar>& work,
      int startLevel, int endLevel );
    void MultiplyHMatFHHFinalizeOuterUpdates
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& allReduceBuffer,
            std::vector<int>& leftOffsets, 
            std::vector<int>& rightOffsets,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;
    void MultiplyHMatFHHFinalizeFormLowRank
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& allReduceBuffer,
            std::vector<int>& leftOffsets,
            std::vector<int>& middleOffsets,
            std::vector<int>& rightOffsets,
            std::vector<Real>& singularValues,
            std::vector<Scalar>& U,
            std::vector<Scalar>& VH,
            std::vector<Scalar>& svdWork,
            std::vector<Real>& svdRealWork,
      int startLevel, int endLevel,
      int startUpdate, int endUpdate, int update ) const;

    void EVDTrunc
    ( Dense<Scalar>& Q, std::vector<Real>& w, Real error );
    void SVDTrunc
    ( Dense<Scalar>& U, std::vector<Real>& w,
      Dense<Scalar>& VH,
      Real error );

    void MultiplyHMatCompress( int startLevel, int endLevel );
    void MultiplyHMatCompressLowRankCountAndResize( int rank );
    void MultiplyHMatCompressLowRankImport( int rank );
    void MultiplyHMatCompressImportU
    ( int rank, const Dense<Scalar>& U );
    void MultiplyHMatCompressImportV
    ( int rank, const Dense<Scalar>& V );
    void MultiplyHMatCompressFPrecompute
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFReduces
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFReducesCount
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFReducesPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFTreeReduces
    ( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const;
    void MultiplyHMatCompressFReducesUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFEigenDecomp
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFPassMatrix
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFPassMatrixCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassMatrixPack
    ( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassMatrixUnpack
    ( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFPassVector
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFPassVectorCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassVectorPack
    ( std::vector<Real>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassVectorUnpack
    ( const std::vector<Real>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFMidcompute
    ( Real error, int startLevel, int endLevel );
    // Did not used.
    void MultiplyHMatCompressFEigenTrunc
    ( Real error, int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackNum
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackNumCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassbackNumPack
    ( std::vector<int>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFPassbackNumUnpack
    ( const std::vector<int>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackData
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackDataCount
    ( std::map<int,int>& sendSizes, std::map<int,int>& recvSizes,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackDataPack
    ( std::vector<Scalar>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFPassbackDataUnpack
    ( const std::vector<Scalar>& buffer, std::map<int,int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFPostcompute
    ( Real error, int startLevel, int endLevel );
    void MultiplyHMatCompressFBroadcastsNum
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFBroadcastsNumCount
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFBroadcastsNumPack
    ( std::vector<int>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFTreeBroadcastsNum
    ( std::vector<int>& buffer, std::vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsNumUnpack
    ( std::vector<int>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFBroadcasts
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFBroadcastsCount
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatCompressFTreeBroadcasts
    ( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const;
    void MultiplyHMatCompressFBroadcastsUnpack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatCompressFFinalcompute
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFCleanup
    ( int startLevel, int endLevel );
    void MultiplyHMatCompressFCompressless
    ( int startLevel, int endLevel );


    // The following group of routines are used for HH compress.
    void MultiplyHMatFHHCompress
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate);
    void MultiplyHMatFHHCompressSum
    ( int startLevel, int endLevel );
    void MultiplyHMatFHHCompressPrecompute
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      int startLevel, int endLevel, 
      int startUpdate, int endUpdate, int update );
    void MultiplyHMatFHHCompressReduces
    ( int startLevel, int endLevel );
    void MultiplyHMatFHHCompressReducesCount
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatFHHCompressReducesPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatFHHCompressTreeReduces
    ( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const;
    void MultiplyHMatFHHCompressReducesUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatFHHCompressMidcompute
    ( Real error, int startLevel, int endLevel );
    void MultiplyHMatFHHCompressBroadcasts
    ( int startLevel, int endLevel );
    void MultiplyHMatFHHCompressBroadcastsCount
    ( std::vector<int>& sizes, int startLevel, int endLevel ) const;
    void MultiplyHMatFHHCompressBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel ) const;
    void MultiplyHMatFHHCompressTreeBroadcasts
    ( std::vector<Scalar>& buffer, std::vector<int>& sizes ) const;
    void MultiplyHMatFHHCompressBroadcastsUnpack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      int startLevel, int endLevel );
    void MultiplyHMatFHHCompressPostcompute
    ( int startLevel, int endLevel );
    void MultiplyHMatFHHCompressCleanup
    ( int startLevel, int endLevel );
    // TODO: The rest of the Compress routines

    /*
     * Private data
     */
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Block _block;

    const Teams* _teams;
    int _level;
    bool _inSourceTeam, _inTargetTeam;
    int _sourceRoot, _targetRoot;

    // For temporary products in an H-matrix/H-matrix multiplication. 
    // These are only needed for the C in C += alpha A B
    MemoryMap<int,MultiplyDenseContext> 
        _mainContextMap, _colFHHContextMap, _rowFHHContextMap;
    MemoryMap<int,Dense<Scalar> > _UMap, _VMap, _ZMap, _colXMap, _rowXMap;
    MemoryMap<int,Dense<Scalar> > _HUMap, _HVMap, _HZMap;
    // _colSqrMap stores (HHR)'(HHR), _rowSqrMap stores (L'HH)(L'HH)'
    // Also they will be used to store low-rank square matrix.
    MemoryMap<int,Dense<Scalar> > _colSqrMap, _rowSqrMap;
    // _colSqrEigMap stores the eigenvalues of _colSqrMap
    // _rowSqrEigMap stores the eigenvalues of _rowSqrMap
    MemoryMap<int,std::vector<Real> > _colSqrEigMap, _rowSqrEigMap;
    // Tmp space for F compression
    Dense<Scalar> _Utmp, _Vtmp;
    Dense<Scalar> _USqr, _VSqr;
    std::vector<Real> _USqrEig, _VSqrEig;
    Dense<Scalar> _BSqr;
    std::vector<Real> _BSqrEig;
    Dense<Scalar> _BSqrU, _BSqrVH;
    std::vector<Real> _BSigma;
    Dense<Scalar> _BL, _BR;
    bool _haveDenseUpdate, _storedDenseUpdate;
    Dense<Scalar> _D, _SFD;

    // For the reuse of the computation of T1 = H Omega1 and T2 = H' Omega2 in 
    // order to capture the column and row space, respectively, of H. These 
    // variables could be mutable since they do not effect the usage of the 
    // logical state of the class and simply help avoid redundant computation.
    bool _beganRowSpaceComp, _finishedRowSpaceComp,
         _beganColSpaceComp, _finishedColSpaceComp;
    Dense<Scalar> _colOmega, _rowOmega, _colT, _rowT;
    
    Dense<Scalar> _colU, _rowU, _colPinv, _rowPinv;
    Dense<Scalar> _colUSqr, _rowUSqr;
    std::vector<Real> _colUSqrEig, _rowUSqrEig;
    MultiplyDenseContext _colContext, _rowContext;
};

} // namespace dmhm

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

namespace dmhm {

/*
 * Private structure member functions
 */
template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Node::Node
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize )
: children(16)
{
    xSourceSizes[0] = xSizeSource/2;
    xSourceSizes[1] = xSizeSource - xSourceSizes[0];
    ySourceSizes[0] = ySizeSource/2;
    ySourceSizes[1] = ySizeSource - ySourceSizes[0];

    sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0]*zSize;
    sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0]*zSize;
    sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1]*zSize;
    sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1]*zSize;

    xTargetSizes[0] = xSizeTarget/2;
    xTargetSizes[1] = xSizeTarget - xTargetSizes[0];
    yTargetSizes[0] = ySizeTarget/2;
    yTargetSizes[1] = ySizeTarget - yTargetSizes[0];

    targetSizes[0] = xTargetSizes[0]*yTargetSizes[0]*zSize;
    targetSizes[1] = xTargetSizes[1]*yTargetSizes[0]*zSize;
    targetSizes[2] = xTargetSizes[0]*yTargetSizes[1]*zSize;
    targetSizes[3] = xTargetSizes[1]*yTargetSizes[1]*zSize;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline DistQuasi2dHMat<Scalar,Conjugated>&
DistQuasi2dHMat<Scalar,Conjugated>::Node::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const DistQuasi2dHMat<Scalar,Conjugated>&
DistQuasi2dHMat<Scalar,Conjugated>::Node::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMat<Scalar,Conjugated>::Node*
DistQuasi2dHMat<Scalar,Conjugated>::NewNode() const
{
    return 
        new Node
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize );
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Block::Block()
: type(EMPTY), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Block::~Block()
{ 
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Block::Clear()
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

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyVectorContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::
Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::
Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::~Block()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::Clear()
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

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Clear()
{
    block.Clear();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyDenseContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::
Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const typename 
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::
Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseContext::DistNode::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::~Block()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::Clear()
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

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Clear()
{
    block.Clear();
}

/*
 * Public member functions
 */

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::Height() const
{
    return _xSizeTarget*_ySizeTarget*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::Width() const
{
    return _xSizeSource*_ySizeSource*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::MaxRank() const
{
    return _maxRank;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::NumLevels() const
{
    return _numLevels;
}

/*
 * Public structures member functions
 */

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Teams::Teams( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::Teams");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing the number of teams
    // (and how many we're the root of)
    unsigned numLevels=1, teamSize=p;
    while( teamSize != 1 )
    {
        ++numLevels;
        if( teamSize >= 4 )
            teamSize >>= 2;
        else // teamSize == 2
            teamSize = 1;
    }

    _teams.resize( numLevels );
    mpi::CommDup( comm, _teams[0] );
    teamSize = p;
    for( unsigned level=1; level<numLevels; ++level )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, _teams[level] );
    }

    _crossTeams.resize( numLevels );
    mpi::CommDup( _teams[numLevels-1], _crossTeams[0] );
    for( unsigned inverseLevel=1; inverseLevel<numLevels; ++inverseLevel )
    {
        const int level = numLevels-1-inverseLevel;
        teamSize = mpi::CommSize( _teams[level] );
        const int teamSizePrev = mpi::CommSize( _teams[level+1] );

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

        mpi::CommSplit( comm, color, key, _crossTeams[inverseLevel] );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Teams::~Teams()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::~Teams");
#endif
    for( unsigned i=0; i<_teams.size(); ++i )
        mpi::CommFree( _teams[i] );
    for( unsigned i=0; i<_crossTeams.size(); ++i )
        mpi::CommFree( _crossTeams[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline unsigned
DistQuasi2dHMat<Scalar,Conjugated>::Teams::NumLevels() const
{
    return _teams.size();
}

template<typename Scalar,bool Conjugated>
inline unsigned
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TeamLevel( unsigned level ) const
{
    return std::min(level,(unsigned)_teams.size()-1);
}

template<typename Scalar,bool Conjugated>
inline MPI_Comm
DistQuasi2dHMat<Scalar,Conjugated>::Teams::Team
( unsigned level ) const
{
    return _teams[TeamLevel(level)];
}

template<typename Scalar,bool Conjugated>
inline MPI_Comm
DistQuasi2dHMat<Scalar,Conjugated>::Teams::CrossTeam
( unsigned inverseLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::CrossTeam");
    if( inverseLevel >= _crossTeams.size() )
        throw std::logic_error("Invalid cross team request");
    PopCallStack();
#endif
    return _crossTeams[inverseLevel];
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeSums
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeSums");
#endif
    const int numLevels = NumLevels();
    const int numAllReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numAllReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
    {
#ifndef RELEASE
    PopCallStack();
#endif
        return;
    }

    // Use O(log(p)) custom method: 
    // - AllReduce over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numAllReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        MPI_Comm crossTeam = CrossTeam( i+1 );
        mpi::AllReduce
        ( (const Scalar*)MPI_IN_PLACE, &buffer[0], partialSize, MPI_SUM,
          crossTeam );
        partialSize -= sizes[numAllReduces-1-i];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeSumToRoots
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeSumToRoots");
#endif
    const int numLevels = NumLevels();
    const int numReduces = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numReduces; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
    {
#ifndef RELEASE
    PopCallStack();
#endif
        return;
    }

    // Use O(log(p)) custom method: 
    // - Reduce to the root of each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numReduces; ++i )
    {
        if( partialSize == 0 )
            break;
        MPI_Comm crossTeam = CrossTeam( i+1 );
        const int crossTeamRank = mpi::CommRank( crossTeam );
        if( crossTeamRank == 0 )
            mpi::Reduce
            ( (const Scalar*)MPI_IN_PLACE, &buffer[0], 
              partialSize, 0, MPI_SUM, crossTeam );
        else
            mpi::Reduce( &buffer[0], 0, partialSize, 0, MPI_SUM, crossTeam );
        partialSize -= sizes[numReduces-1-i];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeBroadcasts
( std::vector<Scalar>& buffer, const std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
    {
#ifndef RELEASE
    PopCallStack();
#endif
        return;
    }

    // Use O(log(p)) custom method: 
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        MPI_Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeBroadcasts
( std::vector<int>& buffer, const std::vector<int>& sizes ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int numBroadcasts = numLevels-1;

    int totalSize = 0;
    for( int i=0; i<numBroadcasts; ++i )
        totalSize += sizes[i];

    if( totalSize == 0 )
    {
#ifndef RELEASE
    PopCallStack();
#endif
        return;
    }

    // Use O(log(p)) custom method: 
    // - Broadcast over each cross communicator
    int partialSize = totalSize;
    for( int i=0; i<numBroadcasts; ++i )
    {
        if( partialSize == 0 )
            break;
        MPI_Comm crossTeam = CrossTeam( i+1 );
        mpi::Broadcast( &buffer[0], partialSize, 0, crossTeam );
        partialSize -= sizes[numBroadcasts-1-i];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}


} // namespace dmhm 

#endif // DMHM_DIST_QUASI2D_HMAT_HPP
