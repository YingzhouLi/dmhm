/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

namespace {

// Manually import Knuth's multiplication constant, 6364136223846793005,
// and his additive constant, 1442695040888963407.
//
// We initialize the state to an arbitrary value.
const dmhm::ExpandedUInt64 serialMultValue={{32557U,19605U,62509U,22609U}};
const dmhm::ExpandedUInt64 serialAddValue={{33103U,63335U,31614U,5125U}};
dmhm::ExpandedUInt64 serialLcgValue={{03U,17U,19U,86U}};

// We initialize the state to an arbitrary value and set the coefficients
// equal to the serial case by default.
dmhm::ExpandedUInt64 parallelMultValue=serialMultValue,
                    parallelAddValue=serialAddValue,
                    parallelLcgValue=serialLcgValue;

} // anonymous namespace

namespace dmhm {

// x := x/2
void Halve( ExpandedUInt64& x )
{
    x[0] >>= 1;
    x[0] |= ((x[1]&0x1) << 16);
    x[1] >>= 1;
    x[1] |= ((x[2]&0x1) << 16);
    x[2] >>= 1;
    x[2] |= ((x[3]&0x1) << 16);
    x[3] >>= 1;
    // The 16th bit of x[3] must be zero since we require the 17th and above
    // to have been zero upon entry.
}

// Return x^n with O(log2(n)) work. This is the "Right-to-left binary method for
// exponentiation" from Knuth's 'Seminumerical Algorithms' volume of TAOCP.
ExpandedUInt64 IntegerPowerWith64BitMod( ExpandedUInt64 x, ExpandedUInt64 n )
{
    ExpandedUInt64 N=n, Z=x, Y={{1U,0U,0U,0U}};
    if( N[0]==0 && N[1]==0 && N[2]==0 && N[3]==0 )
        return Y;
    while( 1 )
    {
        const bool odd = ( N[0] & 1U );
        Halve( N );
        if( odd )
        {
            Y = MultiplyWith64BitMod( Z, Y );
            if( N[0]==0 && N[1]==0 && N[2]==0 && N[3]==0 )
                break;
        }
        Z = MultiplyWith64BitMod( Z, Z );
    }
    return Y;
}

void SeedSerialLcg( UInt64 seed )
{ ::serialLcgValue = Expand( seed ); }

void SeedParallelLcg( UInt32 rank, UInt32 commSize, UInt64 globalSeed )
{
    // Compute a^rank and a^commSize in O(log2(commSize)) work.
    const ExpandedUInt64 myMultValue =
        IntegerPowerWith64BitMod( ::serialMultValue, Expand(rank) );
    ::parallelMultValue =
        IntegerPowerWith64BitMod( ::serialMultValue, Expand(commSize) );

    // Compute (a^rank-1)/(a-1) and (a^commSize-1)/(a-1) in O(commSize) work.
    // This could almost certainly be optimized, but its execution time is
    // probably ignorable.
    ExpandedUInt64 Y={{0U,0U,0U,0U}}, one={{1U,0U,0U,0U}};
    for( unsigned j=0; j<rank; ++j )
    {
        Y = MultiplyWith64BitMod( Y, ::serialMultValue );
        Y = AddWith64BitMod( Y, one );
    }
    const ExpandedUInt64 myAddValue =
        MultiplyWith64BitMod( Y, ::serialAddValue );
    for( unsigned j=rank; j<commSize; ++j )
    {
        Y = MultiplyWith64BitMod( Y, ::serialMultValue );
        Y = AddWith64BitMod( Y, one );
    }
    ::parallelAddValue = MultiplyWith64BitMod( Y, ::serialAddValue );

    // Set our local value equal to
    //     X_rank := a^rank X_0 + (a^rank-1)/(a-1) c mod 2^64
    // where X_0 is 'globalSeed'.
    ::parallelLcgValue = Expand( globalSeed );
    ManualLcg( myMultValue, myAddValue, ::parallelLcgValue );
}

// Return a uniform sample from [0,2^64)
UInt64 SerialLcg()
{
    UInt64 value = Deflate( ::serialLcgValue );
    ManualLcg( ::serialMultValue, ::serialAddValue, ::serialLcgValue );
    return value;
}

// Return a uniform sample from [0,2^64)
UInt64 ParallelLcg()
{
    UInt64 value = Deflate( ::parallelLcgValue );
    ManualLcg( ::parallelMultValue, ::parallelAddValue, ::parallelLcgValue );
    return value;
}

} // namespace dmhm
