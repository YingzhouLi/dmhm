/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
#ifdef HAVE_QT5
 #include <QApplication>
#endif

namespace {

int numDmhmInits=0;
bool dmhmInitializedMpi = false;
#ifdef HAVE_QT5
bool dmhmInitializedQt = false;
bool dmhmOpenedWindow = false;
QCoreApplication* coreApp;
#endif

dmhm::MpiArgs* args = 0;

#ifndef RELEASE
std::stack<std::string> callStack;
#endif

int oversample=4;
float midcomputeTolFloat=1e-5;
float compressionTolFloat=1e-5;
double midcomputeTolDouble=1e-16;
double compressionTolDouble=1e-16;

#ifdef MEMORY_INFO
double memoryUsage=0;
double peakMemoryUsage=0;
#endif
}

namespace dmhm {

#ifdef HAVE_QT5
void OpenedWindow()
{ ::dmhmOpenedWindow = true; }
#endif

bool Initialized()
{ return ::numDmhmInits > 0; }

void Initialize( int& argc, char**& argv )
{
    if( ::numDmhmInits > 0 )
    {
        ++::numDmhmInits;
        return;
    }

    ::args = new MpiArgs( argc, argv );

    ::numDmhmInits = 1;
    if( !mpi::Initialized() )
    {
        if( mpi::Finalized() )
        {
            throw std::logic_error
            ("Cannot initialize DMHM after finalizing MPI");
        }
        mpi::Initialize( argc, argv );
        ::dmhmInitializedMpi = true;
    }

#ifdef HAVE_QT5
    ::coreApp = QCoreApplication::instance();
    if( ::coreApp == 0 )
    {
        ::coreApp = new QApplication( argc, argv );
        ::dmhmInitializedQt = true;
    }
#endif

    // Seed the random number generators
    const int commRank = mpi::CommRank( mpi::COMM_WORLD );
    const int commSize = mpi::CommSize( mpi::COMM_WORLD );
    UInt64 seed;
    seed.d[0] = 17U;
    seed.d[1] = 21U;
    SeedSerialLcg( seed );
    SeedParallelLcg( commRank, commSize, seed );
}

void Finalize()
{
#ifndef RELEASE
    CallStackEntry entry("Finalize");
#endif
    if( ::numDmhmInits <= 0 )
        throw std::logic_error("Finalized DMHM more than initialized");
    --::numDmhmInits;

    if( mpi::Finalized() )
        std::cerr << "Warning: MPI was finalized before DMHM" << std::endl;
    if( ::numDmhmInits == 0 )
    {
        delete ::args;
        ::args = 0;

        if( ::dmhmInitializedMpi )
            mpi::Finalize();
    }

#ifdef HAVE_QT5
    if( ::dmhmInitializedQt )
    {
        if( ::dmhmOpenedWindow )
            ::coreApp->exec();
        else
            ::coreApp->exit();
        delete ::coreApp;
    }
#endif
}
MpiArgs& GetArgs()
{
    if( args == 0 )
        throw std::runtime_error("No available instance of MpiArgs");
    return *::args;
}

#ifndef RELEASE
void PushCallStack( const std::string s )
{ ::callStack.push(s); }

void PopCallStack()
{ ::callStack.pop(); }

void DumpCallStack()
{
    std::cout << "Dumping call stack of size " << ::callStack.size()
              << std::endl;
    std::ostringstream msg;
    while( ! ::callStack.empty() )
    {
        msg << "[" << ::callStack.size() << "]: " << ::callStack.top() << "\n";
        ::callStack.pop();
    }
    std::cerr << msg.str() << std::endl;
}
#endif

int Oversample()
{ return ::oversample; }

void SetOversample( int oversample )
{ ::oversample = oversample; }

template<typename Real>
Real CompressionTolerance();

template<>
float CompressionTolerance<float>()
{ return ::compressionTolFloat; }

template<>
double CompressionTolerance<double>()
{ return ::compressionTolDouble; }

template<typename Real>
void SetCompressionTolerance( Real relTol );

template<>
void SetCompressionTolerance<float>( float relTol )
{ ::compressionTolFloat = relTol; }

template<>
void SetCompressionTolerance<double>( double relTol )
{ ::compressionTolDouble = relTol; }

template<typename Real>
Real MidcomputeTolerance();

template<>
float MidcomputeTolerance<float>()
{ return ::midcomputeTolFloat; }

template<>
double MidcomputeTolerance<double>()
{ return ::midcomputeTolDouble; }

template<typename Real>
void SetMidcomputeTolerance( Real tolerance );

template<>
void SetMidcomputeTolerance<float>( float tolerance )
{ ::midcomputeTolFloat = tolerance; }

template<>
void SetMidcomputeTolerance<double>( double tolerance )
{ ::midcomputeTolDouble = tolerance; }

#ifdef MEMORY_INFO
void ResetMemoryCount()
{   ::memoryUsage = 0;
    ::peakMemoryUsage = 0;
}

void AddToMemoryCount( double size )
{   ::memoryUsage += size;
    if( ::memoryUsage > ::peakMemoryUsage )
        ::peakMemoryUsage = ::memoryUsage;
}

double MemoryUsage()
{ return ::memoryUsage; }

double PeakMemoryUsage()
{ return ::peakMemoryUsage; }
#endif

} // namespace dhmhm
