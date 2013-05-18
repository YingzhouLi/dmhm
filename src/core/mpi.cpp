/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

namespace {

void SafeMpi( int mpiError )
{
#ifndef RELEASE
    if( mpiError != MPI_SUCCESS )
    {
        char errorString[200];
        int lengthOfErrorString;
        MPI_Error_string( mpiError, errorString, &lengthOfErrorString );
        throw std::logic_error( errorString );
    }
#endif
}

} // anonymous namespace

namespace dmhm {
namespace mpi {

// NOTE: This data structure is inspired by Justin Holewinski's blog post at
//       http://jholewinski.org/blog/the-beauty-of-c-templates/
template<typename T>
struct MpiMap
{
    Datatype type;
    MpiMap();
};

template<>
MpiMap<byte>::MpiMap() : type(MPI_UNSIGNED_CHAR) { }
template<>
MpiMap<int>::MpiMap() : type(MPI_INT) { }
template<>
MpiMap<float>::MpiMap() : type(MPI_FLOAT) { }
template<>
MpiMap<double>::MpiMap() : type(MPI_DOUBLE) { }
template<>
MpiMap<std::complex<float> >::MpiMap() : type(MPI_COMPLEX) { }
template<>
MpiMap<std::complex<double> >::MpiMap() : type(MPI_DOUBLE_COMPLEX) { }

//
// Environment routines
//

void Initialize( int& argc, char**& argv )
{ MPI_Init( &argc, &argv ); }

void Finalize()
{ MPI_Finalize(); }

bool Initialized()
{
    int initialized;
    MPI_Initialized( &initialized );
    return initialized;
}

bool Finalized()
{
    int finalized;
    MPI_Finalized( &finalized );
    return finalized;
}

double Time()
{ return MPI_Wtime(); }

//
// Communicator manipulation routines
//

int CommRank( Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::CommRank");
#endif
    int rank;
    SafeMpi( MPI_Comm_rank( comm, &rank ) );
    return rank;
}

int CommSize( Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::CommSize");
#endif
    int size;
    SafeMpi( MPI_Comm_size( comm, &size ) );
    return size;
}


void CommDup( Comm comm, Comm& commDup )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::CommDup");
#endif
    SafeMpi( MPI_Comm_dup( comm, &commDup ) );
}

void CommFree( Comm& comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::CommFree");
#endif
    SafeMpi( MPI_Comm_free( &comm ) );
}

void CommSplit( Comm comm, int color, int key, Comm& newComm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::CommSplit");
#endif
    SafeMpi( MPI_Comm_split( comm, color, key, &newComm ) );
}

//
// Point-to-point communication
//

template<typename R>
void Send( const R* buf, int count, int to, int tag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Send");
#endif
    MpiMap<R> map;
    SafeMpi( MPI_Send( const_cast<R*>(buf), count, map.type, to, tag, comm ) );
}

template<typename R>
void Send( const std::complex<R>* buf, int count, int to, int tag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Send");
#endif
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<R>*>(buf), 2*count, map.type, to, tag, comm )
    );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<R>*>(buf), count, map.type, to, tag, comm )
    );
#endif
}

template void Send( const byte* buf, int count, int to, int tag, Comm comm );
template void Send( const int* buf, int count, int to, int tag, Comm comm );
template void Send( const float* buf, int count, int to, int tag, Comm comm );
template void Send( const double* buf, int count, int to, int tag, Comm comm );
template void Send( const std::complex<float>* buf, int count, int to, int tag, Comm comm );
template void Send( const std::complex<double>* buf, int count, int to, int tag, Comm comm );

template<typename R>
void ISend
( const R* buf, int count, int to, int tag, Comm comm, Request& request )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::ISend");
#endif
    MpiMap<R> map;
    SafeMpi(
        MPI_Isend
        ( const_cast<R*>(buf), count, map.type, to, tag, comm, &request )
    );
}

template<typename R>
void ISend
( const std::complex<R>* buf, int count, int to, int tag, Comm comm,
  Request& request )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::ISend");
#endif
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi(
        MPI_Isend
        ( const_cast<std::complex<R>*>(buf), 2*count, map.type, to, tag, comm,
          &request )
    );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi(
        MPI_Isend
        ( const_cast<std::complex<R>*>(buf), count, map.type, to, tag, comm,
          &request )
    );
#endif
}

template void ISend( const byte* buf, int count, int to, int tag, Comm comm, Request& request );
template void ISend( const int* buf, int count, int to, int tag, Comm comm, Request& request );
template void ISend( const float* buf, int count, int to, int tag, Comm comm, Request& request );
template void ISend( const double* buf, int count, int to, int tag, Comm comm, Request& request );
template void ISend( const std::complex<float>* buf, int count, int to, int tag, Comm comm, Request& request );
template void ISend( const std::complex<double>* buf, int count, int to, int tag, Comm comm, Request& request );

template<typename R>
void Recv( R* buf, int count, int from, int tag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Recv");
#endif
    MpiMap<R> map;
    Status status;
    SafeMpi( MPI_Recv( buf, count, map.type, from, tag, comm, &status ) );
}

template<typename R>
void Recv( std::complex<R>* buf, int count, int from, int tag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Recv");
#endif
    Status status;
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi( MPI_Recv( buf, 2*count, map.type, from, tag, comm, &status ) );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi( MPI_Recv( buf, count, map.type, from, tag, comm, &status ) );
#endif
}

template void Recv( byte* buf, int count, int from, int tag, Comm comm );
template void Recv( int* buf, int count, int from, int tag, Comm comm );
template void Recv( float* buf, int count, int from, int tag, Comm comm );
template void Recv( double* buf, int count, int from, int tag, Comm comm );
template void Recv( std::complex<float>* buf, int count, int from, int tag, Comm comm );
template void Recv( std::complex<double>* buf, int count, int from, int tag, Comm comm );

template<typename R>
void IRecv( R* buf, int count, int from, int tag, Comm comm, Request& request )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::IRecv");
#endif
    MpiMap<R> map;
    SafeMpi( MPI_Irecv( buf, count, map.type, from, tag, comm, &request ) );
}

template<typename R>
void IRecv
( std::complex<R>* buf, int count, int from, int tag, Comm comm, Request& request )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::IRecv");
#endif
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi( MPI_Irecv( buf, 2*count, map.type, from, tag, comm, &request ) );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi( MPI_Irecv( buf, count, map.type, from, tag, comm, &request ) );
#endif
}

template void IRecv( byte* buf, int count, int from, int tag, Comm comm, Request& request );
template void IRecv( int* buf, int count, int from, int tag, Comm comm, Request& request );
template void IRecv( float* buf, int count, int from, int tag, Comm comm, Request& request );
template void IRecv( double* buf, int count, int from, int tag, Comm comm, Request& request );
template void IRecv( std::complex<float>* buf, int count, int from, int tag, Comm comm, Request& request );
template void IRecv( std::complex<double>* buf, int count, int from, int tag, Comm comm, Request& request );

template<typename R>
void SendRecv
( const R* sbuf, int sc, int to,   int stag,
        R* rbuf, int rc, int from, int rtag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::SendRecv");
#endif
    Status status;
    MpiMap<R> map;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<R*>(sbuf), sc, map.type, to,   stag,
          rbuf,                 rc, map.type, from, rtag, comm, &status )
    );
}

template<typename R>
void SendRecv
( const std::complex<R>* sbuf, int sc, int to,   int stag,
        std::complex<R>* rbuf, int rc, int from, int rtag, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::SendRecv");
#endif
    Status status;
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<std::complex<R>*>(sbuf), 2*sc, map.type, to,   stag,
          rbuf,                          2*rc, map.type, from, rtag,
          comm, &status )
    );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<std::complex<R>*>(sbuf), sc, map.type, to,   stag,
          rbuf,                          rc, map.type, from, rtag,
          comm, &status )
    );
#endif
}

template void SendRecv
( const byte* sbuf, int sc, int to, int stag,
        byte* rbuf, int rc, int from, int rtag, Comm comm );
template void SendRecv
( const int* sbuf, int sc, int to, int stag,
        int* rbuf, int rc, int from, int rtag, Comm comm );
template void SendRecv
( const float* sbuf, int sc, int to, int stag,
        float* rbuf, int rc, int from, int rtag, Comm comm );
template void SendRecv
( const double* sbuf, int sc, int to, int stag,
        double* rbuf, int rc, int from, int rtag, Comm comm );
template void SendRecv
( const std::complex<float>* sbuf, int sc, int to, int stag,
        std::complex<float>* rbuf, int rc, int from, int rtag, Comm comm );
template void SendRecv
( const std::complex<double>* sbuf, int sc, int to, int stag,
        std::complex<double>* rbuf, int rc, int from, int rtag, Comm comm );

// Ensure that the request finishes before continuing
void Wait( Request& request )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Wait");
#endif
    Status status;
    SafeMpi( MPI_Wait( &request, &status ) );
}

// Ensure that the request finishes before continuing
void Wait( Request& request, Status& status )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Wait");
#endif
    SafeMpi( MPI_Wait( &request, &status ) );
}

// Ensure that several requests finish before continuing
void WaitAll( int numRequests, Request* requests )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::WaitAll");
#endif
    std::vector<Status> statuses( numRequests );
    SafeMpi( MPI_Waitall( numRequests, requests, &statuses[0] ) );
}

// Ensure that several requests finish before continuing
void WaitAll( int numRequests, Request* requests, Status* statuses )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::WaitAll");
#endif
    SafeMpi( MPI_Waitall( numRequests, requests, statuses ) );
}

//
// Collective communication
//

// Wait until every process in comm reaches this statement
void Barrier( MPI_Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Barrier");
#endif
    SafeMpi( MPI_Barrier( comm ) );
}

template<typename R>
void Broadcast( R* buf, int count, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Broadcast");
#endif
    MpiMap<R> map;
    SafeMpi( MPI_Bcast( buf, count, map.type, root, comm ) );
}

template<typename R>
void Broadcast( std::complex<R>* buf, int count, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Broadcast");
#endif
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi( MPI_Bcast( buf, 2*count, map.type, root, comm ) );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi( MPI_Bcast( buf, count, map.type, root, comm ) );
#endif
}

template void Broadcast( byte* buf, int count, int root, Comm comm );
template void Broadcast( int* buf, int count, int root, Comm comm );
template void Broadcast( float* buf, int count, int root, Comm comm );
template void Broadcast( double* buf, int count, int root, Comm comm );
template void Broadcast( std::complex<float>* buf, int count, int root, Comm comm );
template void Broadcast( std::complex<double>* buf, int count, int root, Comm comm );

template<typename R>
void AllGather
( const R* sbuf, int sc,
        R* rbuf, int rc, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllGather");
#endif
#ifdef USE_BYTE_ALLGATHERS
    SafeMpi(
        MPI_Allgather
        ( const_cast<R*>(sbuf), sizeof(R)*sc, MPI_UNSIGNED_CHAR,
          rbuf,                 sizeof(R)*rc, MPI_UNSIGNED_CHAR, comm )
    );
#else
    MpiMap<R> map;
    SafeMpi(
        MPI_Allgather
        ( const_cast<R*>(sbuf), sc, map.type,
          rbuf,                 rc, map.type, comm )
    );
#endif
}

template<typename R>
void AllGather
( const std::complex<R>* sbuf, int sc,
        std::complex<R>* rbuf, int rc, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllGather");
#endif
#ifdef USE_BYTE_ALLGATHERS
    SafeMpi(
        MPI_Allgather
        ( const_cast<std::complex<R>*>(sbuf), 2*sizeof(R)*sc, MPI_UNSIGNED_CHAR,
          rbuf,                          2*sizeof(R)*rc, MPI_UNSIGNED_CHAR,
          comm )
    );
#else
 #ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi(
        MPI_Allgather
        ( const_cast<std::complex<R>*>(sbuf), 2*sc, map.type,
          rbuf,                          2*rc, map.type, comm )
    );
 #else
    MpiMap<std::complex<R> > map;
    SafeMpi(
        MPI_Allgather
        ( const_cast<std::complex<R>*>(sbuf), sc, map.type,
          rbuf,                          rc, map.type, comm )
    );
 #endif
#endif
}

template void AllGather( const byte* sbuf, int sc, byte* rbuf, int rc, Comm comm );
template void AllGather( const int* sbuf, int sc, int* rbuf, int rc, Comm comm );
template void AllGather( const float* sbuf, int sc, float* rbuf, int rc, Comm comm );
template void AllGather( const double* sbuf, int sc, double* rbuf, int rc, Comm comm );
template void AllGather( const std::complex<float>* sbuf, int sc, std::complex<float>* rbuf, int rc, Comm comm );
template void AllGather( const std::complex<double>* sbuf, int sc, std::complex<double>* rbuf, int rc, Comm comm );

template<typename T>
void Reduce
( const T* sbuf, T* rbuf, int count, Op op, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Reduce");
#endif
    MpiMap<T> map;
    if( count != 0 )
    {
        SafeMpi(
            MPI_Reduce
            ( const_cast<T*>(sbuf), rbuf, count, map.type, op, root, comm )
        );
    }
}

template<typename R>
void Reduce
( const std::complex<R>* sbuf,
        std::complex<R>* rbuf, int count, Op op, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Reduce");
#endif
    if( count != 0 )
    {
#ifdef AVOID_COMPLEX_MPI
        if( op == SUM )
        {
            MpiMap<R> map;
            SafeMpi(
                MPI_Reduce
                ( const_cast<std::complex<R>*>(sbuf),
                  rbuf, 2*count, map.type, op, root, comm )
            );
        }
        else
        {
            MpiMap<std::complex<R> > map;
            SafeMpi(
                MPI_Reduce
                ( const_cast<std::complex<R>*>(sbuf),
                  rbuf, count, map.type, op, root, comm )
            );
        }
#else
        MpiMap<std::complex<R> > map;
        SafeMpi(
            MPI_Reduce
            ( const_cast<std::complex<R>*>(sbuf),
              rbuf, count, map.type, op, root, comm )
        );
#endif
    }
}

template void Reduce( const byte* sbuf, byte* rbuf, int count, Op op, int root, Comm comm );
template void Reduce( const int* sbuf, int* rbuf, int count, Op op, int root, Comm comm );
template void Reduce( const float* sbuf, float* rbuf, int count, Op op, int root, Comm comm );
template void Reduce( const double* sbuf, double* rbuf, int count, Op op, int root, Comm comm );
template void Reduce( const std::complex<float>* sbuf, std::complex<float>* rbuf, int count, Op op, int root, Comm comm );
template void Reduce( const std::complex<double>* sbuf, std::complex<double>* rbuf, int count, Op op, int root, Comm comm );

template<typename T>
void Reduce( T* buf, int count, Op op, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Reduce");
#endif
    MpiMap<T> map;
    if( count != 0 )
    {
        const int commRank = CommRank( comm );
        if( commRank == root )
        {
#ifdef HAVE_MPI_IN_PLACE
            SafeMpi(
                MPI_Reduce( MPI_IN_PLACE, buf, count, map.type, op, root, comm )
            );
#else
            std::vector<T> sendBuf( count );
            MemCopy( &sendBuf[0], buf, count );
            SafeMpi(
                MPI_Reduce( &sendBuf[0], buf, count, map.type, op, root, comm )
            );
#endif
        }
        else
            SafeMpi( MPI_Reduce( buf, 0, count, map.type, op, root, comm ) );
    }
}

template<typename R>
void Reduce( std::complex<R>* buf, int count, Op op, int root, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::Reduce");
#endif
    if( count != 0 )
    {
        const int commRank = CommRank( comm );
#ifdef AVOID_COMPLEX_MPI
        if( op == SUM )
        {
            MpiMap<R> map;
            if( commRank == root )
            {
# ifdef HAVE_MPI_IN_PLACE
                SafeMpi(
                    MPI_Reduce
                    ( MPI_IN_PLACE, buf, 2*count, map.type, op, root, comm )
                );
# else
                std::vector<std::complex<R> > sendBuf( count );
                MemCopy( &sendBuf[0], buf, count );
                SafeMpi(
                    MPI_Reduce
                    ( &sendBuf[0], buf, 2*count, map.type, op, root, comm )
                );
# endif
            }
            else
                SafeMpi(
                    MPI_Reduce( buf, 0, 2*count, map.type, op, root, comm )
                );
        }
        else
        {
            MpiMap<std::complex<R> > map;
            if( commRank == root )
            {
# ifdef HAVE_MPI_IN_PLACE
                SafeMpi(
                    MPI_Reduce
                    ( MPI_IN_PLACE, buf, count, map.type, op, root, comm )
                );
# else
                std::vector<std::complex<R> > sendBuf( count );
                MemCopy( &sendBuf[0], buf, count );
                SafeMpi(
                    MPI_Reduce
                    ( &sendBuf[0], buf, count, map.type, op, root, comm )
                );
# endif
            }
            else
                SafeMpi(
                    MPI_Reduce( buf, 0, count, map.type, op, root, comm )
                );
        }
#else
        MpiMap<std::complex<R> > map;
        if( commRank == root )
        {
# ifdef HAVE_MPI_IN_PLACE
            SafeMpi(
                MPI_Reduce( MPI_IN_PLACE, buf, count, map.type, op, root, comm )
            );
# else
            std::vector<std::complex<R> > sendBuf( count );
            MemCopy( &sendBuf[0], buf, count );
            SafeMpi(
                MPI_Reduce( &sendBuf[0], buf, count, map.type, op, root, comm )
            );
# endif
        }
        else
            SafeMpi( MPI_Reduce( buf, 0, count, map.type, op, root, comm ) );
#endif
    }
}

template void Reduce( byte* buf, int count, Op op, int root, Comm comm );
template void Reduce( int* buf, int count, Op op, int root, Comm comm );
template void Reduce( float* buf, int count, Op op, int root, Comm comm );
template void Reduce( double* buf, int count, Op op, int root, Comm comm );
template void Reduce( std::complex<float>* buf, int count, Op op, int root, Comm comm );
template void Reduce( std::complex<double>* buf, int count, Op op, int root, Comm comm );

template<typename T>
void AllReduce( const T* sbuf, T* rbuf, int count, Op op, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllReduce");
#endif
    MpiMap<T> map;
    if( count != 0 )
    {
        SafeMpi(
            MPI_Allreduce
            ( const_cast<T*>(sbuf), rbuf, count, map.type, op, comm )
        );
    }
}

template<typename R>
void AllReduce
( const std::complex<R>* sbuf, std::complex<R>* rbuf, 
  int count, Op op, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllReduce");
#endif
    if( count != 0 )
    {
#ifdef AVOID_COMPLEX_MPI
        if( op == SUM )
        {
            MpiMap<R> map;
            SafeMpi(
                MPI_Allreduce
                ( const_cast<std::complex<R>*>(sbuf),
                  rbuf, 2*count, map.type, op, comm )
            );
        }
        else
        {
            MpiMap<std::complex<R> > map;
            SafeMpi(
                MPI_Allreduce
                ( const_cast<std::complex<R>*>(sbuf),
                  rbuf, count, map.type, op, comm )
            );
        }
#else
        MpiMap<std::complex<R> > map;
        SafeMpi(
            MPI_Allreduce
            ( const_cast<std::complex<R>*>(sbuf),
              rbuf, count, map.type, op, comm )
        );
#endif
    }
}

template void AllReduce( const byte* sbuf, byte* rbuf, int count, Op op, Comm comm );
template void AllReduce( const int* sbuf, int* rbuf, int count, Op op, Comm comm );
template void AllReduce( const float* sbuf, float* rbuf, int count, Op op, Comm comm );
template void AllReduce( const double* sbuf, double* rbuf, int count, Op op, Comm comm );
template void AllReduce( const std::complex<float>* sbuf, std::complex<float>* rbuf, int count, Op op, Comm comm );
template void AllReduce( const std::complex<double>* sbuf, std::complex<double>* rbuf, int count, Op op, Comm comm );

template<typename R>
void AllToAll
( const R* sbuf, int sc,
        R* rbuf, int rc, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllToAll");
#endif
    MpiMap<R> map;
    SafeMpi(
        MPI_Alltoall
        ( const_cast<R*>(sbuf), sc, map.type,
          rbuf,                 rc, map.type, comm )
    );
}

template<typename R>
void AllToAll
( const std::complex<R>* sbuf, int sc,
        std::complex<R>* rbuf, int rc, Comm comm )
{
#ifndef RELEASE
    CallStackEntry entry("mpi::AllToAll");
#endif
#ifdef AVOID_COMPLEX_MPI
    MpiMap<R> map;
    SafeMpi(
        MPI_Alltoall
        ( const_cast<std::complex<R>*>(sbuf), 2*sc, map.type,
          rbuf,                          2*rc, map.type, comm )
    );
#else
    MpiMap<std::complex<R> > map;
    SafeMpi(
        MPI_Alltoall
        ( const_cast<std::complex<R>*>(sbuf), sc, map.type,
          rbuf,                          rc, map.type, comm )
    );
#endif
}

template void AllToAll
( const byte* sbuf, int sc,
        byte* rbuf, int rc, Comm comm );
template void AllToAll
( const int* sbuf, int sc,
        int* rbuf, int rc, Comm comm );
template void AllToAll
( const float* sbuf, int sc,
        float* rbuf, int rc, Comm comm );
template void AllToAll
( const double* sbuf, int sc,
        double* rbuf, int rc, Comm comm );
template void AllToAll
( const std::complex<float>* sbuf, int sc,
        std::complex<float>* rbuf, int rc, Comm comm );
template void AllToAll
( const std::complex<double>* sbuf, int sc,
        std::complex<double>* rbuf, int rc, Comm comm );

} // namespace mpi
} // namespace dmhm
