/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_TIMER_HPP
#define DMHM_TIMER_HPP 1

namespace dmhm {

class Timer
{
public:
    void Start( int key );
    double Stop( int key );
    double GetTime( int key );

    void Clear();
    void Clear( int key );
private:
    std::map<int,double> _startTimes, _times;
    std::map<int,bool> _running;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

inline void
Timer::Start( int key )
{
#ifndef RELEASE
    PushCallStack("Timer::Start");
#endif
    std::map<int,bool>::iterator it;
    it = _running.find( key );
    if( it == _running.end() )
    {
        _running[key] = true;
        _startTimes[key] = mpi::WallTime();
    }
    else
    {
        if( _running[key] )
            throw std::logic_error
            ("Restarted timer with same key without stopping");
        _running[key] = true;
        _startTimes[key] = mpi::WallTime();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline double
Timer::Stop( int key )
{
#ifndef RELEASE
    PushCallStack("Timer::Stop");
#endif
    double pairTime = 0;

    std::map<int,bool>::iterator runningIt;
    runningIt = _running.find( key );
    if( runningIt == _running.end() || !_running[key] )
        throw std::logic_error("Stopped a timer that was not running");
    else
    {
        pairTime = mpi::WallTime() - _startTimes[key];

        std::map<int,double>::iterator timeIt;
        timeIt = _times.find( key );
        if( timeIt == _times.end() )
            _times[key] = pairTime;
        else
            _times[key] += pairTime;

        _running[key] = false;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return pairTime;
}

inline double
Timer::GetTime( int key )
{
#ifndef RELEASE
    PushCallStack("Timer::GetTime");
#endif
    double time = 0;

    std::map<int,double>::iterator it;
    it = _times.find( key );
    if( it == _times.end() )
        time = 0;
    else
        time = _times[key];
#ifndef RELEASE
    PopCallStack();
#endif
    return time;
}

inline void Timer::Clear()
{
    _running.clear();
    _startTimes.clear();
    _times.clear();
}

inline void Timer::Clear( int key )
{
    _running.erase( key );
    _startTimes.erase( key );
    _times.erase( key );
}

} // namespace dmhm

#endif // ifndef DMHM_TIMER_HPP
