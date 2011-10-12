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

} // namespace dmhm

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

inline void
dmhm::Timer::Start( int key )
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
dmhm::Timer::Stop( int key )
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
dmhm::Timer::GetTime( int key )
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

inline void dmhm::Timer::Clear()
{
    _running.clear();
    _startTimes.clear();
    _times.clear();
}

inline void dmhm::Timer::Clear( int key )
{
    _running.erase( key );
    _startTimes.erase( key );
    _times.erase( key );
}

#endif // DMHM_TIMER_HPP
