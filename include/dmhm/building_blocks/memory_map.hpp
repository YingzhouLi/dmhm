/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_MEMORY_MAP_HPP
#define DMHM_MEMORY_MAP_HPP 1

namespace dmhm {

template<typename T1,typename T2> 
class MemoryMap 
{   
private:
    mutable unsigned _currentIndex;
    mutable typename std::map<T1,T2*>::iterator _it;
    mutable std::map<T1,T2*> _baseMap;
public:
    int Size() const { return _baseMap.size(); }
    void ResetIterator() { _currentIndex=0; _it=_baseMap.begin(); }
    int CurrentIndex() const { return _currentIndex; }

    T1 CurrentKey() const
    { 
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentKey");
        if( _currentIndex >= _baseMap.size() )
            throw std::logic_error("Traversed past end of map");
        PopCallStack();
#endif
        if( _currentIndex == 0 )
            _it = _baseMap.begin();
        return _it->first; 
    }

    T2& Get( int key )
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Get");
#endif
        T2* value = _baseMap[key];
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to access with invalid key.");
        PopCallStack();
#endif
        return *value;
    }
    
    const T2& Get( int key ) const
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Get");
#endif
        T2* value = _baseMap[key];
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to access with invalid key.");
        PopCallStack();
#endif
        return *value;
    }


    void Set( int key, T2* value )
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Set");
        if( _baseMap[key] != 0 )
            throw std::logic_error("Overwrote previous value");
#endif
        _baseMap[key] = value;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    T2* CurrentEntry() 
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentEntry");
        if( _currentIndex >= _baseMap.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( _currentIndex == 0 )
            _it = _baseMap.begin();

        T2* value = _it->second;
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to return null pointer.");
        PopCallStack();
#endif
        return value;
    }

    const T2* CurrentEntry() const
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentEntry");
        if( _currentIndex >= _baseMap.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( _currentIndex == 0 )
            _it = _baseMap.begin();

        const T2* value = _it->second;
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to return null pointer.");
        PopCallStack();
#endif
        return value;
    }
    //CurrentWidth and TotalWidth only supported when T2 has function Width
    const int CurrentWidth() const
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentWidth");
        if( _currentIndex >= _baseMap.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( _currentIndex == 0 )
            _it = _baseMap.begin();

        const T2* value = _it->second;
#ifndef RELEASE
        PopCallStack();
#endif
        return value->Width();
    }

    const int TotalWidth() const
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::TotalWidth");
#endif
        int width=0;
        typename std::map<T1,T2*>::iterator it = _baseMap.begin();
        for( int i=0; i<_baseMap.size(); ++i,++it)
            width += it->second->Width();
#ifndef RELEASE
        PopCallStack();
#endif
        return width;
    }

    const int FirstWidth() const
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::FirstWidth");
#endif
        int width=0;
        if( _baseMap.size() > 0 )
        {
            typename std::map<T1,T2*>::iterator it = _baseMap.begin();
            width = it->second->Width();
        }
#ifndef RELEASE
        PopCallStack();
#endif
        return width;
    }

    void Increment()
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Increment");
        if( _currentIndex >= _baseMap.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( _currentIndex == 0 )
            _it = _baseMap.begin();
        ++_it;
        ++_currentIndex;
#ifndef RELEASE
        PopCallStack();
#endif
    }
    
    void Decrement()
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Decrement");
        if( _currentIndex == 0 )
            throw std::logic_error("Traversed prior to beginning of map");
#endif
        --_it;
        --_currentIndex;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    void Erase( int key )
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Erase");
#endif
        delete _baseMap[key];
        _baseMap[key] = 0;
        _baseMap.erase( key );
        _it = _baseMap.begin();
        _currentIndex = 0;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    void EraseCurrentEntry()
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::EraseCurrentEntry");
#endif
        delete _it->second;
        _it->second = 0;
        _baseMap.erase( _it++ );
#ifndef RELEASE
        PopCallStack();
#endif
    }
    
    void Clear()
    {
        typename std::map<T1,T2*>::iterator it; 
        for( it=_baseMap.begin(); it!=_baseMap.end(); it++ )
        {
            delete it->second;
            it->second = 0;
        }
        _baseMap.clear();
    }

    MemoryMap() : _currentIndex(0) { }
    ~MemoryMap() { Clear(); }
};  

} // namespace dmhm

#endif // ifndef DMHM_MEMORY_MAP_HPP
