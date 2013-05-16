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
    mutable unsigned currentIndex_;
    mutable typename std::map<T1,T2*>::iterator it_;
    mutable std::map<T1,T2*> baseMap_;
public:
    int Size() const { return baseMap_.size(); }
    void ResetIterator() { currentIndex_=0; it_=baseMap_.begin(); }
    int CurrentIndex() const { return currentIndex_; }

    T1 CurrentKey() const
    { 
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentKey");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
        PopCallStack();
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();
        return it_->first; 
    }

    T2& Get( int key )
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Get");
#endif
        T2* value = baseMap_[key];
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
        T2* value = baseMap_[key];
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
        if( baseMap_[key] != 0 )
            throw std::logic_error("Overwrote previous value");
#endif
        baseMap_[key] = value;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    T2* CurrentEntry() 
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::CurrentEntry");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        T2* value = it_->second;
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
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        const T2* value = it_->second;
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
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        const T2* value = it_->second;
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
        typename std::map<T1,T2*>::iterator it = baseMap_.begin();
        for( int i=0; i<baseMap_.size(); ++i,++it)
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
        if( baseMap_.size() > 0 )
        {
            typename std::map<T1,T2*>::iterator it = baseMap_.begin();
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
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();
        ++it_;
        ++currentIndex_;
#ifndef RELEASE
        PopCallStack();
#endif
    }
    
    void Decrement()
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Decrement");
        if( currentIndex_ == 0 )
            throw std::logic_error("Traversed prior to beginning of map");
#endif
        --it_;
        --currentIndex_;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    void Erase( int key )
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::Erase");
#endif
        delete baseMap_[key];
        baseMap_[key] = 0;
        baseMap_.erase( key );
        it_ = baseMap_.begin();
        currentIndex_ = 0;
#ifndef RELEASE
        PopCallStack();
#endif
    }

    void EraseCurrentEntry()
    {
#ifndef RELEASE
        PushCallStack("MemoryMap::EraseCurrentEntry");
#endif
        delete it_->second;
        it_->second = 0;
        baseMap_.erase( it_++ );
#ifndef RELEASE
        PopCallStack();
#endif
    }
    
    void Clear()
    {
        typename std::map<T1,T2*>::iterator it; 
        for( it=baseMap_.begin(); it!=baseMap_.end(); it++ )
        {
            delete it->second;
            it->second = 0;
        }
        baseMap_.clear();
    }

    MemoryMap() : currentIndex_(0) { }
    ~MemoryMap() { Clear(); }
};  

} // namespace dmhm

#endif // ifndef DMHM_MEMORY_MAP_HPP
