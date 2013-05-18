/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"
using namespace dmhm;

int
main( int argc, char* argv[] )
{
    try
    {
        MemoryMap<int,Dense<double> > memoryMap;

        for( int i=0; i<10000; ++i )
            memoryMap.Set(400 - 3*i,new Dense<double>(i%4,i%4));

        int numEntries = memoryMap.Size();
        std::cout << "size of memory map: " << numEntries << std::endl;
        memoryMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry,memoryMap.Increment() )
        {
            const int currentIndex = memoryMap.CurrentIndex();
            const Dense<double>& D = *memoryMap.CurrentEntry(); 
            std::cout << "Index " << currentIndex << ": " 
                      << D.Height() << " x " << D.Width() << "\n"; 
        }
        std::cout << std::endl;

        // Loop over again, and delete the third entry. At the end of each
        // loop, print the total size
        memoryMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            const int currentIndex = memoryMap.CurrentIndex();
            const Dense<double>& D = *memoryMap.CurrentEntry();
            std::cout << "Index " << currentIndex << ": "
                      << D.Height() << " x " << D.Width() << "\n";
            if( entry%3 == 2 )
            {
                memoryMap.EraseCurrentEntry();
                std::cout << "Erased third entry, new size is: " 
                          << memoryMap.Size() << "\n";
            }
            else
                memoryMap.Increment();
        }
        std::cout << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        DumpCallStack();
#endif
    }

    return 0;
}
