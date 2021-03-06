/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/

namespace dmhm {

template<typename Scalar>
void
DistHMat3d<Scalar>::MultiplyHMatParallelQR
( const Vector<int>& numQRs,
  const Vector<Dense<Scalar>*>& Xs,
  const Vector<int>& XOffsets,
        Vector<int>& halfHeights,
  const Vector<int>& halfHeightOffsets,
        Vector<Scalar>& qrBuffer,
  const Vector<int>& qrOffsets,
        Vector<Scalar>& tauBuffer,
  const Vector<int>& tauOffsets,
        Vector<Scalar>& qrWork ) const
{
#ifndef RELEASE
    CallStackEntry entry("DistHMat3d::MultiplyHMatParallelQR");
#endif
    const int globalRank = mpi::CommRank( teams_->Team(0) );
    const int numTeamLevels = teams_->NumLevels();
    const int numSteps = numTeamLevels-1;

    int passes = 0;
    for( int step=0; step<numSteps; ++step )
    {
        mpi::Comm team = teams_->Team( (numSteps-1)-step );
        const int teamSize = mpi::CommSize( team );
        const int teamRank = mpi::CommRank( team );

        // Flip the first bit of our rank in this team to get our partner,
        // and then check if our bit is 0 to see if we're the root
        const unsigned firstPartner = teamRank ^ (1u << passes);
        const unsigned firstRoot = !(teamRank & (1u << passes));

        // Compute the total message size for this step:
        //   The # of bytes for an int and (r*r+r)/2 scalars.
        int msgSize = 0;
        for( int l=0; l<numSteps-step; ++l )
        {
            const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
            for( int k=0; k<numQRs[l]; ++k )
            {
                const int r = XLevel[k]->Width();
                msgSize += sizeof(int) + (r*r+r)/2*sizeof(Scalar);
            }
        }
        Vector<byte> sendBuffer( msgSize ), recvBuffer( msgSize );

        // Pack the messages for the firstPartner
        byte* sendHead = &sendBuffer[0];
        for( int l=0; l<numSteps-step; ++l )
        {
            mpi::Comm parentTeam = teams_->Team(l);
            const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));
            const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
            const Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
            int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

            int qrPieceOffset = 0;
            for( int k=0; k<numQRs[l]; ++k )
            {
                const Dense<Scalar>& X = *XLevel[k];
                const int r = X.Width();
                const int halfHeightOffset = (k*log2ParentTeamSize+passes)*2;

                int halfHeight = 0; // initialize so compiler's don't complain
                if( step == 0 )
                {
                    // The first iteration bootstraps the local height from X
                    halfHeight = std::min((int)X.Height(),r);
                    if( firstRoot )
                        halfHeightLevel[halfHeightOffset] = halfHeight;
                    else
                        halfHeightLevel[halfHeightOffset+1] = halfHeight;
                    Write( sendHead, halfHeight );

                    // Copy our R out of X
                    for( int j=0; j<r; ++j )
                    {
                        const int P = std::min(j+1,halfHeight);
                        Write( sendHead, X.LockedBuffer(0,j), P );
                    }
                }
                else
                {
                    // All but the first iteration read from halfHeights
                    const int sPrev = halfHeightLevel[halfHeightOffset-2];
                    const int tPrev = halfHeightLevel[halfHeightOffset-1];
                    const int halfHeight = std::min(sPrev+tPrev,r);
                    Write( sendHead, halfHeight );

                    // Copy our R out of the last qrBuffer step
                    int qrOffset = qrPieceOffset + (passes-1)*(r*r+r);
                    for( int j=0; j<r; ++j )
                    {
                        const int SPrev = std::min(j+1,sPrev);
                        const int TPrev = std::min(j+1,tPrev);
                        const int P = std::min(j+1,halfHeight);
                        Write( sendHead, &qrLevel[qrOffset], P );
                        qrOffset += SPrev + TPrev;
                    }
                }

                // Move past the padding for this send piece
                if( halfHeight < r )
                {
                    const int sendTrunc = r - halfHeight;
                    sendHead +=
                        (sendTrunc*sendTrunc+sendTrunc)/2*sizeof(Scalar);
                }

                // Advance our index to the next send piece
                qrPieceOffset += log2ParentTeamSize*(r*r+r);
            }
        }

        // Exchange with our first partner
        mpi::SendRecv
        ( &sendBuffer[0], msgSize, firstPartner, 0,
          &recvBuffer[0], msgSize, firstPartner, 0, team );

        throw std::logic_error("The QR method does not work at 3d");

        if( teamSize == 2 )
        {
            // Unpack the recv messages, perform the QR factorizations, and
            // store the local height for the next step if there is one
            const byte* recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                mpi::Comm parentTeam = teams_->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));

                bool rootOfNextStep = false;
                const bool haveAnotherComm = ( l+1 < numSteps-step );
                if( haveAnotherComm )
                    rootOfNextStep = !(globalRank & (1u<<(passes+1)));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset =
                        (k*log2ParentTeamSize+passes)*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + passes*(r*r+r);
                    if( step == 0 )
                    {
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from X
                                MemCopy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j), S );
                                qrOffset += S;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from X
                                MemCopy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j), T );
                                qrOffset += T;
                            }
                        }
                    }
                    else
                    {
                        int qrPrevOffset = qrPieceOffset + (passes-1)*(r*r+r);
                        const int sPrev = halfHeightLevel[halfHeightOffset-2];
                        const int tPrev = halfHeightLevel[halfHeightOffset-1];
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from the previous qrBuffer
                                MemCopy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  S );
                                qrOffset += S;
                                qrPrevOffset += SPrev + TPrev;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from the previous qrBuffer
                                MemCopy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  T );
                                qrOffset += T;
                                qrPrevOffset += SPrev + TPrev;
                            }
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc =
                        ( firstRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead +=
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t, &qrLevel[qrPieceOffset+passes*(r*r+r)],
                      &tauLevel[tauPieceOffset+(passes+1)*r], &qrWork[0] );

                    if( haveAnotherComm )
                    {
                        const int minDim = std::min(s+t,r);
                        if( rootOfNextStep )
                            halfHeightLevel[halfHeightOffset+2] = minDim;
                        else
                            halfHeightLevel[halfHeightOffset+3] = minDim;
                    }

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }
            ++passes;
        }
        else // teamSize >= 4, so this level splits 4 ways
        {
            const unsigned secondPartner = teamRank ^ (1u<<(passes+1));
            const bool secondRoot = !(teamRank & (1u<<(passes+1)));

            // Unpack the recv messages, perform the QR factorizations, and
            // pack the resulting R into the next step and into the next
            // send buffer in a single sweep.
            sendHead = &sendBuffer[0];
            const byte* recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                mpi::Comm parentTeam = teams_->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset =
                        (k*log2ParentTeamSize+passes)*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + passes*(r*r+r);
                    if( step == 0 )
                    {
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from X
                                MemCopy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j), S );
                                qrOffset += S;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from X
                                MemCopy
                                ( &qrLevel[qrOffset], X.LockedBuffer(0,j), T );
                                qrOffset += T;
                            }
                        }
                    }
                    else
                    {
                        int qrPrevOffset = qrPieceOffset + (passes-1)*(r*r+r);
                        const int sPrev = halfHeightLevel[halfHeightOffset-2];
                        const int tPrev = halfHeightLevel[halfHeightOffset-1];
                        if( firstRoot )
                        {
                            s = halfHeightLevel[halfHeightOffset];
                            t = Read<int>( recvHead );
                            halfHeightLevel[halfHeightOffset+1] = t;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from last qrBuffer
                                MemCopy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  S );
                                qrOffset += S;
                                qrPrevOffset += SPrev + TPrev;

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, T );
                                qrOffset += T;
                            }
                        }
                        else
                        {
                            s = Read<int>( recvHead );
                            t = halfHeightLevel[halfHeightOffset+1];
                            halfHeightLevel[halfHeightOffset] = s;
                            for( int j=0; j<r; ++j )
                            {
                                const int S = std::min(j+1,s);
                                const int T = std::min(j+1,t);
                                const int SPrev = std::min(j+1,sPrev);
                                const int TPrev = std::min(j+1,tPrev);

                                // Read column strip from recvBuffer
                                Read( &qrLevel[qrOffset], recvHead, S );
                                qrOffset += S;

                                // Read column strip from last qrBuffer
                                MemCopy
                                ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset],
                                  T );
                                qrOffset += T;
                                qrPrevOffset += SPrev + TPrev;
                            }
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc =
                        ( firstRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead +=
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t,
                      &qrLevel[qrPieceOffset+passes*(r*r+r)],
                      &tauLevel[tauPieceOffset+(passes+1)*r], &qrWork[0] );

                    const int minDim = std::min(s+t,r);
                    Write( sendHead, minDim );
                    if( secondRoot )
                        halfHeightLevel[halfHeightOffset+2] = minDim;
                    else
                        halfHeightLevel[halfHeightOffset+3] = minDim;
                    qrOffset = qrPieceOffset + passes*(r*r+r);
                    for( int j=0; j<r; ++j )
                    {
                        const int S = std::min(j+1,s);
                        const int T = std::min(j+1,t);
                        const int P = std::min(j+1,s+t);
                        Write( sendHead, &qrLevel[qrOffset], P );
                        qrOffset += S + T;
                    }
                    // Move past the send padding
                    const int sendTrunc = r-std::min(minDim,r);
                    sendHead +=
                        (sendTrunc*sendTrunc+sendTrunc)/2*sizeof(Scalar);

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }

            // Exchange with our second partner
            mpi::SendRecv
            ( &sendBuffer[0], msgSize, secondPartner, 0,
              &recvBuffer[0], msgSize, secondPartner, 0, team );

            // Unpack the recv messages and perform the QR factorizations, then
            // fill the local heights for the next step if there is one.
            recvHead = &recvBuffer[0];
            for( int l=0; l<numSteps-step; ++l )
            {
                mpi::Comm parentTeam = teams_->Team(l);
                const int log2ParentTeamSize = Log2(mpi::CommSize(parentTeam));

                bool rootOfNextStep = false;
                const bool haveAnotherComm = ( l+1 < numSteps-step );
                if( haveAnotherComm )
                    rootOfNextStep = !(globalRank & (1u<<(passes+2)));

                const Dense<Scalar>* const* XLevel = &Xs[XOffsets[l]];
                Scalar* qrLevel = &qrBuffer[qrOffsets[l]];
                Scalar* tauLevel = &tauBuffer[tauOffsets[l]];
                int* halfHeightLevel = &halfHeights[halfHeightOffsets[l]];

                int qrPieceOffset=0, tauPieceOffset=0;
                for( int k=0; k<numQRs[l]; ++k )
                {
                    const Dense<Scalar>& X = *XLevel[k];
                    const int r = X.Width();
                    const int halfHeightOffset
                        = (k*log2ParentTeamSize+(passes+1))*2;

                    int s, t;
                    int qrOffset = qrPieceOffset + (passes+1)*(r*r+r);
                    int qrPrevOffset = qrPieceOffset + passes*(r*r+r);
                    const int sPrev = halfHeightLevel[halfHeightOffset-2];
                    const int tPrev = halfHeightLevel[halfHeightOffset-1];
                    if( secondRoot )
                    {
                        s = halfHeightLevel[halfHeightOffset];
                        t = Read<int>( recvHead );
                        halfHeightLevel[halfHeightOffset+1] = t;
                        for( int j=0; j<r; ++j )
                        {
                            const int S = std::min(j+1,s);
                            const int T = std::min(j+1,t);
                            const int SPrev = std::min(j+1,sPrev);
                            const int TPrev = std::min(j+1,tPrev);

                            // Read column strip from last qrBuffer
                            MemCopy
                            ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset], S );
                            qrOffset += S;
                            qrPrevOffset += SPrev + TPrev;

                            // Read column strip from recvBuffer
                            Read( &qrBuffer[qrOffset], recvHead, T );
                            qrOffset += T;
                        }
                    }
                    else
                    {
                        s = Read<int>( recvHead );
                        t = halfHeightLevel[halfHeightOffset+1];
                        halfHeightLevel[halfHeightOffset] = s;
                        for( int j=0; j<r; ++j )
                        {
                            const int S = std::min(j+1,s);
                            const int T = std::min(j+1,t);
                            const int SPrev = std::min(j+1,sPrev);
                            const int TPrev = std::min(j+1,tPrev);

                            // Read column strip from recvBuffer
                            Read( &qrLevel[qrOffset], recvHead, S );
                            qrOffset += S;

                            // Read column strip from last qrBuffer
                            MemCopy
                            ( &qrLevel[qrOffset], &qrLevel[qrPrevOffset], T );
                            qrOffset += T;
                            qrPrevOffset += SPrev + TPrev;
                        }
                    }
                    // Move past the padding for this recv piece
                    const int recvTrunc =
                        ( secondRoot ? r-std::min(t,r) : r-std::min(s,r) );
                    recvHead +=
                        (recvTrunc*recvTrunc+recvTrunc)/2*sizeof(Scalar);

                    hmat_tools::PackedQR
                    ( r, s, t, &qrLevel[qrPieceOffset+(passes+1)*(r*r+r)],
                      &tauLevel[tauPieceOffset+(passes+2)*r], &qrWork[0] );

                    if( haveAnotherComm )
                    {
                        const int minDim = std::min(s+t,r);
                        if( rootOfNextStep )
                            halfHeightLevel[halfHeightOffset+2] = minDim;
                        else
                            halfHeightLevel[halfHeightOffset+3] = minDim;
                    }

                    qrPieceOffset += log2ParentTeamSize*(r*r+r);
                    tauPieceOffset += (log2ParentTeamSize+1)*r;
                }
            }
            passes += 2;
        }
    }
}

} // namespace dmhm
