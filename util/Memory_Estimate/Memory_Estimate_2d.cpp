#include<iostream>
#include<cmath>

using namespace std;


int main( int argc, char* argv[] )
{
    int dim, numProc, xSize, ySize, numLevels, maxRank;
    dim = 2;
    cout << "numProc, xSize, ySize, numLevels, maxRank" << endl;
    cin >> numProc >> xSize >> ySize >> numLevels >> maxRank;
    numLevels -= 1.0;
    double mem_UV, mem_Tmp_UV, mem_Tmp_UVSqr;

    //double cc = pow(3.0, dim)*(pow(2.0, dim)-1);
    double cc = (2*dim+1)*(pow(2.0, dim)-1);
    int N = xSize*ySize;
    mem_UV = cc*maxRank*N*numLevels*2;
    mem_Tmp_UV = pow(cc, 2.0)*maxRank*N*numLevels*(numLevels-1)/2*2;

    mem_Tmp_UVSqr = 0.0;
    int k=int(log(numProc)/log(pow(2.0,dim)));
    int kk = int(numProc/pow(2.0,dim*k));
    for(int l=0; l<k; l++)
        mem_Tmp_UVSqr += cc*cc*cc*l*l*maxRank*maxRank;

    if( kk != 1 )
    {
        k++;
        mem_Tmp_UVSqr += cc*cc*cc*k*k*maxRank*maxRank*pow(2.0, dim)/kk;
    }

    for(int l=k; l<numLevels; l++)
        mem_Tmp_UVSqr += cc*cc*cc*l*l*maxRank*maxRank*pow(pow(2.0,dim),l-k);

    mem_Tmp_UVSqr *= 4.0;

    cout << "MEMROY_ESTIMATE(MB):" << endl;
    cout << "Memory for LowRank:           " << mem_UV*16/1024/1024 << endl;
    cout << "Memory for Temp LowRank:      " << mem_Tmp_UV*16/1024/1024 << endl;
    cout << "Memory for Temp LowRank Sqr:  " << mem_Tmp_UVSqr*16/1024/1024 << endl;
}
