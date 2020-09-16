import os
import math
import re
import numpy as np
import scipy.io

dim = 3

if dim == 2:
    maxitproc = 14
    maxitsize = 16
    proc_list   = [2**i for i in range(4, maxitproc)]
    size_list   = [2**i for i in range(7, maxitsize)] 
else:
    maxitproc = 14
    maxitsize = 11
    proc_list   = [2**i for i in range(4, maxitproc)]
    size_list   = [2**i for i in range(4, maxitsize)] 

pwd           = os.path.dirname(os.path.realpath(__file__))
work_folder   = pwd+'/MatVec'+str(dim)+'D'
out_folder    = work_folder+'/out'

W4 = np.full((maxitproc,maxitsize), np.nan)
W8 = np.full((maxitproc,maxitsize), np.nan)
S4 = np.full((maxitproc,maxitsize), np.nan)
S8 = np.full((maxitproc,maxitsize), np.nan)

for proc in proc_list:
    itproc = math.floor(math.log2(proc)+0.1)
    namestr = 'MV_'+str(dim)+'D_'+str(proc)
    out_file = out_folder+'/out_'+namestr+'.out'
    print(out_file)
    with open(out_file,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not 'TACC:  Starting parallel tasks...' in line:
                continue
            siz    = math.nan
            strong = math.nan
            rank   = math.nan
            tim    = math.nan
            while not 'TACC:  Shutdown complete. Exiting.' in line:
                if '--xSize' in line:
                    nums = [int(w) for w in re.split(r'[|,|]', line)
                            if w.isdigit()]
                    siz = nums[1]
                if '--strong' in line:
                    nums = [int(w) for w in re.split(r'[|,|]', line)
                            if w.isdigit()]
                    strong = nums[1]
                if '--maxRank' in line:
                    nums = [int(w) for w in re.split(r'[|,|]', line)
                            if w.isdigit()]
                    rank = nums[1]
                if 'against vectors...done' in line:
                    strs = [w for w in line.split()]
                    tim = float(strs[5])
                line = f.readline()

            if not np.isnan(tim) and rank != 1:
                itsiz = math.floor(math.log2(siz)+0.1)
                if strong == 0 and rank == 4:
                    W4[itproc, itsiz] = tim
                if strong == 0 and rank == 8:
                    W8[itproc, itsiz] = tim
                if strong == 1 and rank == 4:
                    S4[itproc, itsiz] = tim
                if strong == 1 and rank == 8:
                    S8[itproc, itsiz] = tim
                print(proc, siz, strong, rank, tim)

scipy.io.savemat(work_folder+'/results.mat',
        {'W4':W4, 'W8':W8, 'S4':S4, 'S8':S8})
