import os
import math

dim = 2

if dim == 2:
    proc_list   = [2**i for i in range(4, 15)]
    size_list   = [2**i for i in range(7, 16)] 
else:
    proc_list   = [2**i for i in range(4, 15)]
    size_list   = [2**i for i in range(4, 11)] 

pwd           = os.path.dirname(os.path.realpath(__file__))
work_folder   = pwd+'/MatVec'+str(dim)+'D'
out_folder    = work_folder+'/out'

for proc in proc_list:
    itproc = math.floor(math.log2(proc)+0.1)
    namestr = 'MV_'+str(dim)+'D_'+str(proc)
    out_file = out_folder+'/out_'+namestr+'.out'
    with open(out_file,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line != 'TACC:  Starting parallel tasks...':
                continue
            siz    = math.nan
            strong = math.nan
            rank   = math.nan
            tim    = math.nan
            while line != 'TACC:  Shutdown complete. Exiting.':
                if 'xSize' in line:
                    nums = [int(w) for w in line.split() if w.isdigit()]
                    siz = nums[1]
                if 'strong' in line:
                    nums = [int(w) for w in line.split() if w.isdigit()]
                    strong = nums[1]
                if 'maxRank' in line:
                    nums = [int(w) for w in line.split() if w.isdigit()]
                    rank = nums[1]
                if 'against vectors...done' in line:
                    nums = [double(w) for w in line.split() if w.isdigit()]
                    tim = nums[0]
                line = f.readline()

            if tim != math.nan and rank != 1:
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
