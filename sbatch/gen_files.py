import os
import math

dim = 3

if dim == 2:
    proc_list   = [2**i for i in range(4, 15)]
    size_list   = [2**i for i in range(7, 16)] 
    rank_list   = [4, 8] 
    strong_list = [0, 1]
    numVec      = 128
    nmin        = 8
    exec_path     = '/work/02539/lyzh588/stampede2/dmhm/build/bin/DistHMat2d/RandomMultiplyVectors'
else:
    proc_list   = [2**i for i in range(4, 15)]
    size_list   = [2**i for i in range(4, 11)] 
    rank_list   = [4, 8] 
    strong_list = [0, 1]
    numVec      = 128
    nmin        = 4
    exec_path     = '/work/02539/lyzh588/stampede2/dmhm/build/bin/DistHMat3d/RandomMultiplyVectors'


pwd           = os.path.dirname(os.path.realpath(__file__))
work_folder   = pwd+'/MatVec'+str(dim)+'D'
sbatch_folder = work_folder+'/sbatch'
out_folder    = work_folder+'/out'
err_folder    = work_folder+'/err'

os.makedirs(work_folder, exist_ok=True)
os.makedirs(sbatch_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)
os.makedirs(err_folder, exist_ok=True)

submitfile = work_folder+'/submit.sh'
with open(submitfile,'w') as sh_file:
    sh_file.write('#!/bin/bash\n')
os.chmod(submitfile, 0o775)

for proc in proc_list:
    node = math.ceil(proc/32)
    namestr = 'MV_'+str(dim)+'D_'+str(proc)
    sbatchfile = sbatch_folder+'/'+namestr+'.sbatch'
    with open(sbatchfile,'w') as sh_file:
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH -J %s\n'%namestr)
        sh_file.write('#SBATCH -o %s\n'%(out_folder+'/out_'+namestr+'.out'))
        sh_file.write('#SBATCH -e %s\n'%(err_folder+'/err_'+namestr+'.err'))
        sh_file.write('#SBATCH -t 4:00:00\n')
        sh_file.write('#SBATCH -p normal\n')
        sh_file.write('#SBATCH -A TG-MTH200003\n')
        sh_file.write('#SBATCH -N %d\n'%node)
        sh_file.write('#SBATCH -n %d\n\n'%proc)

    with open(submitfile,'a+') as sh_file:
        sh_file.write('sbatch %s\n'%sbatchfile)

    rep = 1

    for siz in size_list:
        if (siz//nmin)**dim < proc:
            continue
        numL = math.floor(math.log2(siz//nmin)+0.1)
        for strong in strong_list:
            if rep == 1:
                r_list = [1] + rank_list
                rep = 0
            else:
                r_list = rank_list
            for rank in r_list:
                with open(sbatchfile,'a+') as sh_file:
                    sh_file.write('ibrun tacc_affinity %s '%exec_path)
                    sh_file.write('--xSize %d '%siz)
                    sh_file.write('--ySize %d '%siz)
                    if dim == 3:
                        sh_file.write('--zSize %d '%siz)
                    sh_file.write('--numLevels %d '%numL)
                    sh_file.write('--strong %d '%strong)
                    sh_file.write('--maxRank %d '%rank)
                    sh_file.write('--numVectors %d\n'%numVec)
