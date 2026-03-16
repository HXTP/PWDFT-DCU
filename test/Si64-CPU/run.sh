#!/bin/bash
#SBATCH -J c-p-a
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --ntasks-per-node=4
#SBATCH -p normal
#SBATCH --mem=90G
#SBATCH --gres=dcu:4
#SBATCH --exclusive
module purge
module add compiler/rocm/dtk/23.04 compiler/devtoolset/7.3.1 compiler/intel/2017.5.239 mpi/hpcx/2.11.0/gcc-7.3.1

source /opt/hpc/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64

srun hostname | sort -u > nd

sed -i 's/$/ slots=28/g' nd

NNODE=$(wc -l nd | awk '{print $1}' )
NP=$[1*NNODE]

#APP=/public/home/whu_ustc/jiaosz/soft/aimd/cpu/aimd_cpu/examples/pwdft
#APP=/public/home/whu_ustc/jiaosz/soft/hefeidgdft_aimd_dipole/examples/pwdft
#APP=/work1/whu_ustc/jiaosz/AIMD_HPC/dgdft-DCU-para-dtk-23.04/examples/pwdft
APP=/public/home/whu_ustc/xmqin/2025/software/dghf-CPU/examples/pwdft
#APP=/public/home/whu_ustc/xmqin/2025/software/dghf-DCU/examples/pwdft
#APP=/public/home/whu_ustc/yaoyf/soft/2023.9.11/dgdft-yaml-DCU-dtk-21.10.1/examples/pwdft
#APP=/public/home/whu_ustc/yaoyf/soft/2023.12.11/dgdft-DCU-para-dtk-23.10/examples/pwdft
srun --mpi=pmix_v3 $APP
#/opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1/bin/mpirun -np 40 $APP

