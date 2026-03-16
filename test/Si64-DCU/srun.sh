#!/bin/bash
#SBATCH -J d-a-p     
#SBATCH -p newlarge         
#SBATCH -N 1         
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --gres=dcu:1       
#SBATCH -o aimd-%j.log     
#SBATCH -e aimd-%j.err 
#SBATCH --exclusive


module purge
#module load sghpc-mpi-clang-mlnx/25.6
#module add sghpc-mpi-intel/25.6 mpi/intelmpi/2021.14.0
module use /public/software/modules/
module load compiler/intel/2021.3.0
#module load mpi/intelmpi/2021.14.0
module load mpi/intel/2021.3.0 
module load sghpc-mathlib/25.6-intel
source /public/home/sghpc_sdk/Linux_x86_64/25.6/dtk/dcc-2506/env.sh
which hipcc
which mpirun
#export SHCA_DEBUG_FILE=/public/home/acents8f17/yaoyf/work/test-1/shca_file_current.log


##多节点
#mpirun -np 8 -genv UCX_TLS=rc,sm  -genv FI_MPI_FABRICS ofi -genv FI_PROVIDER ucx  ./intelmpi_bind.sh    /public/home/acents8f17/soft/dghf-DCU-0901/examples/pwdft

##单节点

mpirun -np 1 -genv UCX_TLS=rc,sm  -genv FI_MPI_FABRICS ofi -genv FI_PROVIDER ucx  -genv UCX_TLS=sm ./intelmpi_bind.sh   /public/home/acents8f17/soft/dghf-DCU-0901/examples/pwdft
