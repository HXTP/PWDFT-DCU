module purge
#module add sghpc-mpi-intel/25.6 mpi/intelmpi/2021.14.0
#module load sghpc-mpi-clang-mlnx/25.6 sghpc-mathlib/25.6-intel

module use /public/software/modules/
module load compiler/intel/2021.3.0
module load mpi/intelmpi/2021.14.0
#source /public/home/sghpc_sdk/Linux_x86_64/25.6/dtk/dcc-2506/env.sh
source /public/home/sghpc_sdk/Linux_x86_64/25.8/dtk/dtk-25.04.2/env.sh
module load mpi/ucx/1.18.0/dtk-25.04/mlnx
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/home/acents8f17/Math/fftw-3.3.10/lib
which hipcc
which mpirun

DIST_DIR=$( cd -P -- "$(dirname -- "$0")" && pwd -P )
export DGDFT_DIR=${DIST_DIR}
export HIP_DIR=/public/home/sghpc_sdk/Linux_x86_64/25.6/dtk/dcc-2506
export LIBXC_DIR=/public/home/acents8f17/code/libxc-6.2.2
export FFTW_DIR=/public/home/acents8f17/code/fftw-3.3.10
export MKL_ROOT=/public/home/sghpc_sdk/Linux_x86_64/25.6/compilers/intel/2021.3.0/mkl
export YAML_DIR=/public/home/acents8f17/code/yaml-cpp-0.8.0
#export MFFT_DIR=/public/home/whu_ustc/DGDFT-2DFFT/MFFT-test
