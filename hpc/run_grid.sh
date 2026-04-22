#!/bin/bash
#!#############################################################
#!#### SLURM Header Definitions                        ########
#!#############################################################

#SBATCH -J dynspec_gridsearch          # Name of the job
#SBATCH -A MASCOLO-SL2-GPU             # Your specific SL2 Account
#SBATCH -p ampere                      # REQUIRED - The Wilkes3 GPU partition
#SBATCH --nodes=1                      # Use 1 node
#SBATCH --ntasks-per-node=1            # torchrun manages processes within the node
#SBATCH --gres=gpu:1                   # Requesting 1 GPU for speed
#SBATCH --cpus-per-task=16             # 16 CPUs per GPU is optimal for Wilkes3
#SBATCH --time=10:00:00                # Adjust based on your estimate (max 36h)
#SBATCH --mail-type=NONE
#SBATCH --output=gridsearch_%j.out     # Standard output log (%j = JobID)
#SBATCH --error=gridsearch_%j.err      # Standard error log

#! sbatch directives end here

#! ############################################################
#! Environment Setup (adapted from submit_pifold.slurm)
#! ############################################################

module purge
module use /usr/local/software/spack/spack-modules/a100-20210927/linux-centos8-zen2
module use /usr/local/software/spack/spack-modules/a100-20210927/linux-centos8-zen3
module add rhel8/slurm singularity
module add rhel8/global
module add cuda/12.1

export CUDAHOSTCXX=$CONDA_PREFIX/bin/gcc
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

rm -rf ~/.cache/flashinfer/
rm -rf ~/.cache/tvm-ffi/
rm -rf ~/.cache/sglang/

conda activate sglang-dev              

echo "=== Module list ==="
module list
echo ""
echo "=== nvcc version ==="
nvcc --version
echo ""
echo "=== Python ==="
which python
python --version

#! ############################################################
#! Run the grid search
#! ############################################################
python dynamic_spec_gridsearch.py --benchmark-list mtbench:80