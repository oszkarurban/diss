#!/bin/bash

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

# Redirect caches and tmp to RDS (home quota is tight at 50GB)
export TMPDIR=/rds/user/ou222/hpc-work/tmp
mkdir -p $TMPDIR

# Symlink flashinfer cache to RDS if not already
if [ ! -L ~/.cache/flashinfer ]; then
    rm -rf ~/.cache/flashinfer
    mkdir -p /rds/user/ou222/hpc-work/.cache/flashinfer
    ln -s /rds/user/ou222/hpc-work/.cache/flashinfer ~/.cache/flashinfer
fi

rm -rf ~/.cache/flashinfer/*
rm -rf ~/.cache/tvm-ffi/
rm -rf ~/.cache/sglang/

echo "=== Module list ==="
module list
echo ""
echo "=== nvcc version ==="
nvcc --version