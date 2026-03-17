#!/bin/bash

module purge
module use /usr/local/software/spack/spack-modules/a100-20210927/linux-centos8-zen2
module use /usr/local/software/spack/spack-modules/a100-20210927/linux-centos8-zen3
module add rhel8/slurm singularity
module add rhel8/global
module add gcc/9
module add cuda/12.1

rm -rf ~/.cache/flashinfer/

echo "=== Module list ==="
module list
echo ""
echo "=== nvcc version ==="
nvcc --version