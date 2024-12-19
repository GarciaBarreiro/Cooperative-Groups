#!/bin/bash

size=$1

if [ -z $1 ]; then
    size=16
fi

groups=groups_${size}.out
kernels=kernels_${size}.out
pseudo=pseudo_${size}.out

nvcc -DtileSize=${size} -o $groups code/reduction.cu
nvcc -DtileSize=${size} -o $kernels code/reduction_kernels.cu
nvcc -DtileSize=${size} -o $pseudo code/reduction_pseudo.cu

sbatch sbatch.sh $groups
sbatch sbatch.sh $kernels
sbatch sbatch.sh $pseudo
