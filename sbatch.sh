#!/bin/bash
#SBATCH -J job
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 00:10:00
#SBATCH --gres gpu:a100
#SBATCH --mem=32G

for i in {1..10}; do
    srun $1
done
