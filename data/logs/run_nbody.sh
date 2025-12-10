#!/bin/bash
#SBATCH --job-name=nbody_karaen
#SBATCH --account=mpcs56430
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --constraint=v100

#SBATCH --output=nbody_%j.out
#SBATCH --error=nbody_%j.err

module load cuda

# Compile
nvcc -O3 nbody.cu    -o nbody
nvcc -O3 nbody_bh.cu -o nbody_bh

echo "Running brute-force N-body for 3000 particles..."
./nbody 3000

echo "Running Barnes–Hut N-body with theta = 0.7 for 3000 particles..."
./nbody_bh 3000  0.7
