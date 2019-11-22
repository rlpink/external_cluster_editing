#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 5G
#SBATCH -p long
#SBATCH --open-mode append
module add python/3.6.4
python3 experiments_v5.py $1 instances/$1 $2 $3
