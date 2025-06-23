#!/bin/bash
#SBATCH --account=pmlr_jobs
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=logs/train_%j.out

# Change to working directory 
cd /work/scratch/tdieudonne

# Set up environment without conflicting PYTHONPATH
source /work/courses/csnlp/Team3/env.sh
export PYTHONPATH="/work/scratch/tdieudonne:$PYTHONPATH" 
export TRANSFORMERS_CACHE=/work/courses/csnlp/Team3/slt/cache
conda activate csnlp

# Ensure Python uses local modules first
export PYTHONPATH="/work/scratch/tdieudonne:$PYTHONPATH"

echo "Starting train.py at $(date)"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

python train.py
