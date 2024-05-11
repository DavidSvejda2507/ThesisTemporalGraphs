#!/bin/bash

#SBATCH --job-name="validation"
#SBATCH --array=0-997
#SBATCH --partition=slowq
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

python3 TestCalculations.py $SLURM_ARRAY_TASK_ID

exit 0