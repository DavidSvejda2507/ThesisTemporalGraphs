#!/bin/bash

#SBATCH --job-name="order_true"
#SBATCH --array=0-0
#SBATCH --partition=defq
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

python3 GraphMeasuring.py c -i orders_true.txt $SLURM_ARRAY_TASK_ID

exit 0
