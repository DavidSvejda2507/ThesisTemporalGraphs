#!/bin/bash

#SBATCH --job-name="order_false"
#SBATCH --array=0-4
#SBATCH --partition=defq
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

python3 GraphMeasuring.py c orders_false.txt $SLURM_ARRAY_TASK_ID

exit 0
