#!/bin/bash
# Basic usage:

# $ sbatch hello.sbatch
#
# The job is queued and starts as soon as resources are available. The
# script is then executed on one of the allocated tasks, and standard
# output and standard error streams will be redirected to files that
# are prefixed by the job ID and job name. Commands prefixed with
# `srun' are executed on every task acquired by the job allocation.
#
# The sbatch options below allocate a single task on a single node,
# using a single CPU core with a one-hour time limit. To override
# these defaults, you can also supply sbatch options on the command
# line. For example:
#
# $ sbatch --cpus-per-task=32 --time=02:00:00 hello.sbatch

#SBATCH --job-name="hello"
#SBATCH --partition=defq
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=200
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt
#srun source ../D1/graphs/bin/activate

#python3 TestLeiden.py 10000
python3 TestLeidenCons.py 10000

exit 0
