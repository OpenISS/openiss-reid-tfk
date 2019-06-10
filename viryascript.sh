#!/bin/bash

# Specify hard time limit for the job.
#   The job will be aborted if it runs longer than this time.
#$ -l h_rt=24:00:00


# Send an email when the job starts, finishes or if it is aborted.
#$ -m bea
#$ -M haotao.lai@gmail.com

# Give job a name
#$ -N reid-tfk
# Combine output and error files into a single file
#$ -j y
# Specify the output file name
#$ -o reid-tfk.qlog

# Set output directory to current
#$ -cwd

# Requst memory and gpu
#$ -l h_vmem=32G
#$ -l GPU=1

# Execute the script
module load python/3.6.8
python reid.py
