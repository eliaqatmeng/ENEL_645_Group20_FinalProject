#!/bin/bash
#SBATCH --job-name=tumor-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0-2:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --mail-type=END
#SBATCH --mail-user=jlahmed@ucalgary.ca

# Directly set the time limit in seconds for a 10-minute job
export JOB_TIME_LIMIT=7200

source ~/software/init-conda
conda activate pytorch

python brain_tumor_classification.py


