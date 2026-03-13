#!/bin/bash -l
#$ -N t1d_features
#$ -cwd
#$ -o job_logs/features_$JOB_ID.out
#$ -e job_logs/features_$JOB_ID.err
#$ -l h_rt=04:00:00
#$ -pe shared 2
#$ -l h_data=50G                 

echo "Starting T1D Feature Engineering Pipeline..."

# 1. Navigate to your specific project directory
cd /u/home/o/obriscoe/project-xyang123/T1D_ML/

# 2. Make sure the job logs directory exists
mkdir -p job_logs

# 3. Load the necessary Hoffman2 modules (uncomment if you use conda)
module load python/3.9.6 gcc/10.2.0

# activate environment
source clean_env/bin/activate

echo "Environment activated."

python ./src/features/resilience_features.py

echo "Pipeline complete."