#!/bin/bash
#SBATCH --job-name=t1d_features
#SBATCH --output=job_logs/features_%j.out   # Standard output log
#SBATCH --error=job_logs/features_%j.err    # Standard error log
#SBATCH --nodes=1                           # Request 1 node
#SBATCH --ntasks=1                          # Request 1 task
#SBATCH --cpus-per-task=4                   
#SBATCH --mem=100G                          
#SBATCH --time=04:00:00                     

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

# 5. Execute the Python script
python ./src/features/resilience_features.py

echo "Pipeline complete."