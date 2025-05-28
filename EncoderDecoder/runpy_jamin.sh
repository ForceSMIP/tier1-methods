#!/bin/bash

##SBATCH --nodelist=mal-node2
##SBATCH --partition=mal_all
#SBATCH --partition=mal_all
#SBATCH --nodelist=mal-node2
#SBATCH --cpus-per-task=5

# Some more handy options. For a full list or explanation of an optional https://slurm.schedmd.com/sbatch.html
#SBATCH --output=monmaxtasmax_iv.txt
#SBATCH --job-name=monmaxtasmax_iv
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jaminr@colostate.edu
var="zmta"
component="forced"
python run_experiment_new.py "${component}_test_standard_${var}_annual"
python run_experiment_new.py "${component}_test_standard_${var}_1"
python run_experiment_new.py "${component}_test_standard_${var}_2"
python run_experiment_new.py "${component}_test_standard_${var}_3"
python run_experiment_new.py "${component}_test_standard_${var}_4"
python run_experiment_new.py "${component}_test_standard_${var}_5"
python run_experiment_new.py "${component}_test_standard_${var}_6"
python run_experiment_new.py "${component}_test_standard_${var}_7"
python run_experiment_new.py "${component}_test_standard_${var}_8"
python run_experiment_new.py "${component}_test_standard_${var}_9"
python run_experiment_new.py "${component}_test_standard_${var}_10"
python run_experiment_new.py "${component}_test_standard_${var}_11"
python run_experiment_new.py "${component}_test_standard_${var}_12"

