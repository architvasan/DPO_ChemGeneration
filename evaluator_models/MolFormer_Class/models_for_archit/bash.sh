#!/bin/bash -l
#PBS -N cardio5
#PBS -l select=1
#PBS -m abe
#PBS -M lvairus@anl.gov
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience 
#PBS -o logs/cardio5.OU
#PBS -e logs/cardio5.ER

module use /soft/modulefiles
module load conda
conda activate /lus/eagle/projects/datascience/lvairus/envs/sst   

cd /lus/eagle/projects/datascience/lvairus/Pharmacokinetic_Modeling/ModelTraining/models_for_archit

# change sweep_id
CUDA_VISIBLE_DEVICES=0 python run_script_b_single.py -y yaml_files/resp.yaml & 
CUDA_VISIBLE_DEVICES=1 python run_script_b_single.py -y yaml_files/hepa.yaml & 
CUDA_VISIBLE_DEVICES=2 python run_script_b_single.py -y yaml_files/cardio5.yaml & 
CUDA_VISIBLE_DEVICES=3 python run_script_b_single.py -y yaml_files/cardio10.yaml & 

# Wait for all background processes to finish
wait
