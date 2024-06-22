#!/bin/bash
#SBATCH --job-name=baseline-training 
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1        
#SBATCH -c 32                      
#SBATCH --mem=32G                  
#SBATCH --gres=gpu:a100:1          
#SBATCH --time=02:30:00 
#SBATCH --output=single_gpu_train.out  
#SBATCH --error=single_gpu_train.err             


source $STORE/mytorchdist/bin/deactivate

source $STORE/mytorchdist/bin/activate

which python

python single_gpu_train.py




