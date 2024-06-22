#!/bin/bash
#SBATCH --job-name=data_parallel_train
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1        
#SBATCH -c 64                      
#SBATCH --mem=32G                  
#SBATCH --gres=gpu:a100:2        
#SBATCH --time=02:30:00 
#SBATCH --output=data_parallel_train.out  
#SBATCH --error=data_parallel_train.err             


export CUDA_VISIBLE_DEVICES=0,1 

source $STORE/mytorchdist/bin/deactivate

source $STORE/mytorchdist/bin/activate

which python

python data_parallel_train.py  
