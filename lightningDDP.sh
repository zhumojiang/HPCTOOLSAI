#!/bin/bash
#SBATCH --job-name=lightning-ddp-training 
#SBATCH --nodes=2                  
#SBATCH --ntasks-per-node=2        
#SBATCH -c 32                      
#SBATCH --mem=32G                  
#SBATCH --gres=gpu:a100:2          
#SBATCH --time=02:30:00 
#SBATCH --output=lightningDDP.out  
#SBATCH --error=lightningDDP.err             


source $STORE/mytorchdist/bin/deactivate
source $STORE/mytorchdist/bin/activate


export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

which python


srun python lightningDDP.py
