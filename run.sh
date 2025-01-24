#!/bin/bash 
#SBATCH -A IscrC_PECFT
#SBATCH -p boost_usr_prod
#SBATCH --time 20:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1      # 4 gpus per node out of 4
#SBATCH --mem=64GB          # memory per node out of 494000MB 
#SBATCH --job-name=lora_masks
#SBATCH --output=/leonardo_scratch/fast/IscrC_PECFT/eric/lora_merge/outputs/experiment_%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_PECFT/eric/lora_merge/outputs/experiment_%j.err
module load python/3.11
module load cuda/12.1
module load gcc/11.3.0
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
source /leonardo_scratch/fast/IscrC_FoundCL/cl/lowrank/bin/activate
export CUDA_LAUNCH_BLOCKING=1


python s-dora01.py


