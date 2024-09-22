#!/bin/bash
#SBATCH -c 2           # 2 CPUs
#SBATCH --mem=60G       # 5 GB of RAM
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mail-user=prakritishetty02@gmail.com
#SBATCH --mail-type=ALL
module load CUDA
module load singularity
singularity exec --nv prakritishettywavlm_v3.sif bash /home/pss107/myscript.sh
