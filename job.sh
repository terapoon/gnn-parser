#!/bin/sh
#SBATCH -p v
#SBATCH -t 48:0:0
#SBATCH --gres=gpu:4

srun `pipenv run train`
