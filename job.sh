#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short
#SBATCH --gres=gpu:1

cd /home/siaosorio/Tesis

source	venv/bin/activate
echo START__________________________
echo ENTORNO________________________
echo $VIRTUAL_ENV

pwd


echo PYTHON_________________________

python inpainting.py 
