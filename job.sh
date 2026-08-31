#!/bin/bash
#SBATCH --job-name=restauration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/restauration_%j.out
#SBATCH --error=logs/restauration_%j.err

# Ajusta la ruta a tu home en el cluster:
cd /home/siaosorio/Tesis
# cd /home/santiafonso/Tesis

mkdir -p logs results_rgb

# Opcional: prueba rápida con MAX_SIDE=512 antes de la imagen completa
# export MAX_SIDE=512
# export NUM_ITER=5000

export PYTHONUNBUFFERED=1
./venv/bin/python restorationGRIS.py
