#!/bin/bash
#SBATCH --job-name=gris_sweep128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gris_sweep128_%j.out
#SBATCH --error=logs/gris_sweep128_%j.err

# Barrido DIP sobre la imagen del modelo del continuo a 128x128 -- la resolucion
# de referencia ("good resolution") del paper de galpynostatic (ps_100_1_015946,
# N2 = 128x128). Mismo metodo que run_gris_sweep.slurm (64x64), una carpeta por %:
#   results_gris_sweep_128/mf0.XX/
#     mask.png  iter_00000..iter_11000.png  final_comparison.png
#     comparison_annotated.png  psnr_curve.png  metrics.csv
#     restored.png/.npy  original.npy
#
# El diagrama phase_diagram_sim_128.png se genera aparte (generate_phase_diagram.py
# con NUM_XI=NUM_ELL=128) y se sube al cluster; este job solo lo consume.

cd /home/siaosorio/Tesis
# cd /home/santiafonso/Tesis

mkdir -p logs results_gris_sweep_128

export PYTHONUNBUFFERED=1
export IMAGE_PATH=./data/restoration/phase_diagram_sim_128.png
export NUM_ITER=11000
export SHOW_EVERY=100

MASKS="0.50 0.70 0.80 0.90 0.95 0.98 0.99"

for mf in $MASKS; do
    export MASK_FRAC=$mf
    export OUTPUT_DIR=results_gris_sweep_128/mf${mf}
    echo "==================================================================="
    echo "MASK_FRAC=$mf  ->  $OUTPUT_DIR   ($(date))"
    echo "==================================================================="
    ./venv/bin/python restorationGRIS.py
done

echo "Barrido 128x128 completo ($(date)). Resultados en results_gris_sweep_128/"
