#!/bin/bash
#SBATCH --job-name=gris_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gris_sweep_%j.out
#SBATCH --error=logs/gris_sweep_%j.err

# Barrido DIP sobre la imagen del modelo del continuo (phase_diagram_sim.png, 64x64
# gris) con distintas fracciones de mascara. Una carpeta por %:
#   results_gris_sweep/mf0.XX/
#     mask.png                original | enmascarada | (original*mask)
#     iter_00000.png ...      reconstruccion cada SHOW_EVERY iteraciones
#     final_comparison.png    original | enmascarada | reconstruida | |error|  (pixel-exacto)
#     comparison_annotated.png  idem con colorbar en el mapa de error + PSNR/SSIM
#     psnr_curve.png          PSNR/SSIM vs iteracion
#     metrics.csv             tabla de metricas
#     restored.png / .npy     reconstruccion final
#     original.npy            objetivo (tras crop + reflection pad)

cd /home/siaosorio/Tesis
# cd /home/santiafonso/Tesis

mkdir -p logs results_gris_sweep

export PYTHONUNBUFFERED=1
export IMAGE_PATH=./data/restoration/phase_diagram_sim.png
export NUM_ITER=11000
export SHOW_EVERY=100

MASKS="0.50 0.70 0.80 0.90 0.95 0.98 0.99"

for mf in $MASKS; do
    export MASK_FRAC=$mf
    export OUTPUT_DIR=results_gris_sweep/mf${mf}
    echo "==================================================================="
    echo "MASK_FRAC=$mf  ->  $OUTPUT_DIR   ($(date))"
    echo "==================================================================="
    ./venv/bin/python restorationGRIS.py
done

echo "Barrido completo ($(date)). Resultados en results_gris_sweep/"
