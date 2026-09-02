#!/usr/bin/env python
# coding: utf-8
"""Deep Image Prior - runner unificado de inpainting sobre una imagen con
mascara Bernoulli (gris o RGB).

Fusiona los tres scripts que habia en la raiz del repo:
  - restoration.py      (export crudo del notebook: barbara / kate / mapa_suave)
  - restorationGRIS.py   (version gris parametrizada, metricas PSNR/SSIM/MAE, CSV)
  - restorationRGB.py    (version color, solo PSNR)

El metodo es exactamente el de restorationGRIS.py (el que se venia desarrollando):
inpainting DIP con red 'skip' encoder-decoder, perdida MSE solo sobre los pixeles
observados (mse(out*mask, target*mask)), backtracking al ultimo checkpoint si el
PSNR enmascarado cae mas de 5 dB, recorte a multiplo de 64 y reflection pad de 1
px. Lo unico que se generalizo es el numero de canales: con N_CHANNELS=3 se
reproduce lo que hacia restorationRGB.py y ademas se calculan SSIM, MAE, el CSV y
las figuras anotadas (que la version RGB no tenia).

Se ejecuta como modulo desde la raiz del repo:
    ./venv/bin/python -m dip.inpaint

Variables de entorno (todas opcionales):
    IMAGE_PATH      imagen de entrada         (def: ./data/restoration/mapa_suave.png)
    OUTPUT_DIR      carpeta de salida         (def: results/dip)
    N_CHANNELS      1 (gris) o 3 (RGB); 0 = inferir de la imagen  (def: 0)
    MASK_FRAC       fraccion NO observada     (def: 0.50)
    NUM_ITER        iteraciones               (def: 11000)
    LR             learning rate              (def: 0.001)
    REG_NOISE_STD   ruido de regularizacion   (def: 0.03)
    SHOW_EVERY      cada cuantas iter se guarda snapshot + metricas (def: 100)
    MAX_SIDE        si >0, redimensiona el lado mayor a este valor (def: 0)
    SEED           si se define, fija la semilla (mascara + init reproducibles)

Reproduccion de los casos historicos de restoration.py:
    # barbara (gris, 50% oculto)
    IMAGE_PATH=./data/restoration/barbara.png OUTPUT_DIR=results/barbara \
        MASK_FRAC=0.50 ./venv/bin/python -m dip.inpaint
    # kate (RGB, 98% oculto, sin reg-noise, 1000 iter)
    IMAGE_PATH=./data/restoration/kate.png OUTPUT_DIR=results/kate N_CHANNELS=3 \
        MASK_FRAC=0.98 REG_NOISE_STD=0 LR=0.01 NUM_ITER=1000 \
        ./venv/bin/python -m dip.inpaint
"""
from __future__ import print_function

import csv
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from PIL import Image

from dip import metrics
from models import get_net
from utils.common_utils import (
    crop_image,
    get_image,
    get_params,
    np_to_pil,
    np_to_torch,
    optimize,
    pil_to_np,
    torch_to_np,
)
from utils.inpainting_utils import get_bernoulli_mask, get_noise

# --------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------
IMAGE_PATH = os.environ.get("IMAGE_PATH", "./data/restoration/mapa_suave.png")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results/dip")
N_CHANNELS = int(os.environ.get("N_CHANNELS", "0"))  # 0 = inferir
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.50"))  # fraccion NO observada
NUM_ITER = int(os.environ.get("NUM_ITER", "11000"))
LR = float(os.environ.get("LR", "0.001"))
REG_NOISE_STD = float(os.environ.get("REG_NOISE_STD", "0.03"))
SHOW_EVERY = int(os.environ.get("SHOW_EVERY", "100"))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "0"))  # 0 = sin redimensionar
SEED = os.environ.get("SEED")  # None => sin fijar semilla (comportamiento historico)

DIM_DIV_BY = 64  # la red 'skip' baja/sube 5 escalas -> el lado debe ser multiplo
PAD = "reflection"
INPUT = "noise"
INPUT_DEPTH = 32
OPTIMIZER = "adam"
OPT_OVER = "net"
PLOT = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

if SEED is not None:
    seed = int(SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------------------------------------
# Carga + control de resolucion / canales
# --------------------------------------------------------------------------------
img_pil, img_np = get_image(IMAGE_PATH, -1)
print("Imagen original:", IMAGE_PATH, "->", img_np.shape, "(C x H x W)")

n_channels = N_CHANNELS if N_CHANNELS in (1, 3) else (1 if img_np.shape[0] == 1 else 3)
mode = "L" if n_channels == 1 else "RGB"
if img_pil.mode != mode:
    img_pil = img_pil.convert(mode)
    img_np = pil_to_np(img_pil)
print("Canales:", n_channels, "(modo PIL %s)" % mode)

if MAX_SIDE > 0:
    w, h = img_pil.size
    scale = MAX_SIDE / float(max(w, h))
    if scale < 1.0:
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img_np = pil_to_np(img_pil)
        print("Redimensionada a MAX_SIDE=%d:" % MAX_SIDE, img_np.shape)

# Recorte a multiplo de DIM_DIV_BY: evita que las 5 escalas de la red 'skip'
# terminen desalineadas y recorten bordes de forma impredecible (models/common.py
# Concat recorta en silencio ramas que no calzan en tamano).
img_pil = crop_image(img_pil, DIM_DIV_BY)
img_np = pil_to_np(img_pil)
print("Recortada a multiplo de %d:" % DIM_DIV_BY, img_np.shape, "(C x H x W)")

# Padding por reflexion de 1 px (igual que el pipeline barbara/mapa_suave)
img_np = nn.ReflectionPad2d(1)(np_to_torch(img_np))[0].numpy()
img_pil = np_to_pil(img_np)
print("Tras reflection pad(+1):", img_np.shape, "-- este es el tamano de salida")

img_mask = get_bernoulli_mask(img_pil, MASK_FRAC)
img_mask_np = pil_to_np(img_mask)
img_masked = img_np * img_mask_np
mask_var = np_to_torch(img_mask_np).to(device)


def save_grid(images_np, path):
    """Concatena imagenes C x H x W en [0, 1] lado a lado y guarda un PNG con PIL
    (pixel a pixel, sin ejes/margenes/remuestreo de matplotlib)."""
    imgs = [np_to_pil(np.clip(x, 0, 1)) for x in images_np]
    w, h = imgs[0].size
    grid = Image.new(imgs[0].mode, (w * len(imgs), h))
    for k, im in enumerate(imgs):
        grid.paste(im, (k * w, 0))
    grid.save(path)


save_grid([img_np, img_mask_np, img_masked], os.path.join(OUTPUT_DIR, "mask.png"))

# --------------------------------------------------------------------------------
# Red
# --------------------------------------------------------------------------------
net = get_net(
    INPUT_DEPTH,
    "skip",
    PAD,
    n_channels=n_channels,
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    upsample_mode="bilinear",
).to(device)

mse = torch.nn.MSELoss().to(device)
img_var = np_to_torch(img_np).to(device)
net_input = get_noise(INPUT_DEPTH, INPUT, img_np.shape[1:]).to(device).detach()

print(
    "Resolucion net_input / img_var / mask_var:",
    tuple(net_input.shape),
    tuple(img_var.shape),
    tuple(mask_var.shape),
)

# --------------------------------------------------------------------------------
# Bucle principal: MSE enmascarado + backtracking por caida de PSNR enmascarado
# --------------------------------------------------------------------------------
metrics_log = []  # (iter, psnr_full, psnr_masked, ssim_full)


def closure():
    global i, psrn_masked_last, last_net, net_input

    if REG_NOISE_STD > 0:
        net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)

    out = net(net_input)

    if out.shape != img_var.shape:
        raise RuntimeError(
            "La salida de la red %s no coincide con la resolucion esperada %s. "
            "Revisar DIM_DIV_BY / arquitectura." % (tuple(out.shape), tuple(img_var.shape))
        )

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    out_np = out.detach().cpu().numpy()[0]
    psrn_masked = metrics.psnr_masked(img_masked, out_np * img_mask_np)
    psrn = metrics.psnr(img_np, out_np)

    print(
        "Iteration %05d    Loss %f PSNR_masked %f PSNR %f"
        % (i, total_loss.item(), psrn_masked, psrn),
        "\r",
        end="",
    )

    if PLOT and i % SHOW_EVERY == 0:
        if psrn_masked - psrn_masked_last < -5 and last_net is not None:
            print("\nFalling back to previous checkpoint.")
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.to(device))
            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_masked_last = psrn_masked

        out_clip = np.clip(out_np, 0, 1)
        np_to_pil(out_clip).save(os.path.join(OUTPUT_DIR, "iter_%05d.png" % i))
        ssim_full = metrics.ssim(img_np, out_clip, n_channels)
        metrics_log.append((i, float(psrn), float(psrn_masked), float(ssim_full)))

    i += 1
    return total_loss


last_net = None
psrn_masked_last = 0
i = 0

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR=LR, num_iter=NUM_ITER)

# --------------------------------------------------------------------------------
# Salida final -- pixel-exacta + copias crudas en .npy para comparacion posterior
# --------------------------------------------------------------------------------
with torch.no_grad():
    out_np = np.clip(torch_to_np(net(net_input)), 0, 1)

print("\nResolucion final reconstruida:", out_np.shape, "(C x H x W)")

np_to_pil(out_np).save(os.path.join(OUTPUT_DIR, "restored.png"))
np.save(os.path.join(OUTPUT_DIR, "restored.npy"), out_np)
np.save(os.path.join(OUTPUT_DIR, "original.npy"), img_np)

# Panel crudo (pixel-exacto): original | enmascarada | reconstruida | |error|
abs_err = metrics.abs_error_map(img_np, out_np)
save_grid(
    [img_np, img_masked, out_np, np.clip(abs_err / max(abs_err.max(), 1e-8), 0, 1)],
    os.path.join(OUTPUT_DIR, "final_comparison.png"),
)

final_psnr = metrics.psnr(img_np, out_np)
final_ssim = metrics.ssim(img_np, out_np, n_channels)
final_mae = metrics.mae(img_np, out_np)


def _disp(x):
    """C x H x W -> algo que imshow entienda (H x W gris, o H x W x 3 RGB)."""
    return x[0] if n_channels == 1 else np.moveaxis(np.clip(x, 0, 1), 0, -1)


_cmap = "gray" if n_channels == 1 else None

# Figura anotada con colorbar en el mapa de error (para mostrar)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, title, data in [
    (axes[0], "Original", img_np),
    (axes[1], "Enmascarada (%.0f%% oculto)" % (MASK_FRAC * 100), img_masked),
    (axes[2], "Reconstruida DIP", out_np),
]:
    ax.imshow(_disp(data), cmap=_cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
im = axes[3].imshow(abs_err.mean(0), cmap="inferno")
axes[3].set_title("|error|  (MAE=%.4f)" % final_mae)
axes[3].axis("off")
fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
fig.suptitle(
    "MASK_FRAC=%.3f   PSNR=%.2f dB   SSIM=%.4f" % (MASK_FRAC, final_psnr, final_ssim)
)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "comparison_annotated.png"), dpi=120)
plt.close(fig)

# Curva de metricas vs iteracion
if metrics_log:
    it = [m[0] for m in metrics_log]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(it, [m[1] for m in metrics_log], color="tab:blue", label="PSNR (full)")
    ax1.plot(
        it, [m[2] for m in metrics_log], color="tab:cyan", ls="--", label="PSNR (masked)"
    )
    ax1.set_xlabel("iteracion")
    ax1.set_ylabel("PSNR [dB]", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(it, [m[3] for m in metrics_log], color="tab:red", label="SSIM (full)")
    ax2.set_ylabel("SSIM", color="tab:red")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
    ax1.set_title("MASK_FRAC=%.3f  -  metricas vs iteracion" % MASK_FRAC)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "psnr_curve.png"), dpi=120)
    plt.close(fig)

with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["iter", "psnr_full", "psnr_masked", "ssim_full"])
    w.writerows(metrics_log)
    w.writerow([])
    w.writerow(["final", final_psnr, "", final_ssim])
    w.writerow(["mae", final_mae, "", ""])

print("PSNR final (imagen completa):", final_psnr)
print("SSIM final (imagen completa):", final_ssim)
print("MAE final:", final_mae)
print("Listo. Salidas en:", OUTPUT_DIR)
