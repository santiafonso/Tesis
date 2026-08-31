#!/usr/bin/env python
# coding: utf-8

"""Deep Image Prior - restauracion en escala de grises (pipeline "mapa_suave").

Mismo metodo que la version anterior de este script (inpainting DIP con mascara
Bernoulli 50%, red 'skip' encoder-decoder, backtracking por caida de PSNR
enmascarado). Cambios respecto de la version anterior, todos relacionados con
resolucion/guardado/dispositivo (ver mensaje de chat para el detalle):

  1. Se recorta la imagen a un tamano multiplo de `dim_div_by` ANTES de construir
     la mascara y la red (esa variable ya estaba declarada pero nunca se usaba).
  2. Las imagenes de salida se guardan con PIL (pixel a pixel), no con
     matplotlib.savefig (que renderizaba ejes/margenes/remuestreo y no conservaba
     la resolucion real del array).
  3. La red y los tensores se mueven al dispositivo (GPU si esta disponible) en
     vez de quedarse en CPU pese a que el job de SLURM pide GPU.
"""

from __future__ import print_function

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

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
# Configuracion (sin cambios de metodo respecto del script original)
# --------------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results_gris")
os.makedirs(OUTPUT_DIR, exist_ok=True)

f = os.environ.get("IMAGE_PATH", "./data/restoration/mapa_suave.png")
# f = './data/restoration/barbara.png'

imsize = -1
dim_div_by = 64  # ya estaba declarado en el script original; antes no se usaba
show_every = int(os.environ.get("SHOW_EVERY", "100"))
pad = 'reflection'
INPUT = 'noise'
input_depth = 32
OPTIMIZER = 'adam'
OPT_OVER = 'net'
LR = 0.001
num_iter = int(os.environ.get("NUM_ITER", "11000"))
reg_noise_std = 0.03
PLOT = True
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.50"))  # fraccion NO observada (enmascarada)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------------------------------------
# Carga + control de resolucion
# --------------------------------------------------------------------------------
img_pil, img_np = get_image(f, imsize)
print("Imagen original:", f, "->", img_np.shape, "(C x H x W)")

if img_np.shape[0] != 1:
    img_pil = img_pil.convert('L')
    img_np = pil_to_np(img_pil)

# Recorte a multiplo de dim_div_by: evita que las 5 escalas de downsampling/
# upsampling de la red 'skip' terminen desalineadas y recorten bordes de forma
# impredecible (ver models/common.py:Concat, que recorta en silencio ramas que
# no calzan en tamano).
img_pil = crop_image(img_pil, dim_div_by)
img_np = pil_to_np(img_pil)
print("Imagen recortada a multiplo de %d:" % dim_div_by, img_np.shape, "(C x H x W)")

# Padding por reflexion de 1 px (igual que el pipeline original de barbara/mapa_suave)
img_np = nn.ReflectionPad2d(1)(np_to_torch(img_np))[0].numpy()
img_pil = np_to_pil(img_np)
print("Imagen tras reflection pad(+1):", img_np.shape,
      "(C x H x W) -- este es el tamano que tendran todas las imagenes de salida")

img_mask = get_bernoulli_mask(img_pil, MASK_FRAC)
img_mask_np = pil_to_np(img_mask)

img_masked = img_np * img_mask_np
mask_var = np_to_torch(img_mask_np).to(device)


def save_triptych(images_np, path):
    """Concatena imagenes 1xHxW [0,1] lado a lado y guarda un PNG, sin pasar por
    matplotlib (evita el remuestreo/ejes/margenes que alteraban la resolucion)."""
    imgs = [np_to_pil(np.clip(x, 0, 1)) for x in images_np]
    w, h = imgs[0].size
    grid = Image.new(imgs[0].mode, (w * len(imgs), h))
    for k, im in enumerate(imgs):
        grid.paste(im, (k * w, 0))
    grid.save(path)


save_triptych([img_np, img_mask_np, img_masked], os.path.join(OUTPUT_DIR, "mask.png"))

# --------------------------------------------------------------------------------
# Red (misma arquitectura/hiperparametros que el script original)
# --------------------------------------------------------------------------------
net = get_net(input_depth, 'skip', pad, n_channels=1,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').to(device)

mse = torch.nn.MSELoss().to(device)
img_var = np_to_torch(img_np).to(device)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).to(device).detach()

print("Resolucion net_input / img_var / mask_var:",
      tuple(net_input.shape), tuple(img_var.shape), tuple(mask_var.shape))

# --------------------------------------------------------------------------------
# Bucle principal (misma logica: MSE enmascarado + backtracking por caida de PSNR)
# --------------------------------------------------------------------------------


metrics_log = []  # (iter, psnr_full, psnr_masked, ssim_full)


def closure():
    global i, psrn_masked_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    if out.shape != img_var.shape:
        raise RuntimeError(
            "La salida de la red %s no coincide con la resolucion esperada %s. "
            "Revisar dim_div_by / arquitectura." % (tuple(out.shape), tuple(img_var.shape))
        )

    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    out_np = out.detach().cpu().numpy()[0]
    psrn_masked = compare_psnr(img_masked, out_np * img_mask_np)
    psrn = compare_psnr(img_np, out_np)

    print('Iteration %05d    Loss %f PSNR_masked %f PSNR %f' %
          (i, total_loss.item(), psrn_masked, psrn), '\r', end='')

    if PLOT and i % show_every == 0:
        if psrn_masked - psrn_masked_last < -5 and last_net is not None:
            print('\nFalling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.to(device))
            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_masked_last = psrn_masked

        out_clip = np.clip(out_np, 0, 1)
        np_to_pil(out_clip).save(os.path.join(OUTPUT_DIR, "iter_%05d.png" % i))
        ssim_full = compare_ssim(img_np[0], out_clip[0], data_range=1.0)
        metrics_log.append((i, float(psrn), float(psrn_masked), float(ssim_full)))

    i += 1
    return total_loss


last_net = None
psrn_masked_last = 0
i = 0

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR=LR, num_iter=num_iter)

# --------------------------------------------------------------------------------
# Salida final -- pixel-exacta + copias crudas en .npy para la comparacion posterior
# --------------------------------------------------------------------------------
with torch.no_grad():
    out_np = np.clip(torch_to_np(net(net_input)), 0, 1)

print("Resolucion final reconstruida:", out_np.shape, "(C x H x W)")

np_to_pil(out_np).save(os.path.join(OUTPUT_DIR, "restored.png"))
np.save(os.path.join(OUTPUT_DIR, "restored.npy"), out_np)
np.save(os.path.join(OUTPUT_DIR, "original.npy"), img_np)

# Panel crudo (pixel-exacto): original | enmascarada | reconstruida | |error|
abs_err = np.abs(img_np - out_np)
save_triptych(
    [img_np, img_masked, out_np, np.clip(abs_err / max(abs_err.max(), 1e-8), 0, 1)],
    os.path.join(OUTPUT_DIR, "final_comparison.png"),
)

final_psnr = compare_psnr(img_np, out_np)
final_ssim = compare_ssim(img_np[0], out_np[0], data_range=1.0)
final_mae = float(abs_err.mean())

# Figura anotada con colorbar en el mapa de error (para mostrarle al profe)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, title, data, cmap in [
    (axes[0], "Original", img_np[0], "gray"),
    (axes[1], "Enmascarada (%.0f%% oculto)" % (MASK_FRAC * 100), img_masked[0], "gray"),
    (axes[2], "Reconstruida DIP", np.clip(out_np[0], 0, 1), "gray"),
]:
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
im = axes[3].imshow(abs_err[0], cmap="inferno")
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
    ax1.plot(it, [m[2] for m in metrics_log], color="tab:cyan", ls="--",
             label="PSNR (masked)")
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
