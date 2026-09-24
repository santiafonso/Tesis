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
    ./venv/bin/python -m dip.restoration

Variables de entorno (todas opcionales):
    IMAGE_PATH      imagen de entrada         (def: ./data/restoration/mapa_suave.png)
    OUTPUT_DIR      carpeta de salida         (def: results/dip)
    N_CHANNELS      1 (gris) o 3 (RGB); 0 = inferir de la imagen  (def: 0)
    MASK_FRAC       fraccion NO observada     (def: 0.50)
    NUM_ITER        iteraciones               (def: 11000)
    LR             learning rate              (def: 0.001)
    REG_NOISE_STD   ruido de regularizacion   (def: 0.03; bajar a ~0.01 ayuda con
                   frentes de fase abruptos -- g negativos)
    SHOW_EVERY      cada cuantas iter se loguea metrica en el CSV/curva (def: 100)
    SNAPSHOT_EVERY  cada cuantas iter se guarda un PNG en snapshots/ (def: =SHOW_EVERY)
    PSNR_DROP_TOL   caida de PSNR_masked [dB] respecto de su media movil que dispara
                   el backtracking al ultimo checkpoint (def: -5.0; MAS negativo =
                   backtracking MENOS agresivo -- usar -8/-10 en frentes abruptos)
    MAX_FALLBACKS   rollbacks seguidos permitidos antes de aceptar el estado y
                   seguir; evita que un checkpoint malo congele la corrida (def: 3)
    MAX_SIDE        si >0, redimensiona el lado mayor a este valor (def: 0)
    SEED           si se define, fija la semilla (mascara + init reproducibles)
    MASK_PATH      .npy (bool H x W) con una mascara fija de pixeles observados;
                   si se define, ignora MASK_FRAC y no usa Bernoulli. Se le aplica
                   el mismo reflection-pad(+1) que a la imagen. Ver dip.frontier_mask.

Reproduccion de los casos historicos de restoration.py:
    # barbara (gris, 50% oculto)
    IMAGE_PATH=./data/restoration/barbara.png OUTPUT_DIR=results/barbara \
        MASK_FRAC=0.50 ./venv/bin/python -m dip.restoration
    # kate (RGB, 98% oculto, sin reg-noise, 1000 iter)
    IMAGE_PATH=./data/restoration/kate.png OUTPUT_DIR=results/kate N_CHANNELS=3 \
        MASK_FRAC=0.98 REG_NOISE_STD=0 LR=0.01 NUM_ITER=1000 \
        ./venv/bin/python -m dip.restoration
"""
from __future__ import print_function

import csv
import glob
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
SNAPSHOT_EVERY = int(os.environ.get("SNAPSHOT_EVERY", str(SHOW_EVERY)))
PSNR_DROP_TOL = float(os.environ.get("PSNR_DROP_TOL", "-5.0"))  # dB; <0 (mas neg = menos agresivo)
MAX_FALLBACKS = int(os.environ.get("MAX_FALLBACKS", "3"))  # rollbacks seguidos antes de aceptar y seguir
PSNR_EMA_ALPHA = 0.3  # peso del valor nuevo en la media movil de PSNR_masked
MAX_SIDE = int(os.environ.get("MAX_SIDE", "0"))  # 0 = sin redimensionar
SEED = os.environ.get("SEED")  # None => sin fijar semilla (comportamiento historico)
MASK_PATH = os.environ.get("MASK_PATH")  # si se define, mascara fija en vez de Bernoulli

DIM_DIV_BY = 32  # la red 'skip' baja/sube 5 escalas con stride 2 (2**5=32) -> el lado debe ser multiplo
PAD = "reflection"
# Arquitectura (defaults = los historicos; expuestos para la busqueda de autoexp/)
INPUT = os.environ.get("INPUT_TYPE", "noise")  # "noise" | "meshgrid" (meshgrid fuerza 2 canales)
INPUT_DEPTH = 2 if INPUT == "meshgrid" else int(os.environ.get("INPUT_DEPTH", "32"))
NET_WIDTH = int(os.environ.get("NET_WIDTH", "128"))  # canales de skip_n33d/skip_n33u
SKIP_N11 = int(os.environ.get("SKIP_N11", "4"))
NUM_SCALES = int(os.environ.get("NUM_SCALES", "5"))
UPSAMPLE_MODE = os.environ.get("UPSAMPLE_MODE", "bilinear")
OPTIMIZER = "adam"
OPT_OVER = "net"
PLOT = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Los snapshots por iteracion (decenas por corrida, todos "del mismo estilo") van
# a su propia subcarpeta; en OUTPUT_DIR quedan solo los entregables.
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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

if MASK_PATH:
    m = np.load(MASK_PATH).astype(np.float32)
    if m.ndim == 3:
        m = m[0]
    ph = img_np.shape[1] - m.shape[0]
    pw = img_np.shape[2] - m.shape[1]
    if (ph, pw) == (2, 2):
        m = np.pad(m, 1, mode="reflect")  # mismo reflection-pad(+1) que la imagen
    elif (ph, pw) != (0, 0):
        raise RuntimeError(
            "MASK_PATH %s: shape %s incompatible con la imagen %s "
            "(se admite igual, o 2 px menos por el pad)"
            % (MASK_PATH, m.shape, tuple(img_np.shape[1:]))
        )
    img_mask_np = np.repeat(m[None], n_channels, axis=0).astype(np.float32)
    _obs = float(img_mask_np[0].mean())
    print(
        "Mascara fija:", MASK_PATH,
        "-> %.2f%% observado (MASK_FRAC efectivo %.3f)" % (100 * _obs, 1 - _obs),
    )
else:
    img_mask = get_bernoulli_mask(img_pil, MASK_FRAC)
    img_mask_np = pil_to_np(img_mask)
img_masked = img_np * img_mask_np
mask_var = np_to_torch(img_mask_np).to(device)

# Fraccion realmente oculta (de la mascara aplicada). Con MASK_PATH puede diferir
# de MASK_FRAC; se usa para rotular figuras.
mask_frac_eff = 1.0 - float(img_mask_np[0].mean())


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
    skip_n33d=NET_WIDTH,
    skip_n33u=NET_WIDTH,
    skip_n11=SKIP_N11,
    num_scales=NUM_SCALES,
    upsample_mode=UPSAMPLE_MODE,
).to(device)

mse = torch.nn.MSELoss().to(device)
img_var = np_to_torch(img_np).to(device)
net_input = get_noise(INPUT_DEPTH, INPUT, img_np.shape[1:]).float().to(device).detach()  # meshgrid sale en float64

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


def crop_out(out):
    # Sin canales de skip (SKIP_N11=0, config "vase" del paper) la red no recorta sus
    # ramas y con un lado que no es multiplo de 2**NUM_SCALES (130 tras el pad) la salida
    # sale mas grande (160): se recorta al tamano de la imagen.
    H, W = img_var.shape[-2:]
    if out.shape[-2] >= H and out.shape[-1] >= W:
        return out[..., :H, :W]
    return out


def closure():
    global i, psrn_masked_last, last_net, net_input, n_fallbacks
    global psnr_ema, consec_fallbacks

    if REG_NOISE_STD > 0:
        net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)

    out = crop_out(net(net_input))

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
        ref = psrn_masked_last if psnr_ema is None else psnr_ema
        if (
            psrn_masked - ref < PSNR_DROP_TOL
            and last_net is not None
            and consec_fallbacks < MAX_FALLBACKS
        ):
            n_fallbacks += 1
            consec_fallbacks += 1
            print(
                "\nFalling back to previous checkpoint (#%d, %d/%d seguidos)."
                % (n_fallbacks, consec_fallbacks, MAX_FALLBACKS)
            )
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.to(device))
            return total_loss * 0

        # aceptar el estado actual: nuevo checkpoint + actualizar la media movil.
        # Al toparse con MAX_FALLBACKS se cae aca aunque el PSNR haya bajado, asi
        # un unico checkpoint malo no deja la corrida en un bucle de rollback.
        consec_fallbacks = 0
        last_net = [x.detach().cpu() for x in net.parameters()]
        psrn_masked_last = psrn_masked
        psnr_ema = (
            psrn_masked
            if psnr_ema is None
            else PSNR_EMA_ALPHA * psrn_masked + (1 - PSNR_EMA_ALPHA) * psnr_ema
        )

        out_clip = np.clip(out_np, 0, 1)
        if i % SNAPSHOT_EVERY == 0:
            np_to_pil(out_clip).save(os.path.join(SNAPSHOT_DIR, "iter_%05d.png" % i))
        ssim_full = metrics.ssim(img_np, out_clip, n_channels)
        metrics_log.append((i, float(psrn), float(psrn_masked), float(ssim_full)))

    i += 1
    return total_loss


last_net = None
psrn_masked_last = 0
psnr_ema = None
n_fallbacks = 0
consec_fallbacks = 0
i = 0

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR=LR, num_iter=NUM_ITER)

# --------------------------------------------------------------------------------
# Salida final -- pixel-exacta + copias crudas en .npy para comparacion posterior
# --------------------------------------------------------------------------------
with torch.no_grad():
    out_np = np.clip(torch_to_np(crop_out(net(net_input))), 0, 1)

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
    (axes[1], "Enmascarada (%.1f%% oculto)" % (mask_frac_eff * 100), img_masked),
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
    "MASK_FRAC=%.3f   PSNR=%.2f dB   SSIM=%.4f"
    % (mask_frac_eff, final_psnr, final_ssim)
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
    ax1.set_title("MASK_FRAC=%.3f  -  metricas vs iteracion" % mask_frac_eff)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "psnr_curve.png"), dpi=120)
    plt.close(fig)

# Hoja de contacto de la trayectoria: una sola imagen con todos los snapshots en
# grilla, en vez de abrir de a uno decenas de iter_*.png "iguales".
snaps = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "iter_*.png")))
if snaps:
    ncol = min(6, len(snaps))
    nrow = int(np.ceil(len(snaps) / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(2.1 * ncol, 2.2 * nrow), squeeze=False
    )
    for ax in axs.flat:
        ax.axis("off")
    for ax, sp in zip(axs.flat, snaps):
        ax.imshow(Image.open(sp), cmap=_cmap, vmin=0, vmax=255)
        ax.set_title("iter %d" % int(os.path.basename(sp)[5:-4]), fontsize=7)
    fig.suptitle(
        "MASK_FRAC=%.3f  -  trayectoria DIP (%d snapshots, backtracks=%d)"
        % (mask_frac_eff, len(snaps), n_fallbacks)
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "snapshots_contact.png"), dpi=110)
    plt.close(fig)

with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["iter", "psnr_full", "psnr_masked", "ssim_full"])
    w.writerows(metrics_log)
    w.writerow([])
    w.writerow(["final", final_psnr, "", final_ssim])
    w.writerow(["mae", final_mae, "", ""])
    w.writerow(["fallbacks", n_fallbacks, "", ""])

print("PSNR final (imagen completa):", final_psnr)
print("SSIM final (imagen completa):", final_ssim)
print("MAE final:", final_mae)
print("Backtracks (PSNR_DROP_TOL=%.1f dB):" % PSNR_DROP_TOL, n_fallbacks)
print("Listo. Salidas en:", OUTPUT_DIR, "(snapshots en", SNAPSHOT_DIR + ")")
