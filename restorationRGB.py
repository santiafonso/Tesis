#!/usr/bin/env python
# coding: utf-8
"""Deep Image Prior — inpainting RGB (mapa_suave) con salidas a color."""

from __future__ import print_function

import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from models import get_net
from utils.common_utils import crop_image, get_image_grid, get_params, optimize, pil_to_np
from utils.inpainting_utils import get_bernoulli_mask, get_noise, np_to_pil, np_to_torch, torch_to_np

# --- Config (sobreescribible desde el cluster con variables de entorno) ---
IMAGE_PATH = os.environ.get("IMAGE_PATH", "./data/restoration/mapa_suave.png")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results_rgb")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "0"))  # 0 = sin redimensionar; ej. 512 para pruebas
MASK_FRAC = float(os.environ.get("MASK_FRAC", "0.50"))
NUM_ITER = int(os.environ.get("NUM_ITER", "11000"))
LR = float(os.environ.get("LR", "0.001"))
REG_NOISE_STD = float(os.environ.get("REG_NOISE_STD", "0.03"))
SHOW_EVERY = int(os.environ.get("SHOW_EVERY", "100"))
SEED = int(os.environ.get("SEED", "42"))

PLOT = True
INPUT = "noise"
INPUT_DEPTH = 32
PAD = "reflection"
OPTIMIZER = "adam"
OPT_OVER = "net"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rgb_image(path, max_side=0):
    """RGBA/RGB → RGB, recorte divisible por 64, opcional downscale."""
    img = Image.open(path).convert("RGB")
    if max_side > 0:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = crop_image(img, 64)
    return img, pil_to_np(img)


def save_image_grid(images_np, path, nrow=1, factor=5, dpi=150):
    """Guarda mosaico RGB/escala de grises sin plt.show (apt para SLURM)."""
    n_channels = max(x.shape[0] for x in images_np)
    assert n_channels in (1, 3)
    images_np = [
        x if x.shape[0] == n_channels else np.concatenate([x, x, x], axis=0)
        for x in images_np
    ]
    grid = get_image_grid(images_np, nrow)
    h, w = grid.shape[1], grid.shape[2]
    fig_w = max(4, w / 128 * factor * nrow)
    fig_h = max(4, h / 128 * factor)
    plt.figure(figsize=(fig_w, fig_h))
    if n_channels == 1:
        plt.imshow(np.clip(grid[0], 0, 1), cmap="gray", interpolation="lanczos")
    else:
        plt.imshow(np.clip(grid.transpose(1, 2, 0), 0, 1), interpolation="lanczos")
    plt.axis("off")
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def psnr_rgb(reference, estimate, mask=None):
    """PSNR para tensores C×H×W en [0, 1]."""
    ref = np.clip(reference, 0, 1)
    est = np.clip(estimate, 0, 1)
    if mask is not None:
        ref = ref * mask
        est = est * mask
    try:
        return compare_psnr(ref, est, data_range=1.0, channel_axis=0)
    except TypeError:
        return compare_psnr(ref, est, data_range=1.0, multichannel=True)


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = get_device()
print("Dispositivo:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

# --- Carga imagen a color ---
img_pil, img_np = load_rgb_image(IMAGE_PATH, max_side=MAX_SIDE)
print("Imagen:", IMAGE_PATH, "→", img_np.shape, "(C×H×W)")

img_np = nn.ReflectionPad2d(1)(np_to_torch(img_np).to(device))[0].cpu().numpy()
img_pil = np_to_pil(img_np)

img_mask = get_bernoulli_mask(img_pil, MASK_FRAC)
img_mask_np = pil_to_np(img_mask)
img_masked = img_np * img_mask_np

img_var = np_to_torch(img_np).to(device)
mask_var = np_to_torch(img_mask_np).to(device)
mse = torch.nn.MSELoss().to(device)

save_image_grid(
    [img_np, img_mask_np, img_masked],
    os.path.join(OUTPUT_DIR, "mask.png"),
    nrow=3,
    factor=4,
)

# --- Red 3 canales (mismo esquema skip que barbara, salida RGB) ---
net = get_net(
    INPUT_DEPTH,
    "skip",
    PAD,
    n_channels=3,
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    upsample_mode="bilinear",
).to(device)

net_input = get_noise(INPUT_DEPTH, INPUT, img_np.shape[1:]).to(device).detach()
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

last_net = None
psrn_masked_last = 0.0
i = 0


def closure():
    global i, psrn_masked_last, last_net, net_input

    if REG_NOISE_STD > 0:
        net_input = net_input_saved + noise.normal_() * REG_NOISE_STD

    out = net(net_input)
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    out_np = torch_to_np(out)
    psrn_masked = psnr_rgb(img_np, out_np, img_mask_np)
    psrn = psnr_rgb(img_np, out_np)

    print(
        "Iteration %05d    Loss %f PSNR_masked %f PSNR %f"
        % (i, total_loss.item(), psrn_masked, psrn),
        end="\r",
    )

    if PLOT and i % SHOW_EVERY == 0:
        if psrn_masked - psrn_masked_last < -5 and last_net is not None:
            print("\nFalling back to previous checkpoint.")
            for saved, param in zip(last_net, net.parameters()):
                param.data.copy_(saved.to(device))
            return total_loss * 0
        last_net = [p.detach().cpu() for p in net.parameters()]
        psrn_masked_last = psrn_masked

        save_image_grid(
            [np.clip(out_np, 0, 1)],
            os.path.join(OUTPUT_DIR, "iter_%05d.png" % i),
            factor=6,
        )

    i += 1
    return total_loss


print("Iniciando optimización (%d iteraciones)…" % NUM_ITER)
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR=LR, num_iter=NUM_ITER)

with torch.no_grad():
    out_np = np.clip(torch_to_np(net(net_input)), 0, 1)

save_image_grid(
    [out_np, img_np],
    os.path.join(OUTPUT_DIR, "final.png"),
    nrow=2,
    factor=8,
    dpi=200,
)
save_image_grid([out_np], os.path.join(OUTPUT_DIR, "restored_color.png"), factor=10, dpi=200)

np.save(os.path.join(OUTPUT_DIR, "restored.npy"), out_np)
print("\nListo. Salidas en:", OUTPUT_DIR)
print("PSNR final:", psnr_rgb(img_np, out_np))
