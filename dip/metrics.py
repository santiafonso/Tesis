"""Metricas de calidad de reconstruccion usadas por dip.inpaint.

Nada nuevo respecto de restorationGRIS.py: son los mismos calculos que ya estaban
inline, movidos a un solo lugar para poder leerlos/testearlos aparte.

  - PSNR  : skimage.metrics.peak_signal_noise_ratio  (MSE -> dB)
  - SSIM  : skimage.metrics.structural_similarity     (ventana local ~7x7)
  - MAE   : a mano con numpy, np.abs(orig - rec).mean()

Convencion: todos los arrays son C x H x W en [0, 1] (float), como los devuelve
utils.common_utils.pil_to_np / torch_to_np.
"""
from __future__ import print_function

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _compare_psnr
from skimage.metrics import structural_similarity as _compare_ssim


def psnr(reference, estimate):
    """PSNR sobre la imagen completa (C x H x W en [0, 1]).

    Identico a `compare_psnr(img_np, out_np)` del script original: sin pasar
    data_range explicito (skimage lo infiere de [0, 1] para float).
    """
    return _compare_psnr(reference, estimate)


def psnr_masked(reference_masked, estimate_masked):
    """PSNR calculado solo sobre los pixeles observados.

    El caller pasa los arrays ya enmascarados, igual que antes:
        psnr_masked(img_masked, out_np * img_mask_np)
    """
    return _compare_psnr(reference_masked, estimate_masked)


def ssim(reference, estimate, n_channels=1):
    """SSIM sobre la imagen completa.

    - 1 canal : `compare_ssim(ref[0], est[0], data_range=1.0)`  (como el original)
    - 3 canales: se pasa H x W x C con channel_axis para el termino multicanal.
    """
    if n_channels == 1:
        return _compare_ssim(reference[0], estimate[0], data_range=1.0)
    ref_hwc = np.moveaxis(reference, 0, -1)
    est_hwc = np.moveaxis(estimate, 0, -1)
    try:
        return _compare_ssim(ref_hwc, est_hwc, data_range=1.0, channel_axis=-1)
    except TypeError:  # skimage viejo
        return _compare_ssim(ref_hwc, est_hwc, data_range=1.0, multichannel=True)


def abs_error_map(reference, estimate):
    """Mapa |orig - rec| por pixel (mismo shape C x H x W)."""
    return np.abs(reference - estimate)


def mae(reference, estimate):
    """Error absoluto medio: promedio de abs_error_map. En unidades de la imagen."""
    return float(np.abs(reference - estimate).mean())
