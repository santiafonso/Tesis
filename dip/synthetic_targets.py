#!/usr/bin/env python
# coding: utf-8
"""Imagenes sinteticas "de libro" para probar dip.restoration mas alla de los
diagramas de fase: sirven de sanity-check (si esto falla, algo esta mal en
el pipeline) y para relacionar la pregunta de "cuantos puntos hacen falta"
con la frecuencia espacial del contenido (Nyquist), no solo con el caso
particular del frente de fase.

Familias:
  gauss       campana de Gauss 2D -- caso facil/suave, piso de referencia.
  freq_sweep  rejilla sinusoidal cuya frecuencia crece de izquierda a
              derecha -- conecta directo con Nyquist: a partir de que
              frecuencia (columna) el punteo de puntos deja de alcanzar.
  rosenbrock  funcion de Rosenbrock (valle curvo y angosto rodeado de
              zonas planas) -- analogo "famoso" a la meseta+frente que ya
              se viene estudiando, pero sin ser un caso ad-hoc.
  himmelblau  funcion de Himmelblau (4 minimos separados) -- estructura
              multi-modal, mas texturada que un solo frente.
  checkerboard tablero de ajedrez -- alta frecuencia uniforme, el peor caso
              posible para cualquier distribucion de puntos.

Uso:
    ./venv/bin/python -m dip.synthetic_targets

Escribe en data/restoration/synthetic/:
    gauss.png  freq_sweep.png  rosenbrock.png  himmelblau.png  checkerboard.png
(todas 128x128, 8-bit grises, normalizadas a [0,255])

Cada una despues se usa igual que cualquier otra imagen:
    IMAGE_PATH=data/restoration/synthetic/gauss.png \
    MASK_FRAC=0.98 ./venv/bin/python -m dip.restoration
"""
import os

import numpy as np
from PIL import Image

OUT_DIR = os.environ.get("OUT_DIR", "data/restoration/synthetic")
RES = int(os.environ.get("RES", "128"))

os.makedirs(OUT_DIR, exist_ok=True)


def _normalize(a):
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / max(hi - lo, 1e-12)


def _save(name, field):
    img = (np.clip(_normalize(field), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(os.path.join(OUT_DIR, "%s.png" % name))
    print("Escrito %s.png  shape=%s" % (name, img.shape))


def make_gauss(res):
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx ** 2 + yy ** 2) / 2.0)


def make_freq_sweep(res, f_min=0.02, f_max=0.5):
    x = np.arange(res)
    y = np.arange(res)
    xx, yy = np.meshgrid(x, y)
    freq = f_min + (f_max - f_min) * (xx / max(res - 1, 1))
    return np.sin(2 * np.pi * freq * yy)


def make_rosenbrock(res, a=1.0, b=100.0):
    x = np.linspace(-2, 2, res)
    y = np.linspace(-1, 3, res)
    xx, yy = np.meshgrid(x, y)
    z = (a - xx) ** 2 + b * (yy - xx ** 2) ** 2
    return -np.log1p(z)  # log-invertido: el valle queda "alto" (blanco), no un pico agudo


def make_himmelblau(res):
    x = np.linspace(-5, 5, res)
    y = np.linspace(-5, 5, res)
    xx, yy = np.meshgrid(x, y)
    z = (xx ** 2 + yy - 11) ** 2 + (xx + yy ** 2 - 7) ** 2
    return -np.log1p(z)


def make_checkerboard(res, n_cells=8):
    step = max(1, res // n_cells)
    xx, yy = np.meshgrid(np.arange(res), np.arange(res))
    return ((xx // step) + (yy // step)) % 2


def main():
    _save("gauss", make_gauss(RES))
    _save("freq_sweep", make_freq_sweep(RES))
    _save("rosenbrock", make_rosenbrock(RES))
    _save("himmelblau", make_himmelblau(RES))
    _save("checkerboard", make_checkerboard(RES))
    print("Listo en", OUT_DIR)


if __name__ == "__main__":
    main()
