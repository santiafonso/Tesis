"""Codigo propio de la tesis para el pipeline Deep Image Prior (DIP).

No confundir con el fork upstream de deep-image-prior, que vive en `models/` y
`utils/` en la raiz del repo y no se toca.

Modulos:
  - inpaint        : runner unificado de inpainting DIP (gris o RGB) sobre una
                     imagen con mascara Bernoulli. Reemplaza a los antiguos
                     restoration.py / restorationGRIS.py / restorationRGB.py.
  - metrics        : PSNR / SSIM / MAE y el mapa de error absoluto.
  - phase_diagram  : genera el diagrama de fases SoC_max(log Xi, log l) con el
                     modelo del continuo (galpynostatic).
  - sample_points  : elige un subconjunto disperso de puntos (xi, l) de la
                     grilla del diagrama de fases para correr KMC real.

Todos se ejecutan como modulo desde la raiz del repo, p. ej.:
    ./venv/bin/python -m dip.inpaint
"""
