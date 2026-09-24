"""Reconstruccion clasica por interpolacion (scipy) -- MODIFICABLE por el loop.

Dos usos:
  1. Baseline "interpolacion 2D de scipy vs DIP" (punto 6 de la reunion del 17/9).
     Nota: `scipy.interpolate.interp2d` (el del link) ya no existe en scipy moderno;
     `griddata` y `RBFInterpolator` son sus reemplazos para puntos dispersos.
  2. Fuente de pseudo-puntos para DIP (`pseudo_points`): valores inventados a partir
     de lo observado, sin mirar la imagen real.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator, griddata

METHODS = ("nearest", "linear", "cubic", "rbf_tps", "rbf_linear", "rbf_cubic", "rbf_gauss")


def reconstruct(ij, v, shape, method="rbf_tps", smoothing=0.0, epsilon=None):
    """ij (N,2) pixeles consultados, v (N,) valores -> mapa (H,W) en [0,1]."""
    H, W = shape
    ij = np.asarray(ij, float)
    v = np.asarray(v, float)
    yy, xx = np.mgrid[0:H, 0:W]
    q = np.column_stack([yy.ravel(), xx.ravel()])
    if method in ("nearest", "linear", "cubic"):
        out = griddata(ij, v, q, method=method)
        if method != "nearest":  # fuera del casco convexo -> vecino mas cercano
            nn = griddata(ij, v, q, method="nearest")
            out = np.where(np.isnan(out), nn, out)
    else:
        kernel = {"rbf_tps": "thin_plate_spline", "rbf_linear": "linear",
                  "rbf_cubic": "cubic", "rbf_gauss": "gaussian"}[method]
        s = max(H, W)  # coordenadas normalizadas a ~[0,1]
        kw = {"kernel": kernel, "smoothing": smoothing}
        if kernel == "gaussian":
            kw["epsilon"] = epsilon or 8.0
        out = RBFInterpolator(ij / s, v, **kw)(q / s)
    return np.clip(out.reshape(H, W), 0.0, 1.0)


def pseudo_points(ij, v, shape, n_pseudo=256, method="rbf_tps", where="plateau", grad_q=0.5,
                  min_dist=3.0):
    """Pseudo-observaciones para DIP a partir de lo consultado (no gastan presupuesto).

    Se ponen en una grilla regular de ~n_pseudo pixeles, descartando los que estan a
    menos de `min_dist` px de un punto real. Con where="plateau" solo se quedan los de
    las zonas donde el interpolante es plano (|grad| por debajo del cuantil `grad_q`):
    ahi la interpolacion es confiable, y cerca de la frontera se deja que decida DIP.
    Con where="all" se usan todos.

    Devuelve (pseudo_ij (M,2) int, pseudo_v (M,), mapa interpolado (H,W)).
    """
    H, W = shape
    est = reconstruct(ij, v, shape, method)
    step = max(1, int(round(np.sqrt(H * W / max(n_pseudo, 1)))))
    yy, xx = np.mgrid[step // 2:H:step, step // 2:W:step]
    cand = np.column_stack([yy.ravel(), xx.ravel()])
    ij = np.asarray(ij, int)
    if len(ij):
        d = np.sqrt(((cand[:, None, :] - ij[None, :, :]) ** 2).sum(-1)).min(1)
        cand = cand[d >= min_dist]
    if where == "plateau":
        gy, gx = np.gradient(est)
        gm = np.hypot(gx, gy)
        thr = np.quantile(gm, grad_q)
        cand = cand[gm[cand[:, 0], cand[:, 1]] <= thr]
    return cand, est[cand[:, 0], cand[:, 1]], est
