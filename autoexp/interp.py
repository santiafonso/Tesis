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


def front_split(ij, v, shape, jump=0.3, max_len=24.0, deg=1, base="rbf_tps", smoothing=0.0,
                full_width=True, split=True, along=1.0, band=8.0):
    """Interpolacion que respeta un frente abrupto (sin mirar la imagen real).

    1. Triangulacion de Delaunay de lo consultado; una arista es "cruce de frente" si
       |dv| >= jump y mide <= max_len px (los pares que el muestreo adaptativo deja a
       ambos lados).
    2. Por los puntos medios de esas aristas se ajusta el frente como polinomio de grado
       `deg`, fila = f(columna). Con full_width=True vale en todo el ancho (del lado
       derecho, donde ya no hay salto, partir no molesta); si no, solo entre la primera y
       la ultima columna con cruce (dejaba una costura vertical al final del tramo).
    3. Cada pixel (y cada punto) se etiqueta segun de que lado del frente cae (fuera del
       tramo valido, un solo lado) y se interpola cada lado solo con sus puntos.

    Si no hay cruces suficientes, cae a `base` sin partir.

    along < 1 (solo con deg=1): cerca del frente se interpola en coordenadas alineadas a
    la recta (u a lo largo, d a traves) con u escalado por `along`, asi el perfil medido
    en unas pocas columnas se traslada paralelo al frente en vez de armar "cuentas"
    entre columnas. Se mezcla con la interpolacion isotropa con peso exp(-(d/band)^2).
    split=False: no parte por lados (para frentes suaves, donde partir crea un escalon).
    """
    from scipy.spatial import Delaunay

    H, W = shape
    ij = np.asarray(ij, float)
    v = np.asarray(v, float)
    tri = Delaunay(ij)
    edges = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edges.add(tuple(sorted((s[a], s[b]))))
    mids = []
    for a, b in edges:
        if abs(v[a] - v[b]) >= jump and np.linalg.norm(ij[a] - ij[b]) <= max_len:
            mids.append((ij[a] + ij[b]) / 2)
    if len(mids) < deg + 2:
        return reconstruct(ij, v, shape, base, smoothing), None
    mids = np.array(mids)
    coef = np.polyfit(mids[:, 1], mids[:, 0], deg)
    x0, x1 = (-np.inf, np.inf) if full_width else (mids[:, 1].min(), mids[:, 1].max())

    def side(rows, cols):  # True = debajo del frente (fila mayor) dentro del tramo
        return (rows > np.polyval(coef, cols)) & (cols >= x0) & (cols <= x1)

    yy, xx = np.mgrid[0:H, 0:W]
    pix_low = side(yy, xx) if split else np.zeros((H, W), bool)
    pt_low = side(ij[:, 0], ij[:, 1]) if split else np.zeros(len(ij), bool)

    if along < 1.0 and deg == 1:
        m, c = coef  # fila = m*col + c
        nrm = np.hypot(1.0, m)
        t = np.array([m, 1.0]) / nrm  # direccion (fila, col) a lo largo del frente
        dvec = np.array([1.0, -m]) / nrm  # normal

        def fwd(P):
            P = np.asarray(P, float) - np.array([c, 0.0])
            return np.column_stack([P @ t * along, P @ dvec])

        pix = np.column_stack([yy.ravel(), xx.ravel()])
        dist = np.abs(fwd(pix)[:, 1]).reshape(H, W)
        wgt = np.exp(-(dist / band) ** 2)
    else:
        wgt = None

    out = np.zeros((H, W))
    for lab in (True, False):
        sel = pt_low == lab
        if not (pix_low == lab).any():
            continue
        if sel.sum() < 3:
            sel = np.ones_like(sel)
        full = reconstruct(ij[sel], v[sel], shape, base, smoothing)
        if wgt is not None:
            q = fwd(pix)
            s_ = max(H, W)
            kern = {"rbf_tps": "thin_plate_spline"}.get(base, "thin_plate_spline")
            ani = RBFInterpolator(fwd(ij[sel]) / s_, v[sel], kernel=kern, smoothing=smoothing)(q / s_)
            full = wgt * np.clip(ani.reshape(H, W), 0, 1) + (1 - wgt) * full
        out[pix_low == lab] = full[pix_low == lab]
    return np.clip(out, 0, 1), (coef, x0, x1)
