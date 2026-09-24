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
                full_width=True, split="auto", along=1.0, band=8.0):
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
    # cruces: aristas cortas que quedan a ambos lados del nivel medio (sirve para frentes
    # abruptos y para rampas suaves); el punto de cruce se interpola sobre la arista.
    level = 0.5 * (v.min() + v.max())
    mids, slopes = [], []
    for a, b in edges:
        L = np.linalg.norm(ij[a] - ij[b])
        if L <= max_len and (v[a] - level) * (v[b] - level) < 0:
            f = (level - v[a]) / (v[b] - v[a])
            mids.append(ij[a] + f * (ij[b] - ij[a]))
            slopes.append((abs(v[a] - v[b]), L))
    if len(mids) < deg + 2:
        return reconstruct(ij, v, shape, base, smoothing), None
    mids = np.array(mids)
    if split == "auto":  # abrupto si en los corchetes mas cortos el salto sigue siendo grande
        sl = sorted(slopes, key=lambda x: x[1])[: max(3, len(slopes) // 3)]
        split = bool(np.median([dv for dv, _ in sl]) >= jump)
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


def cliff(ij, v, shape, eps=0.004, steep_min=0.1, steep_r=5.0, max_len=4.0, deg=1, base="rbf_tps",
          smoothing=0.0, outlier_px=3.0, along=0.5, band=15.0, vert=0.6, mono=True,
          adapt=True):
    """Reconstruccion con acantilado a cero (sin mirar la imagen real).

    Modelo: debajo del acantilado el mapa vale 0; arriba es suave. El acantilado se ubica
    con las aristas cortas (<= max_len px) de la triangulacion que unen un punto nulo
    (v < eps) con uno no nulo; su punto medio es un punto del borde. Se ajusta
    fila = f(columna) (polinomio de grado `deg`) y:
      - pixeles con fila > f(col): 0
      - resto: `base` interpolando SOLO los puntos de arriba del acantilado.
    Solo cuentan como borde los pares donde, a <= steep_r px del lado no nulo, algun punto
    consultado vale >= steep_min (el ultimo pixel antes del 0 puede valer 0.02 en un
    acantilado real; lo que lo distingue de la cola suave de la zona que se desvanece es
    que un poco mas arriba ya es alto). El ajuste se
    repite una vez sin los puntos a mas de outlier_px de la primera recta.
    """
    from scipy.spatial import Delaunay

    H, W = shape
    ij = np.asarray(ij, float)
    v = np.asarray(v, float)
    tri = Delaunay(ij)
    edges = set()
    for s_ in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            edges.add(tuple(sorted((s_[a], s_[b]))))
    def steep(a, b):  # a o b nulo; cerca del no nulo el mapa ya es alto -> acantilado
        nz = b if v[a] < eps else a
        near = np.linalg.norm(ij - ij[nz], axis=1) <= steep_r
        return v[near].max() >= steep_min

    mids = [(ij[a] + ij[b]) / 2 for a, b in edges
            if (v[a] < eps) != (v[b] < eps) and np.linalg.norm(ij[a] - ij[b]) <= max_len
            and steep(a, b)]
    if len(mids) < deg + 2:
        return reconstruct(ij, v, shape, base, smoothing), None
    mids = np.array(mids)
    coef = np.polyfit(mids[:, 1], mids[:, 0], deg)
    res = np.abs(mids[:, 0] - np.polyval(coef, mids[:, 1]))
    if (res <= outlier_px).sum() >= deg + 2:  # ajuste robusto: un reajuste sin outliers
        coef = np.polyfit(mids[res <= outlier_px, 1], mids[res <= outlier_px, 0], deg)
    yy, xx = np.mgrid[0:H, 0:W]
    below = yy > np.polyval(coef, xx)
    up = ~(ij[:, 0] > np.polyval(coef, ij[:, 1]))
    if vert < 1.0:
        # lejos del borde la estructura es casi solo funcion de x (la transicion vertical de
        # la zona que se desvanece): interpolar con la distancia vertical comprimida por `vert`
        s_ = max(H, W)
        sc = np.array([vert, 1.0]) / s_
        q = np.column_stack([yy.ravel(), xx.ravel()]) * sc
        rec = np.clip(RBFInterpolator(ij[up] * sc, v[up], kernel="thin_plate_spline",
                                      smoothing=smoothing)(q).reshape(H, W), 0, 1)
    else:
        rec = reconstruct(ij[up], v[up], shape, base, smoothing)
    if adapt:
        adapt = {} if adapt is True else adapt
        # franja adaptativa: el valor justo arriba del acantilado dice que tan abrupto es el
        # frente (g=-4: ~0.8, salto seco; g suaves: ~0.02-0.4, la rampa ya bajo casi todo).
        # Frente suave -> franja ancha y mas comprimida a lo largo del borde.
        dist_pts = np.polyval(coef, ij[:, 1]) - ij[:, 0]
        edge = v[(dist_pts > 0) & (dist_pts <= 2.5) & (v >= eps)]
        v_edge = float(np.median(edge)) if len(edge) else 0.8
        f = float(np.clip(v_edge / adapt.get("v_sharp", 0.7), 0.0, 1.0))  # 1 = abrupto
        along = adapt.get("along_soft", 0.3) + f * (adapt.get("along_sharp", 0.5) - adapt.get("along_soft", 0.3))
        band = adapt.get("band_soft", 25.0) + f * (adapt.get("band_sharp", 15.0) - adapt.get("band_soft", 25.0))
    if along < 1.0 and deg == 1:
        # la rampa previa al acantilado se traslada paralela al borde: cerca de el,
        # interpolar en (u*along, d) con u a lo largo del borde y d la distancia a el;
        # lejos (> band), la interpolacion isotropa (ahi hay estructura no alineada, como
        # la transicion vertical de la zona que se desvanece).
        m, c = coef
        nrm = np.hypot(1.0, m)
        t_ = np.array([m, 1.0]) / nrm
        n_ = np.array([1.0, -m]) / nrm

        def fwd(P):
            P = np.asarray(P, float) - np.array([c, 0.0])
            return np.column_stack([P @ t_ * along, P @ n_])

        pix = np.column_stack([yy.ravel(), xx.ravel()])
        q = fwd(pix)
        s_ = max(H, W)
        ani = RBFInterpolator(fwd(ij[up]) / s_, v[up], kernel="thin_plate_spline",
                              smoothing=smoothing)(q / s_).reshape(H, W)
        w = np.exp(-(np.abs(q[:, 1]).reshape(H, W) / band) ** 2)
        rec = w * ani + (1 - w) * rec
    rec[below] = 0.0
    if mono:
        rec = monotone_2d(np.clip(rec, 0, 1))
        rec[below] = 0.0
    return np.clip(rec, 0, 1), (coef, mids)


def _pava_dec(y):
    """Regresion isotonica no creciente de una secuencia (pool adjacent violators)."""
    vals, wts, lens = [], [], []
    for x in y:
        vals.append(float(x)); wts.append(1.0); lens.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n = lens[-2] + lens[-1]
            vals[-2:] = [v]; wts[-2:] = [w]; lens[-2:] = [n]
    return np.repeat(vals, lens)


def monotone_2d(img, iters=100):
    """Proyeccion (Dykstra) sobre mapas no crecientes hacia abajo (filas) y hacia la
    derecha (columnas). El diagrama real cumple esto exacto en todos los g (SoC_max baja
    al crecer l y Xi): cualquier ondulacion de la interpolacion lo viola."""
    x = img.astype(float).copy()
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    for _ in range(iters):
        y = x + p
        yc = np.column_stack([_pava_dec(y[:, j]) for j in range(y.shape[1])])
        p = y - yc
        z = yc + q
        zr = np.vstack([_pava_dec(z[i, :]) for i in range(z.shape[0])])
        q = z - zr
        if np.abs(zr - x).max() < 1e-6:
            x = zr
            break
        x = zr
    return x
