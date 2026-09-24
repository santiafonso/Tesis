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

    if sigma:
        np_ = noise_params(sigma)
        eps, steep_min, smoothing = np_["eps"], np_["steep_min"], np_["smoothing"]
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


def _robust_line(mids, deg, outlier_px):
    """fila = f(col). Con deg=1: RANSAC exhaustivo (todas las rectas por pares de puntos de
    borde; gana la de mas inliers a <= outlier_px, desempate por residuo) y reajuste por
    minimos cuadrados con los inliers. Con ruido, la cola suave de la zona que se desvanece
    genera pares 0/no-0 falsos que un solo reajuste no alcanzaba a sacar."""
    # bordes a < 2 px entre si cuentan una sola vez (un grupo de bordes falsos no suma votos)
    keep = []
    for k in range(len(mids)):
        if all(np.hypot(*(mids[k] - mids[q])) >= 2.0 for q in keep):
            keep.append(k)
    mids = mids[keep]
    x, y = mids[:, 1], mids[:, 0]
    if deg != 1 or len(mids) < 4:
        coef = np.polyfit(x, y, deg)
        res = np.abs(y - np.polyval(coef, x))
        if (res <= outlier_px).sum() >= deg + 2:
            coef = np.polyfit(x[res <= outlier_px], y[res <= outlier_px], deg)
        return coef
    best, best_key = None, None
    for a in range(len(mids)):
        for b in range(a + 1, len(mids)):
            if abs(x[b] - x[a]) < 1e-9:
                continue
            m = (y[b] - y[a]) / (x[b] - x[a])
            c = y[a] - m * x[a]
            res = np.abs(y - (m * x + c))
            inl = res <= outlier_px
            key = (inl.sum(), -res[inl].sum())
            if best_key is None or key > best_key:
                best, best_key = inl, key
    return np.polyfit(x[best], y[best], 1)


def noise_params(sigma):
    """Umbrales robustos a ruido a partir de un unico sigma (ruido por punto, en KMC se estima
    con corridas repetidas). sigma=0 -> los valores validados sin ruido."""
    if not sigma:
        return {}
    return {"eps": max(0.004, 3 * sigma), "steep_min": max(0.1, 5 * sigma), "smoothing": 1e-4}


def cliff(ij, v, shape, eps=0.004, steep_min=0.1, steep_r=5.0, max_len=4.0, deg=1, base="rbf_tps",
          smoothing=1e-4, outlier_px=3.0, along=0.5, band=15.0, vert=0.6, mono=True,
          adapt=True, bounds=True, sigma=0.0):
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

    if sigma:
        np_ = noise_params(sigma)
        eps, steep_min, smoothing = np_["eps"], np_["steep_min"], np_["smoothing"]
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
    coef = _robust_line(mids, deg, outlier_px)
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
    if bounds:  # recorte a las cotas exactas por monotonia, antes de proyectar
        L, U = monotone_bounds(ij, v, shape)
        rec = np.minimum(np.maximum(rec, L), U)
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


def monotone_bounds(ij, v, shape):
    """Cotas exactas por monotonia (mapa no creciente hacia abajo y a la derecha):
    un punto consultado q abajo-derecha de p (q_i >= p_i, q_j >= p_j) da v(q) <= f(p);
    uno arriba-izquierda da f(p) <= v(q). Devuelve (L, U) de shape (H, W)."""
    H, W = shape
    M = np.zeros((H, W))
    N = np.ones((H, W))
    ij = np.asarray(ij, int)
    for (i, j), x in zip(ij, v):
        M[i, j] = max(M[i, j], x)
        N[i, j] = min(N[i, j], x)
    L = np.maximum.accumulate(np.maximum.accumulate(M[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    U = np.minimum.accumulate(np.minimum.accumulate(N, axis=0), axis=1)
    return L, U


def monotone_violations(ij, v, tol):
    """Pares de puntos consultados que violan 'no creciente hacia abajo y a la derecha' por
    mas de tol (p arriba-izquierda de q con v(p) < v(q) - tol)."""
    ij = np.asarray(ij, int)
    v = np.asarray(v, float)
    le = (ij[:, None, 0] <= ij[None, :, 0]) & (ij[:, None, 1] <= ij[None, :, 1])
    np.fill_diagonal(le, False)
    return int((le & (v[:, None] < v[None, :] - tol)).sum())


def auto(ij, v, shape, sigma=0.0, tol=0.02, loo_verts=(1.0, 0.6), verbose=False):
    """Reconstruccion SIN suponer la forma de antemano: cada supuesto del modelo de
    acantilado se chequea con los propios puntos y, si no se cumple, se apaga.

    - monotonia (proyeccion + cotas): solo si ningun par de puntos la viola por mas de
      tol + 3 sigma.
    - acantilado a 0: solo si hay >= 3 puntos debajo de la recta y >= 90 % valen ~0.
    - compresion vertical: la que de menor error de validacion cruzada (LOO) sobre los puntos.
    Si no hay acantilado valido: TPS comun (suavizado 1e-4), + monotonia si corresponde.
    Devuelve (reconstruccion, dict con las decisiones)."""
    ij = np.asarray(ij, float)
    v = np.asarray(v, float)
    eps = max(0.004, 3 * sigma)
    mono_ok = monotone_violations(ij, v, tol + 3 * sigma) == 0
    info_d = {"mono": mono_ok}
    base_kw = dict(sigma=sigma, mono=mono_ok, bounds=mono_ok)
    rec, info = cliff(ij, v, shape, **base_kw)
    cliff_ok = False
    if info is not None:
        below = (ij[:, 0] - np.polyval(info[0], ij[:, 1])) > 1.0
        cliff_ok = below.sum() >= 3 and float(np.mean(v[below] < eps)) >= 0.9
    info_d["cliff"] = cliff_ok
    if not cliff_ok:
        rec = reconstruct(ij, v, shape, "rbf_tps", smoothing=1e-4)
        if mono_ok:
            L, U = monotone_bounds(ij, v, shape)
            rec = monotone_2d(np.clip(np.minimum(np.maximum(rec, L), U), 0, 1))
        info_d["vert"] = None
        if verbose:
            print("   auto:", info_d)
        return np.clip(rec, 0, 1), info_d
    # compresion vertical por LOO (sin mono/cotas: solo para rankear, es mucho mas rapido)
    best_v, best_e = 1.0, np.inf
    for vt in loo_verts:
        errs = []
        for k in range(len(ij)):
            m = np.ones(len(ij), bool)
            m[k] = False
            r_k, _ = cliff(ij[m], v[m], shape, sigma=sigma, mono=False, bounds=False, vert=vt)
            errs.append((r_k[int(ij[k, 0]), int(ij[k, 1])] - v[k]) ** 2)
        e = float(np.mean(errs))
        if e < best_e:
            best_v, best_e = vt, e
    info_d["vert"] = best_v
    rec, _ = cliff(ij, v, shape, vert=best_v, **base_kw)
    if verbose:
        print("   auto:", info_d)
    return rec, info_d
