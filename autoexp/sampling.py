"""Estrategias de muestreo -- MODIFICABLE por el loop.

Regla: solo se usa `oracle.query(...)` / `oracle.observed()`. Nada de leer la imagen
real ni el soc.npy (eso es lo que hacia frontier_mix: sirve de cota, no es desplegable
con KMC real).

Cada estrategia gasta a lo sumo `n` consultas del oraculo y no devuelve nada: lo
observado queda en el oraculo.
"""
import numpy as np

from autoexp import interp


def grid(oracle, n, nx=None, offset=0.5, margin=0.0):
    """Grilla regular nx x ny con ny = n // nx (por defecto casi cuadrada).

    offset: posicion dentro de cada celda (0.5 = centro). margin: fraccion de la
    imagen que se deja libre en cada borde (0 = celdas que cubren todo).
    """
    H, W = oracle.shape
    nx = nx or int(round(np.sqrt(n * W / H)))
    nx = max(1, min(nx, n))
    ny = max(1, n // nx)
    xs = margin * W + (np.arange(nx) + offset) * (W * (1 - 2 * margin)) / nx
    ys = margin * H + (np.arange(ny) + offset) * (H * (1 - 2 * margin)) / ny
    pts = [(min(H - 1, int(y)), min(W - 1, int(x))) for y in ys for x in xs]
    oracle.query(pts[: oracle.remaining])


def _halton(k, base):
    f, r = 1.0, 0.0
    while k > 0:
        f /= base
        r += f * (k % base)
        k //= base
    return r


def halton(oracle, n, skip=1):
    """Secuencia de Halton (bases 2, 3): cubre parejo sin la regularidad de la grilla."""
    H, W = oracle.shape
    pts, k = [], skip
    while len(set(pts)) < n:
        pts.append((min(H - 1, int(_halton(k, 3) * H)), min(W - 1, int(_halton(k, 2) * W))))
        k += 1
    oracle.query(list(dict.fromkeys(pts))[:n])


def uniform(oracle, n, seed=0):
    H, W = oracle.shape
    rng = np.random.default_rng(seed)
    flat = rng.choice(H * W, size=n, replace=False)
    oracle.query([(f // W, f % W) for f in flat])


def adaptive(oracle, n, n1=32, batch=8, first="grid", recon="rbf_tps", power=1.0, **kw):
    """Muestreo activo en dos etapas (como se haria con KMC real, por tandas):

    1. `n1` puntos con la estrategia `first` (grid/halton) para una idea global.
    2. El resto, en tandas de `batch`: se interpola lo observado, y cada punto nuevo
       va donde el mapa interpolado cambia mas (|grad|, la frontera estimada) Y lejos
       de lo ya consultado:  score = |grad|^power * distancia_al_punto_mas_cercano.
       Greedy dentro de la tanda (cada elegido "tapa" su zona para el siguiente).
    """
    H, W = oracle.shape
    if n1 > 0:
        {"grid": grid, "halton": halton}[first](oracle, min(n1, n))
    yy, xx = np.mgrid[0:H, 0:W]
    while oracle.n_used < n:
        ij, v = oracle.observed()
        est = interp.reconstruct(ij, v, oracle.shape, recon)
        gy, gx = np.gradient(est)
        gmag = np.hypot(gx, gy)
        gmag = gmag / max(gmag.max(), 1e-12)
        d2 = np.full((H, W), np.inf)
        for i, j in ij:
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        new = []
        for _ in range(min(batch, n - oracle.n_used)):
            score = (gmag + 1e-3) ** power * np.sqrt(d2)
            k = int(np.argmax(score))
            i, j = divmod(k, W)
            new.append((i, j))
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        oracle.query(new)


def bisect(oracle, n, n1=36, jump=0.25, tol=1, nx=None, offset=0.5, fill="adaptive", target="mid", eps=0.004,
           cliff_min=0.05, **kw):
    """Grilla gruesa + busqueda binaria vertical del frente en cada columna con salto.

    1. `n1` puntos en grilla (nx columnas).
    2. En cada columna de la grilla, cada par de filas vecinas con |dv| >= jump es un
       "corchete" donde cruza el frente. Se biseca por turnos (una consulta por corchete
       por ronda, asi un presupuesto corto se reparte parejo) hasta que el corchete
       mide <= tol px. El umbral de cada corchete es el punto medio de sus dos extremos.
    3. Lo que sobre, con `fill` (adaptive: |grad| x distancia, como `adaptive`).

    target="zero": en vez del nivel medio, se biseca el ACANTILADO -- el borde entre
    valor exactamente 0 (v < eps; el PNG es de 8 bit, 1/255 = 0.0039) y no-cero, solo
    donde el lado no nulo es >= cliff_min (en la zona que se desvanece arriba a la
    derecha los valores bajan suave hasta ~0.02 y eso no es un acantilado). En todos los g el perfil vertical es meseta
    -> rampa suave -> salto a 0; el nivel medio caia dentro de la rampa.
    """
    grid(oracle, min(n1, n), nx=nx, offset=offset)
    ij, v = oracle.observed()
    val = {(int(i), int(j)): x for (i, j), x in zip(ij, v)}
    ys = sorted(set(ij[:, 0].tolist()))
    br = []
    for x in sorted(set(ij[:, 1].tolist())):
        col = [(y, val[(y, x)]) for y in ys if (y, x) in val]
        for (y0, v0), (y1, v1) in zip(col, col[1:]):
            if target == "zero":
                if (v0 < eps) != (v1 < eps) and max(v0, v1) >= cliff_min:
                    br.append([x, y0, v0, y1, v1, eps])
            elif abs(v1 - v0) >= jump:
                br.append([x, y0, v0, y1, v1, (v0 + v1) / 2])
    while oracle.remaining > 0 and any(b[3] - b[1] > tol for b in br):
        for b in br:
            if oracle.remaining == 0:
                break
            if b[3] - b[1] <= tol:
                continue
            ym = (b[1] + b[3]) // 2
            vm = oracle.query([(ym, b[0])])[0]
            if (vm >= b[5]) == (b[2] >= b[5]):
                b[1], b[2] = ym, vm
            else:
                b[3], b[4] = ym, vm
    if oracle.remaining > 0 and fill == "adaptive":
        adaptive(oracle, n, n1=0, **kw)
    elif oracle.remaining > 0 and fill == "cliff":
        cliff_fill(oracle, n, **kw)
    elif oracle.remaining > 0 and fill == "loo":
        loo_fill(oracle, n, **kw)
    elif oracle.remaining > 0 and fill == "mix":  # mitad adaptativo (pegado al borde), mitad cliff
        adaptive(oracle, oracle.n_used + oracle.remaining // 2, n1=0)
        cliff_fill(oracle, n, **kw)


def cliff_fill(oracle, n, batch=4, power=1.0, gap=2.0, **kw):
    """Relleno para el modelo de acantilado: como `adaptive`, pero el |grad| sale de
    interp.cliff y solo del lado de arriba (el acantilado ya esta ubicado; lo que falta es
    la estructura suave, ej. la transicion vertical de la zona que se desvanece)."""
    H, W = oracle.shape
    yy, xx = np.mgrid[0:H, 0:W]
    while oracle.n_used < n:
        ij, v = oracle.observed()
        est, info = interp.cliff(ij, v, oracle.shape, **dict(kw, mono=False))
        gy, gx = np.gradient(est)
        gmag = np.hypot(gx, gy)
        if info is not None:
            below = yy > np.polyval(info[0], xx)
            # afuera el acantilado y una franja de `gap` px arriba (el salto contamina el grad)
            gmag[below | (yy > np.polyval(info[0], xx) - gap)] = 0.0
        gmag = gmag / max(gmag.max(), 1e-12)
        d2 = np.full((H, W), np.inf)
        for i, j in ij:
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        new = []
        for _ in range(min(batch, n - oracle.n_used)):
            k = int(np.argmax((gmag + 1e-3) ** power * np.sqrt(d2)))
            i, j = divmod(k, W)
            new.append((i, j))
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        oracle.query(new)


def loo_fill(oracle, n, batch=4, gap=4.0, power=1.0, **kw):
    """Relleno por validacion cruzada: cada punto de arriba del acantilado se predice
    con interp.cliff sin el; |error LOO| se interpola (TPS) a toda la imagen y los puntos
    nuevos van donde ese error es alto y lejos de lo consultado. Fuera: debajo del borde y
    una franja de `gap` px arriba de el (ya resuelto por la biseccion)."""
    from scipy.interpolate import RBFInterpolator

    H, W = oracle.shape
    yy, xx = np.mgrid[0:H, 0:W]
    while oracle.n_used < n:
        ij, v = oracle.observed()
        kw = dict(kw, mono=False)  # la proyeccion monotona es cara y no cambia el LOO de forma util
        _, info = interp.cliff(ij, v, oracle.shape, **kw)
        if info is None:
            return adaptive(oracle, n, n1=0)
        line = lambda r, c: np.polyval(info[0], c) - r  # > 0: arriba del borde
        up = np.where(line(ij[:, 0], ij[:, 1]) > gap)[0]
        errs = []
        for k in up:
            m = np.ones(len(ij), bool)
            m[k] = False
            rec, _ = interp.cliff(ij[m], v[m], oracle.shape, **kw)
            errs.append(abs(rec[ij[k, 0], ij[k, 1]] - v[k]))
        s_ = float(max(H, W))
        field = RBFInterpolator(ij[up] / s_, np.array(errs), kernel="linear")(
            np.column_stack([yy.ravel(), xx.ravel()]) / s_).reshape(H, W)
        field = np.clip(field, 0, None)
        field[line(yy, xx) <= gap] = 0.0
        d2 = np.full((H, W), np.inf)
        for i, j in ij:
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        new = []
        for _ in range(min(batch, n - oracle.n_used)):
            k = int(np.argmax((field + 1e-4) ** power * np.sqrt(d2)))
            i, j = divmod(k, W)
            new.append((i, j))
            d2 = np.minimum(d2, (yy - i) ** 2 + (xx - j) ** 2)
        oracle.query(new)


STRATEGIES = {"grid": grid, "halton": halton, "uniform": uniform, "adaptive": adaptive, "bisect": bisect, "cliff_fill": cliff_fill}


def sample(oracle, strategy, n, **kw):
    STRATEGIES[strategy](oracle, n, **kw)
    return oracle.observed()
