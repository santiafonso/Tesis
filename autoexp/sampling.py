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


STRATEGIES = {"grid": grid, "halton": halton, "uniform": uniform, "adaptive": adaptive}


def sample(oracle, strategy, n, **kw):
    STRATEGIES[strategy](oracle, n, **kw)
    return oracle.observed()
