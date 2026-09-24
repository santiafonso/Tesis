"""Oraculo de consultas puntuales -- ARCHIVO FIJO, el loop de experimentacion no lo toca.

Simula lo que va a pasar con KMC real: cada punto del diagrama cuesta una corrida
cara, asi que un metodo solo puede PREGUNTAR el valor en pixeles concretos, con un
presupuesto maximo de consultas distintas. Nunca ve la imagen completa.

Cada consulta queda registrada en `queries.json` (lo escribe el oraculo, no el
metodo), y `autoexp.eval` usa ese registro para verificar el presupuesto y que los
valores sean los reales. Todo valor que el metodo le pase a DIP y que NO este en ese
registro es un pseudo-punto inventado (ej. interpolado), que esta permitido pero no
cuenta como observacion.

Ruido (opcional, `noise` > 0): simula que cada punto sale de una corrida de KMC, que es
estocastica. Cada pixel recibe un ruido gaussiano fijo de desvio `noise`, determinado por
(noise_seed, i, j), asi que repetir una consulta devuelve el mismo valor, recortado a [0, 1].
El puntaje se sigue calculando contra el mapa LIMPIO.

Verdad de referencia: `results/phase_diagram_g/g<val>/sim_128.png` en [0, 1], la
misma imagen que se venia reconstruyendo con DIP.
"""
import json
import os

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE_DIR = os.environ.get("AUTOEXP_PHASE_DIR", os.path.join(REPO, "results", "phase_diagram_g"))
DEFAULT_BUDGET = 64


class BudgetExceeded(RuntimeError):
    pass


SYNTH_DIR = os.path.join(REPO, "data", "restoration", "synthetic")


def truth_path(g):
    """g numerico -> diagrama de fases; si no, una imagen sintetica por nombre (gauss,
    rosenbrock, ...), para probar el metodo en formas que no vio."""
    p = os.path.join(PHASE_DIR, "g%s" % g, "sim_128.png")
    if not os.path.exists(p) and os.path.exists(os.path.join(SYNTH_DIR, "%s.png" % g)):
        return os.path.join(SYNTH_DIR, "%s.png" % g)
    return p


def load_truth(g):
    """Solo para autoexp.eval. Un metodo de reconstruccion NO debe llamar esto."""
    return np.asarray(Image.open(truth_path(g)).convert("L"), dtype=np.float64) / 255.0


class Oracle:
    def __init__(self, g, budget=DEFAULT_BUDGET, noise=0.0, noise_seed=0):
        self.g = g
        self.budget = budget
        self.noise = float(noise)
        self.noise_seed = int(noise_seed)
        self.__img = load_truth(g)
        self.shape = self.__img.shape
        self.queried = {}  # (i, j) -> valor

    @property
    def n_used(self):
        return len(self.queried)

    @property
    def remaining(self):
        return self.budget - self.n_used

    def query(self, points):
        """points: iterable de (i, j) enteros (fila, columna). Devuelve np.array de valores.

        Repetir un punto ya consultado no gasta presupuesto.
        """
        pts = [(int(i), int(j)) for i, j in points]
        H, W = self.shape
        new = {p for p in pts if p not in self.queried}
        for i, j in new:
            if not (0 <= i < H and 0 <= j < W):
                raise ValueError("punto fuera de la imagen: (%d, %d)" % (i, j))
        if self.n_used + len(new) > self.budget:
            raise BudgetExceeded(
                "%d usadas + %d nuevas > presupuesto %d" % (self.n_used, len(new), self.budget)
            )
        for i, j in new:
            x = float(self.__img[i, j])
            if self.noise > 0:
                rng = np.random.default_rng([self.noise_seed, i, j])
                x = float(np.clip(x + self.noise * rng.standard_normal(), 0.0, 1.0))
            self.queried[(i, j)] = x
        return np.array([self.queried[p] for p in pts])

    def observed(self):
        """(ij (N,2) int, valores (N,)) de todo lo consultado hasta ahora."""
        if not self.queried:
            return np.zeros((0, 2), int), np.zeros(0)
        ij = np.array(list(self.queried.keys()), dtype=int)
        v = np.array(list(self.queried.values()))
        return ij, v

    def save(self, path):
        with open(path, "w") as f:
            json.dump(
                {
                    "g": self.g,
                    "budget": self.budget,
                    "noise": self.noise,
                    "noise_seed": self.noise_seed,
                    "shape": list(self.shape),
                    "points": [[i, j, v] for (i, j), v in self.queried.items()],
                },
                f,
            )
