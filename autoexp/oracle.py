"""Oraculo de consultas puntuales -- ARCHIVO FIJO, el loop de experimentacion no lo toca.

Simula lo que va a pasar con KMC real: cada punto del diagrama cuesta una corrida
cara, asi que un metodo solo puede PREGUNTAR el valor en pixeles concretos, con un
presupuesto maximo de consultas distintas. Nunca ve la imagen completa.

Cada consulta queda registrada en `queries.json` (lo escribe el oraculo, no el
metodo), y `autoexp.eval` usa ese registro para verificar el presupuesto y que los
valores sean los reales. Todo valor que el metodo le pase a DIP y que NO este en ese
registro es un pseudo-punto inventado (ej. interpolado), que esta permitido pero no
cuenta como observacion.

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


def truth_path(g):
    return os.path.join(PHASE_DIR, "g%s" % g, "sim_128.png")


def load_truth(g):
    """Solo para autoexp.eval. Un metodo de reconstruccion NO debe llamar esto."""
    return np.asarray(Image.open(truth_path(g)).convert("L"), dtype=np.float64) / 255.0


class Oracle:
    def __init__(self, g, budget=DEFAULT_BUDGET):
        self.g = g
        self.budget = budget
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
            self.queried[(i, j)] = float(self.__img[i, j])
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
                    "shape": list(self.shape),
                    "points": [[i, j, v] for (i, j), v in self.queried.items()],
                },
                f,
            )
