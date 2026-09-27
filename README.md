# Thesis: Deep Image Prior for KMC phase diagrams

Undergraduate thesis in Computer Science at FaMAF, Universidad Nacional de Córdoba. Work in progress.

The goal is to reconstruct phase diagrams of electrochemical intercalation from a small set of
kinetic Monte Carlo (KMC) simulations. Each point of the diagram is an expensive simulation, so
only a sparse sample is run and the rest is filled in with Deep Image Prior, treating the diagram
as an image with missing pixels.

## Contents

| Path | What it is |
| --- | --- |
| `KMC-Galvanostatic_noclus_param.cpp`, `acumulador.h` | Galvanostatic KMC simulation in C++ with OpenMP |
| `dip/` | Reconstruction code: DIP inpainting, metrics, phase diagrams, point sampling |
| `slurm/` | Job scripts for the SLURM cluster (DIP runs and KMC sweeps) |
| `analysis/` | One-off analysis scripts |
| `models/`, `utils/`, `notebooks/` | Code from the original [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior) repository |

## Running

KMC simulation:

```bash
g++ -O3 -fopenmp KMC-Galvanostatic_noclus_param.cpp -o KMC-Galvanostatic_noclus_param
./KMC-Galvanostatic_noclus_param <xi> <el> <run_index>
```

DIP reconstruction (from the repository root):

```bash
pip install -r requirements.txt
python -m dip.restoration
```

See `dip/README.md` for the available options.

## Credits

Built on [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior) by Dmitry Ulyanov,
Andrea Vedaldi and Victor Lempitsky (CVPR 2018). Licensed under Apache 2.0, see `LICENSE` and
`NOTICE`.
