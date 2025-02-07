# [CVKAN: Complex-Valued Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.02417)
Authors: Matthias Wolff, Florian Eilers, Xiaoyi Jiang \
University of Münster, Department of Computer Science

Link to Paper: https://arxiv.org/abs/2502.02417

### Abstract
In this work we propose $`\mathbb{C}`$KAN, a complex-valued KAN, to join the intrinsic interpretability of KANs and the advantages of Complex-Valued Neural Networks (CVNNs). We show how to transfer a KAN and the necessary associated mechanisms into the complex domain. To confirm that $`\mathbb{C}`$KAN meets expectations we conduct experiments on symbolic complex-valued function fitting and physically meaningful formulae as well as on a more realistic dataset from knot theory. Our proposed $`\mathbb{C}`$KAN is more stable and performs on par or better than real-valued KANs while requiring less parameters and a shallower network architecture, making it more explainable.

## Table of Contents

- [experiments](experiments): Scripts for our experiments and corresponding results
  - [fit_formulas.py](experiments/fit_formulas.py): Experiments for function fitting. Simple arbitrary $\left(z^2, \quad \sin(z), \quad z_1 \cdot z_2, \quad (z_1^2 + z_2^2)^2 \right)\quad$ as well as physically meaningful formulae: (circuit & holography).
  - [knot_dataset.py](experiments/knot_dataset.py): Experiments for knot classification.
  - [results.json](experiments/results.json): All of our results as a list of dictionaries, stored as JSON
  - [run_crossval.py](experiments/run_crossval.py): 
- [utils](utils): miscellanious utils
  - [dataloading](utils/dataloading): 
    - [create_complex_dataset.py](utils/dataloading/create_complex_dataset.py)
  - [plotting](utils/plotting): 
- [images](images): The images used in our paper
- [models](models):

## How to use

### $`\mathbb{C}`$KAN

### Plotting