# Variational Temperature-differential (VaTD) Model Optimization
Pytorch implement of paper *"Deep generative modeling of the canonical ensemble with differentiable thermal properties"*. [arXiv link](https://arxiv.org/abs/2404.18404)

## Abstract

It is a long-standing challenge to accurately and efficiently compute thermodynamic quantities of condensed-matter many-body systems at thermal equilibrium. The conventional methods, e.g., Markov chain Monte Carlo (MCMC) method, require lots of steps to equilibrate. The recently developed deep learning methods can directly sample independent equilibrium samples, but have to train separate models for each temperature point, essentially breaking continuous thermal effects, so they are often inadequate for directly representing physical ensembles, particularly during phase transitions (PTs). Here, we propose a variational modeling method for canonical ensembles with differentiable temperature, which gives accurate thermodynamic quantities as continuous functions of temperature akin to an analytical solution. Using a deep generative model, the free energy is estimated and minimized in a continuous temperature range. At optimal, this generative model is directly a Boltzmann distribution with temperature dependence. This training requires no dataset, and works with arbitrary explicit density generative models. We applied our method to study the PTs in the Ising and XY models, and showed that our direct-sampling simulations are as accurate as the MCMC simulation, but more efficient. Moreover, the differentiable free energy from our method aligns closely with the exact one to the second-order derivative, indicating that incorporating temperature dependence enables the otherwise biased variational model to capture the subtle thermal effects at the PTs. The functional dependence on external parameters along with the exceptional fitting ability of deep learning models sheds light on the direct simulation of physical systems.

## For results of 2D XY model

Training,
```bash
python xyVonMcubic.py
```

Plotting,
```bash
python xyPlot.py
```

## For results of 2D Ising model

Training,
```bash
python isingPixelCNN.py
```

Plotting,
```bash
python isingPlot.py
```

## Citation

Please consider cite our work,
```
@article{li2024deep,
  title = {Deep generative modeling of the canonical ensemble with differentiable thermal properties},
  author = {Li, Shuo-Hui, Zhang, Yao-Wen, and Pan, Ding},
  journal = {Phys. Rev. Lett.},
  pages = {--},
  year = {2025},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/8wx7-kyx8},
  url = {https://link.aps.org/doi/10.1103/8wx7-kyx8}
}

```

