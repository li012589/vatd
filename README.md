# Variational Temperature-continuous Optimization
Pytorch implement of paper *"Estimation of Temperature-continuous Statistical Mechanics Using Variational Boltzmann Distribution Optimization"*. [ArXiv link](https://arxiv.org/abs/tbd)

## Abstract
We propose a general framework for estimating the behaviors of a finite statistical system as continuous functions of the temperature. We utilize deep-learning explicit density generative models to model the distribution of the systems. Along with the sample configurations, the temperature is also used to estimate the probability. When optimized with the reverse Kullback-Leibler divergence, at optimal point, this learnt distribution can be proved to be a temperature-continuous Boltzmann distribution. This training requires no dataset but only the energy function. Beside being a direct sampler, the learnt model gives temperature-continuous estimations of the system’s properties, e.g., the partition function. With the help of the automatic differentiation, these continuous estimations can be differentiated in a way similar to what one can do with an analytical solution. To show the generality of the proposed method, we demonstrate this framework on the phase transition of Ising model (discrete-variable model with PixelCNN) and XY model (continuous-variable model with Normalizing flow).

## For results of 2D XY model

Training,
```bash
python xyVonMcubic.py
```

Plotting,
```bash
python xyCapPlot.py
```

## For results of 2D Ising model

Training,
```bash
python .py
```

Plotting,
```bash
python .py
```
