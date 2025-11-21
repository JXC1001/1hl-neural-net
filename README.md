# Neural Network With One Hidden Layer (Python)

## Overview
This project implements a neural-network gradient-descent algorithm using a leaky ReLU activation function.

Given data points \((x, y)\) on a user-defined domain, the program attempts to learn the parameters of the model:

$$
p(x) = \left(
\sum_{j=1}^{N}
w_{o_j} \cdot 
\max(\alpha(w_{h_j}x + b_{h_j}),\ w_{h_j}x + b_{h_j})
\right) + b_o
$$

(Where N = the neuron count)

The `NeuralNet` class is implemented entirely in pure Python (no external dependencies; only Python's built-in modules).

---

## Details
- 3N + 1 trainable parameters where N = neuron count
  - Random parameter initialization.
- Domain normalization to \([-1, 1]\) for numerical stability.
- Mean squared error loss.
- Manual gradient computation and gradient-descent updates.
  - Reusable partial derivatives for each neuron: no neuron limit.
- Simple command-line interface for:
  - number of neurons  
  - domain values  
  - y-values  
  - learning rate  
  - ReLU leakiness  
  - iterations  
- `main.py` uses NumPy and matplotlib to create graphs.  
  (**Not required** to use the `NeuralNet` class, only to visualize the model's outputs.)

---

## Limitations & Notes
- Built as an upgrade from the polynomial curve-fitting project.
- Can fit more functions and is more customizable than my older polynomial model.
- Convergence depends on:
  - initialization  
  - learning rate  
  - α (ReLU leakiness)  
  - number of neurons
    - The more complex the function is, the more neurons needed, thus more computations.
  - domain length  
- Large neuron counts significantly increase runtime because all computations are done in pure Python.
- Not optimized for production:
  - no Adam  
  - no batching  
  - no vectorization  
  - no regularization  
- Results vary between runs due to random initialization.

---

## Usage

To run the neural network trainer:

```bash
python src/ONE_HL_NN.py
```

To run the visualization script (requires NumPy + matplotlib):

```bash
python src/main.py
```
