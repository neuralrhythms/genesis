# Case Study 01 — Car Price Prediction

Predict used car prices using multiple linear regression, implemented as a single PyTorch neuron.

**Website:** [ai.neuralrhythms.in/case-studies/car-prices/](https://ai.neuralrhythms.in/case-studies/car-prices/)

---

## Mathematical Foundation

This case study implements **multiple linear regression** — one of the oldest and most widely used techniques in statistics and machine learning. It was first described by [Legendre](https://en.wikipedia.org/wiki/Adrien-Marie_Legendre) and [Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) in the early 1800s, and remains the starting point for most predictive modelling today.

### The Model Equation

```
price = w₁ × age + w₂ × mileage + b
```

The model learns three numbers:
- **w₁** — how much each year of age reduces the price
- **w₂** — how much each mile of driving reduces the price
- **b** — the baseline price when age and mileage are both zero (the intercept)

This is a **linear combination** of input features plus a bias. Geometrically, the model fits a flat plane through 3D space (age × mileage → price). The training process tilts and shifts this plane until it best fits the data.

### The Objective — Minimising Loss

"Best fit" is defined mathematically as the set of weights that minimises the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error):

```
MSE = (1/n) × Σ(predicted − actual)²
```

The squared term penalises large errors more than small ones, pushing the model to avoid big mistakes. The sum is averaged over all *n* training examples.

### Why "Linear" Regression?

| Term | Meaning |
|------|---------|
| **Linear** | The output is a straight-line (or flat-plane) function of the inputs. There are no curves, no powers, no interactions between features. Each feature contributes independently and proportionally. |
| **Multiple** | "Multiple" means more than one input feature. Simple linear regression uses one feature (`y = wx + b`); multiple linear regression uses two or more (`y = w₁x₁ + w₂x₂ + ... + b`). |
| **Regression** | The model predicts a continuous number (price in dollars), not a category. Predicting "expensive vs cheap" would be classification; predicting "$34,500" is regression. |

### Neural Network or Statistics?

A single neuron with no activation function (`nn.Linear(2, 1)`) is **mathematically identical** to [ordinary least squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares) linear regression. The difference is only in how we find the optimal weights:

| Approach | Method | When to use |
|----------|--------|-------------|
| Classical statistics | Normal equation (closed-form) | Small datasets, few features |
| Neural network | Gradient descent (iterative) | Any size, scales to deep networks |

We use the neural network approach because:
1. It teaches the **training loop** (`zero_grad → forward → loss → backward → step`) that is used for ALL neural networks — from this simple model to GPT-4.
2. It **scales**. The normal equation becomes impractical with millions of features or data points. Gradient descent works regardless of scale.
3. It introduces **normalisation**, **loss curves**, and **model persistence** — skills needed for every future model.

### Limitations of Linear Regression

Linear regression assumes a straight-line relationship. It cannot capture:
- **Non-linear patterns** — e.g. luxury brands holding value differently than economy brands
- **Feature interactions** — e.g. low mileage matters more for newer cars than older ones
- **Complex decision boundaries** — e.g. a price cliff at a certain age threshold

These limitations motivate **Chapter 2: Activation Functions** — where non-linearity is introduced, allowing models to learn curves instead of only straight lines.

---

## Scripts

Run in order — each builds on the previous:

| # | File | What it does | Key concept |
|---|------|-------------|-------------|
| 01 | `01_explore_data.py` | Load and inspect the CSV | Pandas basics |
| 02 | `02_first_neuron.py` | Create a 2-input neuron (no training) | Multi-feature tensors |
| 03 | `03_training_raw.py` | Train on raw data | Why normalisation is needed |
| 04 | `04_normalised.py` | Train with normalised data | Feature scaling |
| 05 | `05_visualise_loss.py` | Plot the loss curve | Matplotlib |
| 06 | `06_save_model.py` | Save model and stats to disk | Model persistence |
| 07 | `07_load_model.py` | Load and predict (no training) | Inference mode |
| 08 | `08_exercise_solution.py` | Add accident history (3 features) | Categorical encoding |

---

## Dataset

**Source:** [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data) by Taeef Najib  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Size:** 4,009 rows × 12 columns

Features used:
- `model_year` → derived as `age` (max_year − model_year)
- `milage` → cleaned from string format (e.g. `"51,000 mi."` → `51000`)
- `accident` → encoded as binary (1 = no accident, 0 = accident reported)
- `price` → target variable, cleaned from `"$10,300"` → `10300`

---

## Running the Code

```bash
cd case-studies/car-prices
python 01_explore_data.py
python 02_first_neuron.py
# ... and so on
```

See [SETUP.md](../../SETUP.md) for full environment setup instructions.

---

## Further Reading

- [Multiple Linear Regression — Wikipedia](https://en.wikipedia.org/wiki/Linear_regression#Multiple_linear_regression)
- [Ordinary Least Squares — Wikipedia](https://en.wikipedia.org/wiki/Ordinary_least_squares)
- [Feature Scaling — Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
- [Gradient Descent — Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [PyTorch nn.Linear documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
