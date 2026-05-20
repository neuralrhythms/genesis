# Genesis

**A repository of learning materials. From a simple neuron to Artificial Intelligence.**

This is the companion code repository for [Neurogenesis](https://ai.neuralrhythms.in) — an interactive AI education platform by [NeuralRhythms](https://www.neuralrhythms.in).

---

## What's Inside

Each folder contains runnable Python code that accompanies a chapter or case study on the website. The code is heavily commented so you can read it like a tutorial — even without deep coding experience.

```
genesis/
├── chapters/
│   └── chapter-01/          # The First Neuron
│       ├── simple_neuron.py
│       ├── neuron_training.py
│       └── neuron_training_batch.py
├── case-studies/
│   └── car-prices/          # Case Study 01: Car Price Prediction
│       ├── 01_explore_data.py
│       ├── 02_first_neuron.py
│       ├── 03_training_raw.py
│       ├── 04_normalised.py
│       ├── 05_visualise_loss.py
│       ├── 06_save_model.py
│       ├── 07_load_model.py
│       ├── 08_exercise_solution.py
│       └── data/
│           └── used_cars.csv
├── requirements.txt
└── LICENSE
```

---

## Getting Started

For detailed setup instructions covering **Windows**, **macOS**, and **Linux**, see:

👉 **[SETUP.md](./SETUP.md)** — complete environment setup guide

### Quick start (if you already have Python 3.10+):

### 1. Clone the repository

```bash
git clone https://github.com/neuralrhythms/genesis.git
cd genesis
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run any script

```bash
cd case-studies/car-prices
python 04_normalised.py
```

---

## Recommended Editor

[Visual Studio Code](https://code.visualstudio.com/) with the Python extension.

Quick setup in VS Code:
1. Open the `genesis` folder
2. `Ctrl+Shift+P` → "Python: Create Environment" → Venv → select interpreter → select `requirements.txt`
3. Open any `.py` file and press `F5` to run

---

## Index of Chapters

| # | Title | Website | Code |
|---|-------|---------|------|
| 01 | The First Neuron | [Read →](https://ai.neuralrhythms.in/chapters/chapter-01/) | `chapters/chapter-01/` |

## Index of Case Studies

| # | Title | Website | Code |
|---|-------|---------|------|
| 01 | Car Price Prediction | [Read →](https://ai.neuralrhythms.in/case-studies/car-prices/) | `case-studies/car-prices/` |

---

## Dependencies

All code uses:
- **Python 3.10+**
- **PyTorch** — neural network framework
- **Pandas** — data loading and cleaning
- **Matplotlib** — loss curve visualisation

See `requirements.txt` for exact versions.

---

## License

This repository is licensed under the [MIT License](./LICENSE).

The used car dataset (`case-studies/car-prices/data/used_cars.csv`) is by
[Taeef Najib](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data)
and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Links

- Website: [ai.neuralrhythms.in](https://ai.neuralrhythms.in)
- Portfolio: [neuralrhythms.in](https://www.neuralrhythms.in)
- Website repo: [github.com/neuralrhythms/neurogenesis](https://github.com/neuralrhythms/neurogenesis)
