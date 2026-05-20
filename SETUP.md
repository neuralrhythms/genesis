# Environment Setup Guide

Complete instructions for setting up your development environment to run the Genesis code on **Windows**, **macOS**, and **Linux**.

---

## Prerequisites

Before you begin, make sure you have:

| Requirement | Minimum Version | How to check |
|-------------|----------------|--------------|
| Python | 3.10 or higher | `python --version` |
| pip | (comes with Python) | `pip --version` |
| Git | any recent version | `git --version` |

### Installing Python

**Windows:**
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer — **check "Add Python to PATH"** at the bottom of the first screen
3. Click "Install Now"

**macOS:**
```bash
# Option 1: Download from python.org
# Option 2: Install via Homebrew
brew install python
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Linux (Fedora):**
```bash
sudo dnf install python3 python3-pip
```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/neuralrhythms/genesis.git
cd genesis
```

---

## Step 2 — Create a Virtual Environment

A virtual environment keeps this project's packages isolated from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv venv
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
```

**macOS / Linux:**
```bash
python3 -m venv venv
```

---

## Step 3 — Activate the Virtual Environment

You must activate the environment every time you open a new terminal.

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

> **If you get an execution policy error**, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> This allows scripts for the current session only. It resets when you close the terminal.

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

When activated, you will see `(venv)` at the start of your terminal prompt.

---

## Step 4 — Install Dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs:
- **PyTorch** — the neural network framework
- **Pandas** — data loading and manipulation
- **Matplotlib** — plotting loss curves and charts

---

## Step 5 — Run a Script

Navigate to any chapter or case study folder and run a Python file:

```bash
cd case-studies/car-prices
python 01_explore_data.py
```

Run the scripts in numerical order — each one builds on the previous.

---

## Recommended Editor: Visual Studio Code

[Download VS Code](https://code.visualstudio.com/) — free, cross-platform, excellent Python support.

### Quick setup in VS Code:

1. Open the `genesis` folder in VS Code (`File → Open Folder`)
2. Install the **Python extension** (by Microsoft) if prompted
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → type "Python: Select Interpreter"
4. Choose the interpreter inside your `venv` folder (it will say `./venv/...`)
5. Open any `.py` file and press `F5` to run it

### Alternative: VS Code guided environment creation

1. `Ctrl+Shift+P` → "Python: Create Environment"
2. Select "Venv"
3. Select your Python interpreter
4. Select `requirements.txt` when prompted

VS Code will create the venv and install dependencies automatically.

---

## Troubleshooting

**`python` command not found (Windows)**
→ Python wasn't added to PATH during installation. Reinstall and check "Add Python to PATH", or use `py` instead of `python`.

**`python3` command not found (macOS/Linux)**
→ Try `python` instead. Some systems use `python3` and some use `python`.

**`pip install` fails with permission error**
→ Make sure your virtual environment is activated (you should see `(venv)` in your prompt).

**PowerShell execution policy error on Windows**
→ Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating, or use Command Prompt instead.

**`ModuleNotFoundError: No module named 'torch'`**
→ Your virtual environment is not activated, or dependencies are not installed. Activate it and run `pip install -r requirements.txt`.

**PyTorch installation is very large / slow**
→ PyTorch is ~2 GB. This is normal. If you have a slow connection, be patient. You only download it once.

**Scripts fail with `FileNotFoundError: ./data/used_cars.csv`**
→ Make sure you are running the script from inside the correct folder (e.g. `case-studies/car-prices/`), not from the repository root.

---

## Deactivating the Virtual Environment

When you are done working:

```bash
deactivate
```

This returns your terminal to the system Python. You can reactivate later with the same activate command.

---

## Updating Dependencies

If the `requirements.txt` is updated in a future commit:

```bash
git pull
pip install -r requirements.txt --upgrade
```

---

## Platform Notes

| Platform | Python command | Activate command | Notes |
|----------|---------------|-----------------|-------|
| Windows (PowerShell) | `python` | `venv\Scripts\Activate.ps1` | May need execution policy bypass |
| Windows (CMD) | `python` | `venv\Scripts\activate.bat` | No policy issues |
| macOS | `python3` | `source venv/bin/activate` | Install via Homebrew or python.org |
| Linux | `python3` | `source venv/bin/activate` | Install via package manager |

---

*Part of the [Neurogenesis](https://ai.neuralrhythms.in) learning platform by [NeuralRhythms](https://www.neuralrhythms.in).*
