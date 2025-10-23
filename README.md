Here’s the full updated `README.md` ready to copy and paste:

````markdown
# Online Deep Learning from Doubly-Streaming Data
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Abstract
This project implements the **OLD³S** framework proposed in the paper *Online Deep Learning from Doubly-Streaming Data*.  
The framework addresses a new online learning problem where:

1. **Data instances arrive continuously** and may follow **non-stationary distributions**, requiring models to update in real-time (concept drift).
2. **Feature spaces evolve dynamically**, meaning new features emerge while others become obsolete over time.

To handle these challenges, OLD³S:
- Learns a **shared latent subspace** to connect old and new feature spaces.
- Uses **adaptive model capacity** to balance shallow (fast) and deep (expressive) learning.
- Employs **dual autoencoders** and **Hedge Backpropagation** to transfer knowledge during feature evolution.

The approach is evaluated on multiple real-world datasets, demonstrating strong performance under feature drift and class imbalance.

---

## Requirements
This code has been tested on **Python 3.9+** on **Windows, Linux, and macOS**.

Create a virtual environment and install dependencies:

```bash
# Create environment
conda create -n OLDS python=3.9
conda activate OLDS

# Install core dependencies
pip install torch torchvision torchaudio
pip install pandas matplotlib scikit-learn scipy river
````

---

## Project Structure

```
project/
├── autoencoder.py        # Shallow autoencoder architecture
├── mlp.py                 # Multi-layer perceptron with Hedge Backprop
├── loaddatasets.py        # Data loading & preprocessing
├── model.py               # Core OLD³S algorithm
├── train.py               # Entry point to run experiments
├── metric.py              # Metrics and plotting functions
├── plot_all_metrics.py    # Script to visualize results
├── vae.py                 # Variational Autoencoder (optional)
└── data/
    ├── magic04_X.csv
    ├── magic04_y.csv
    ├── adult.data
    ├── car.data
    ├── arrhythmia.data
    ├── new-thyroid.data
    └── INSECTS.csv
```

---

## Run

The main entry point is `train.py`.
Here is an example command:

```bash
python train.py -DataName=magic -AutoEncoder='AE' -beta=0.9 -eta=-0.01 -learningrate=0.001 -RecLossFunc=MSELoss
```

### **Arguments**

| Argument        | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `-DataName`     | Dataset to run: `magic`, `adult`, `car`, `arrhythmia`, `thyroid`, `insects` |
| `-AutoEncoder`  | Autoencoder type: `AE` (shallow) or `VAE` (variational)                     |
| `-beta`         | Hedge Backpropagation beta parameter                                        |
| `-eta`          | Learning rate for adaptive weight updates                                   |
| `-learningrate` | Base learning rate for optimizers                                           |
| `-RecLossFunc`  | Reconstruction loss: `BCE`, `Smooth`, `KL`, or `MSE`                        |

---

### **Examples**

**MAGIC dataset:**

```bash
python train.py -DataName=magic -AutoEncoder='AE' -beta=0.9 -eta=-0.01 -learningrate=0.001 -RecLossFunc=Smooth
```

**Adult dataset:**

```bash
python train.py -DataName=adult -AutoEncoder='AE' -beta=0.9 -eta=-0.01 -learningrate=0.001 -RecLossFunc=Smooth
```

**INSECTS dataset:**

```bash
python train.py -DataName=insects -AutoEncoder='AE' -beta=0.9 -eta=-0.01 -learningrate=0.001 -RecLossFunc=Smooth
```

---

## Output

Training will generate the following files:

```
./data/parameter_<dataset>/
├── net_model1.pth       # Classifier before feature drift
├── net_model2.pth       # Classifier after feature drift
├── Accuracy             # Saved accuracy list (windowed)
└── metrics/
    └── all_metrics.npz  # Cohen Kappa, G-Mean, PR-AUC, drift points
```

---

## Visualizing Results

After training, generate and view plots:

```bash
python plot_all_metrics.py
```

Plots will be saved in:

```
./data/parameter_<dataset>/metrics/
```

Example to open on Linux:

```bash
xdg-open ./data/parameter_magic/metrics/oca.png
```

On macOS:

```bash
open ./data/parameter_magic/metrics/oca.png
```

On Windows PowerShell:

```bash
start data\parameter_magic\metrics\oca.png
```

---

## Metric Calculation

The main metric is **ACR (Averaged Cumulative Regret)**:

```python
f_star = max(accuracy_list)
acr = mean([f_star - i for i in accuracy_list])
```

Other metrics computed:

* Cohen’s Kappa
* G-Mean
* PR-AUC
* Drift detection points using **ADWIN**

These are saved automatically in `all_metrics.npz`.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

