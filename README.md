Author: Duncan Duncan
# Iris Flower Classifier

A simple end-to-end machine learning pipeline that classifies iris flowers into three
species (setosa, versicolor, virginica) from four measurements: sepal length, sepal
width, petal length and petal width.

Built for AIN1103 – Fundamentals of Programming for Machine Learning
(Assignment 5: Version Control and a Simple ML Pipeline).

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | The full pipeline: load, explore, visualise, train, evaluate, interpret |
| `requirements.txt` | Python packages with the versions the project was tested on |
| `README.md` | Project description and run instructions |
| `outputs/` | Generated plots, saved model and metrics |

## Pipeline

1. **Load** the Iris dataset (scikit-learn) into a **Pandas** DataFrame.
2. **Explore** with summary statistics, missing-value check and class counts.
3. **Visualise** feature histograms and a petal-size scatter plot (**Matplotlib**).
4. **Prepare** a **NumPy** feature array and a stratified 80/20 train/test split.
5. **Train** a pre-built `RandomForestClassifier` (100 trees, `random_state=42`).
6. **Evaluate** with accuracy, per-class precision/recall/F1 and a confusion matrix.
7. **Interpret** using feature importances.

## How to run

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the dependencies
pip install -r requirements.txt

# 3. run the project
python main.py
```

## Results

Test accuracy: **0.90** on 30 held-out samples.

- Setosa is classified perfectly.
- The few errors are between versicolor and virginica, whose measurements overlap.
- Petal width (0.44) and petal length (0.43) are the most important features; sepal
  width contributes almost nothing (0.015).

## Outputs (in `outputs/`)

- `feature_histograms.png`, `petal_scatter.png` – exploratory plots
- `feature_importance.png`, `confusion_matrix.png` – model interpretation
- `model.joblib` – trained model
- `metrics.json` – test accuracy

## Using the saved model

```python
import joblib

model = joblib.load("outputs/model.joblib")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))  # 0 = setosa
```
