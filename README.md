# Heart Disease Analysis

A data mining project that builds a K-Nearest Neighbors (KNN) classifier to predict the presence of heart disease using the UCI processed Cleveland Heart Disease dataset.

## Project Structure

```
.
├── data/
│   ├── processed.cleveland.data     # Raw dataset (14 attributes, 303 patients)
│   └── heart-disease.names.txt      # Attribute documentation
├── notebooks/
│   └── Heart_Disease_KNN_Project.ipynb   # Full analysis notebook
└── outputs/
    └── figures/                     # Exported charts (EDA, accuracy curve, confusion matrix)
```

## Pipeline

1. **Data inspection & cleaning** — loaded the 14-attribute dataset, identified and handled missing values (`ca`, `thal`), converted the multi-class diagnosis into a binary target (disease / no disease).
2. **Exploratory data analysis** — class balance, age/sex breakdowns, feature boxplots by diagnosis, correlation heatmap.
3. **Feature normalization** — standardized all features with `StandardScaler` (fit on training data only).
4. **Train/test split** — 80/20 stratified split.
5. **KNN classification** — trained a baseline model, then swept `k` from 1 to 30 to find the optimal number of neighbors.
6. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix on the held-out test set.

## Results

Best model: KNN with **k = 3**

| Metric | Value |
|---|---|
| Accuracy | 0.900 |
| Precision | 0.893 |
| Recall | 0.893 |
| F1-score | 0.893 |

## Dataset

[UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```
