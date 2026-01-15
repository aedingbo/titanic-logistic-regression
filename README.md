# Titanic Survival Prediction (Logistic Regression)

Predict whether a passenger survived the Titanic using supervised classification. This repo includes two notebooks:

- **01_workbook.ipynb** — full class workflow + notes (EDA → preprocessing → modeling → evaluation)
- **02_clean.ipynb** — minimal, portfolio-ready version with a concise conclusion

---

## Problem
Given passenger attributes (e.g., sex, class, age, fare, family relations, embarkation port), predict **Survived (1) vs. Not Survived (0)**.  
This project emphasizes **classification metrics** (precision/recall/F1/accuracy) and demonstrates both a library implementation and a custom implementation.

---

## Skills Demonstrated
- **Exploratory Data Analysis (EDA):** Inspected distributions, missingness (notably `Age`), and target balance to guide preprocessing.
- **Preprocessing & Feature Engineering:** One-hot encoded categorical variables (e.g., `Sex`, `Embarked`) and removed non-informative text/ID columns.
- **Missing Value Imputation:** Used **KNN imputation** to fill missing `Age` values using correlated numeric/categorical features.
- **Model Training & Evaluation:** Trained and evaluated **Logistic Regression (scikit-learn)** using `classification_report`.
- **From-scratch Implementation:** Implemented a **custom logistic regression** (weights + prediction logic) to validate understanding of the underlying mechanics and compare performance.

---

## Project Workflow (01_workbook.ipynb)
### 1) Data Cleaning & EDA
- Loaded the dataset and checked shape / missing values.
- Identified columns to drop (IDs/text) and fields needing preprocessing.

### 2) Preprocessing
- Dropped non-model features (e.g., identifiers and high-cardinality text fields).
- One-hot encoded categorical features.
- Imputed missing `Age` values using KNN imputation.
- Final prepared dataset shape after preprocessing: **(889, 11)**.

### 3) Modeling + Evaluation
- **Logistic Regression (scikit-learn):** trained and evaluated with a classification report.
- **Custom Logistic Regression:** computed weights and generated predictions; compared performance against sklearn output.

---

## Results (Summary)
### Logistic Regression (scikit-learn)
- **Accuracy:** **0.81** (889 samples)
- Class 0 (Not Survived): precision **0.83**, recall **0.87**
- Class 1 (Survived): precision **0.77**, recall **0.71**

### Custom Logistic Regression
- **Accuracy:** **0.67** (889 samples)
- Class 0 (Not Survived): recall **0.95** (high), but
- Class 1 (Survived): recall **0.22** (low)

**Key takeaway:** the sklearn baseline provides a strong, balanced benchmark, while the custom implementation highlights how thresholding/optimization details can dramatically affect minority-class recall.

---

## Repo Contents
- `01_workbook.ipynb` — full workflow + notes
- `02_clean.ipynb` — minimal version for portfolio review
- *(dataset file used in the notebook)*

---

## How to Run (optional)
### Prerequisites
- Python 3.x
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib scikit-learn
  ```
Ensure `train.csv` is in the same folder as the notebooks.

---

## Acknowledgments
- Completed as part of the Applied Machine Learning Bootcamp (Columbia University).
- Titanic dataset used for educational modeling and evaluation.
