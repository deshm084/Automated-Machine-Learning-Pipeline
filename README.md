# 🏭 Automated Machine Learning Pipeline

A modular ML pipeline built with Scikit-Learn that automates the end-to-end workflow: Preprocessing → Feature Selection → Model Selection → Hyperparameter Tuning.

## 🎯 The Problem
Manual model tuning is prone to **Data Leakage** (e.g., scaling data before splitting) and is tedious to reproduce. 

## 💡 The Solution
I implemented a `Pipeline` architecture wrapped in a `GridSearchCV` engine.
* **Leakage Prevention:** All preprocessing (Imputation/Scaling) is fitted *only* on the training folds during Cross-Validation.
* **Automated Search:** The system automatically competes different algorithms (Random Forest vs. Logistic Regression) against each other to find the optimal configuration.

## 🛠 Tech Stack
* **Scikit-Learn:** `Pipeline`, `GridSearchCV`, `SelectKBest`.
* **Algorithms:** Random Forest, Logistic Regression.
* **Metrics:** Precision, Recall, F1-Score.

## 🏆 Results
The system identified that a **Logistic Regression** model using the top **10 Features** yielded the highest accuracy (97%), outperforming the more complex Random Forest for this specific dataset.
