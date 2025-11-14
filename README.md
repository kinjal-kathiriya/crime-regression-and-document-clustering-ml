# 🔍 Crime Regression & Document Clustering using Classical Machine Learning

This project explores two major machine learning pipelines:

1. **Regression Modeling** on the Communities & Crime dataset  
2. **Document Clustering** on a 5-category subset of the 20 Newsgroups dataset  

It includes regression models, feature selection, regularization techniques, SGD optimization, custom clustering, cosine distance, TF–IDF preprocessing, and cluster evaluation.

---

# 📂 Datasets Overview

## **1. Communities & Crime Dataset**
Contains socio-economic, demographic, and crime-related variables from:
- 1990 U.S. Census  
- 1995 FBI Crime Statistics  

Target Variable:
- **ViolentCrimesPerPop**

Preprocessing:
- Removed identifier fields: `state`, `communityname`
- Handled missing values using mean imputation
- Train/Test Split: **80% train / 20% test** (random_state=33)

Used for:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- SGD Regression  
- Feature Selection (SelectPercentile)  
- RMSE & MAE evaluation  

---

## **2. Newsgroups5 Dataset**
A subset of the classical 20 Newsgroups dataset.

- 2,500 documents  
- 5 categories:
  - windows (0)
  - crypt (1)
  - christian (2)
  - hockey (3)
  - forsale (4)
- 9,328 term stems
- Files:
  - `matrix.txt` → term × document matrix
  - `terms.txt` → vocabulary
  - `classes.txt` → true labels

Before clustering:
- Matrix transposed to **document × term**
- Train/Test Split: **80% train / 20% test** (random_state=99)
- TF–IDF transformation applied

Used for:
- Custom KMeans clustering (MLA version)
- Cosine distance implementation  
- Cluster analysis  
- Homogeneity & Completeness evaluation  

---

## 🛠️ Tech Stack Used

- Python

- NumPy

- Pandas

- Matplotlib

- scikit-learn

- Custom KMeans (MLA code)

- TF–IDF Transformer

- StandardScaler


## ▶️ How to Run

pip install -r requirements.txt

jupyter notebook notebooks/Crime_regression_ml.ipynb
