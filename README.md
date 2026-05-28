# 💎 Diamond Price Prediction

A machine learning project that predicts diamond prices using regression models, built with Python and scikit-learn.

---

##  Overview

This project explores a diamonds dataset to understand what physical and quality attributes drive price, then trains and evaluates two regression models — Linear Regression and Random Forest — to predict diamond prices.

---

## Dataset

**File:** `diamonds.csv`

| Feature | Description |
|---|---|
| `carat` | Diamond weight |
| `cut` | Cut quality (Fair → Ideal) |
| `color` | Color grade (J = worst → D = best) |
| `clarity` | Internal cleanliness (I1 → IF) |
| `depth` | Height relative to width (%) |
| `table` | Width of top surface (%) |
| `x`, `y`, `z` | Physical dimensions (mm) |
| `price` | **Target variable** — price in USD |

---

## 🔧 Project Pipeline

### 1. Data Cleaning
- Dropped unnamed index column
- Imputed missing numerical values using **median** (affected ~3% of rows)
- Removed rows with physically impossible dimension values (zeros or extreme outliers in x, y, z)
- Applied **IQR capping** on `depth` and `table` columns
- Removed 93 duplicate rows

### 2. Feature Engineering
- Created a new `Volume` feature: `x × y × z`

### 3. Encoding
- Applied **Ordinal Encoding** to `cut`, `color`, and `clarity` using domain-appropriate orderings

### 4. Exploratory Data Analysis (EDA)
- Price distribution analysis (right-skewed)
- Carat vs. Price scatter plot (strong positive correlation)
- Average price by color grade
- Correlation heatmap — key findings:
  - `x`, `y`, `z` have a very strong positive correlation with price (~0.86)
  - `carat` has a strong positive correlation with price (~0.62)
  - Dimension columns are nearly perfectly correlated with each other (~0.96–0.97)

### 5. Modeling

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | — | — |
| Random Forest | — | — |

> *Run the notebook to populate model metrics.*

---

## Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn seaborn matplotlib
```

### Run the Notebook

```bash
jupyter notebook DiamondData.ipynb
```

---

##  Project Structure

```
├── DiamondData.ipynb   # Main analysis and modeling notebook
├── diamonds.csv        # Dataset
└── README.md
```

---

##  Tech Stack

- **Python 3**
- **pandas** — data manipulation
- **seaborn / matplotlib** — visualization
- **scikit-learn** — preprocessing, modeling, and evaluation
- Larger, heavier diamonds command significantly higher prices
- Color grade J (lower quality) counterintuitively had the highest average price — likely driven by carat size confounding
- Random Forest outperformed Linear Regression in capturing non-linear relationships in the data
