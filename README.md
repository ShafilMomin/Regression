# Regression Models in Python

### Six regression methods, explained through practical notebooks

This repository shows how machine learning models predict numerical values, such as **salary** and **business profit**. Each notebook covers loading data, preparing inputs, training a model and making predictions. Most also include plots to show how the model fits the data.

**Python · pandas · NumPy · Matplotlib · scikit-learn · Jupyter / Google Colab**

## Notebooks at a glance

| Notebook | What it predicts | Main idea |
| :--- | :--- | :--- |
| [Simple Linear Regression](simple_linear_regression.ipynb) | Salary from years of experience | Fit a straight line using one input |
| [Multiple Linear Regression](multiple_linear_regression.ipynb) | Startup profit from spending and state | Use several inputs together |
| [Polynomial Regression](polynomial_regression.ipynb) | Salary from position level | Fit a curve using degree-4 polynomial features |
| [Support Vector Regression](support_vector_regression.ipynb) | Salary from position level | Use an RBF kernel with scaled data |
| [Decision Tree Regression](decision_tree_regression.ipynb) | Salary from position level | Split the input into regions |
| [Random Forest Regression](random_forest_regression.ipynb) | Salary from position level | Combine predictions from 10 trees |

## Datasets

| File | Rows | Inputs used | Target |
| :--- | :---: | :--- | :--- |
| [Salary_Data.csv](Salary_Data.csv) | 30 | YearsExperience | Salary |
| [50_Startups.csv](50_Startups.csv) | 50 | R&D Spend, Administration, Marketing Spend, State | Profit |
| [Position_Salaries.csv](Position_Salaries.csv) | 10 | Level | Salary |

`Position_Salaries.csv` also contains job titles in the `Position` column. The notebooks use the numerical `Level` column as their input. All three CSV files are included; no blank cells were found during inspection.

## What each notebook does

### 1. Simple Linear Regression

Estimates salary from years of experience. It uses an **80% training / 20% testing split**, fits `LinearRegression`, and plots the training and test results. It also predicts salary for **3.9 years** of experience and displays the fitted slope and intercept.

### 2. Multiple Linear Regression

Estimates startup profit from three spending columns and the company's state. `OneHotEncoder` converts state names into numerical columns. The notebook uses an **80/20 split**, trains `LinearRegression`, and prints predicted profit beside actual profit for the ten test rows.

### 3. Polynomial Regression

Compares a straight-line model with a **degree-4 polynomial model** on the position-level dataset. The plots show how powers of the input can capture a curved relationship. Both models predict salary at level **6.5**.

### 4. Support Vector Regression

Uses separate `StandardScaler` objects for position levels and salaries, then fits `SVR(kernel='rbf')`. Predictions are converted back to the original salary scale. The notebook predicts level **6.5** and plots the fitted curve.

### 5. Decision Tree Regression

Fits `DecisionTreeRegressor(random_state=0)` to the position-level dataset. It predicts salary at level **6.5** and uses a fine plotting grid to show the tree's step-like predictions.

### 6. Random Forest Regression

Fits `RandomForestRegressor(n_estimators=10, random_state=0)` to the same dataset. It combines ten trees, predicts salary at level **6.5**, and plots the predictions.

## Example predictions in the saved notebooks

These values are **saved notebook outputs**, rounded to two decimal places. They were inspected for this README, not produced by rerunning the notebooks.

| Model | Example input | Saved prediction |
| :--- | :--- | ---: |
| Simple Linear Regression | 3.9 years of experience | 63,099.14 |
| Linear baseline in the polynomial notebook | Position level 6.5 | 330,378.79 |
| Polynomial Regression, degree 4 | Position level 6.5 | 158,862.45 |
| Support Vector Regression | Position level 6.5 | 170,370.02 |
| Decision Tree Regression | Position level 6.5 | 150,000.00 |
| Random Forest Regression | Position level 6.5 | 167,000.00 |

The multiple-linear notebook displays predicted-versus-actual profit pairs instead of a single example prediction.

**These predictions are not accuracy scores.** The notebooks do not report MAE, RMSE or R², so this repository does not claim that one model is the best.

## Run the notebooks

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Use **File → Open notebook → GitHub** and enter `https://github.com/ShafilMomin/Regression`.
3. Select a notebook.
4. Upload its matching CSV to the Colab Files panel. Opening a notebook from GitHub does not automatically upload its dataset.
5. Run the cells in order from top to bottom.

The polynomial, SVR, decision-tree and random-forest notebooks also contain an **Open in Colab** button.

### Local setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ShafilMomin/Regression.git
cd Regression
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

Install the libraries and start Jupyter:

```bash
python -m pip install numpy pandas matplotlib scikit-learn notebook
python -m notebook
```

Open notebooks from the repository folder so their relative CSV paths work. Package versions are not pinned; saved outputs come from earlier environments, so compatibility with every current release has not been verified.

## Evaluation and learning notes

- Simple and multiple linear regression use a train-test split with `random_state=0`.
- Polynomial, SVR, decision-tree and random-forest examples fit **all 10 rows**. Their plots illustrate model behaviour, not performance on unseen data.
- The multiple-linear notebook fits the state encoder before splitting. For a stronger evaluation, split first and fit preprocessing only on the training data, ideally in a pipeline.
- The saved SVR output includes a target-shape warning. `y.ravel()` provides the one-dimensional target expected by `SVR.fit`.
- Some plotting cells pass array-valued bounds to `np.arange`. If a newer NumPy version rejects these, use scalar bounds such as `X.min()` and `X.max()` in your working copy.
- A fair model comparison needs the same dataset, validation setup and regression metrics. These notebooks are learning examples, not a production prediction service.

## Suggested learning order

**Simple Linear → Multiple Linear → Polynomial → SVR → Decision Tree → Random Forest**

Start with straight-line relationships, then explore curves and tree-based methods. Focus on how predictions and data preparation change between models.
