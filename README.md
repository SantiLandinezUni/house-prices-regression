# ML Assignment 1 — House Prices: Advanced Regression Techniques

Kaggle competition: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>

**Task.** Predict the sale price of residential homes in Ames, Iowa, from 79
explanatory variables. This is a supervised **regression** problem.

**Evaluation metric.** RMSE on `log(SalePrice)` — i.e. Kaggle scores
`sqrt(mean((log(y_pred) - log(y_true))^2))`. Errors on cheap and expensive
houses therefore count equally in *relative* terms.

## Project structure

```
ML_Assignment_1/
├── data/
│   ├── raw/            # train.csv, test.csv, sample_submission.csv (from Kaggle)
│   └── processed/      # intermediate cleaned datasets
├── notebooks/          # main deliverable notebook
├── src/                # reusable functions: loading, cleaning, features
├── models/             # saved model artefacts
├── submissions/        # candidate submission CSVs
├── report/             # ~1 page report
│   └── figures/        # plots exported from the notebook
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

The environment is a **named conda environment** called `ML_House_Prices`,
living in `/opt/anaconda3/envs/`. Conda is used rather than a pip `venv` so it
can manage the compiled dependencies as well as the Python ones -- notably
`llvm-openmp`, which XGBoost requires on macOS.

```bash
conda create -n ML_House_Prices python=3.11 -y
conda activate ML_House_Prices
conda config --env --set channel_priority strict
conda install -c conda-forge -y pandas numpy matplotlib seaborn scikit-learn xgboost jupyter ipykernel
python -m ipykernel install --user --name ml_house_prices --display-name "Python (ML_House_Prices)"
```

Reproducibility files are regenerated after install:

```bash
conda env export --from-history -n ML_House_Prices > environment.yml
pip list --format=freeze > requirements.txt
```

Then in VS Code: open a notebook -> **Select Kernel** -> **Python (ML_House_Prices)**.

## Data

Download from the competition's **Data** tab (rules must be accepted first) and
place `train.csv`, `test.csv` and `sample_submission.csv` in `data/raw/`.
The data itself is not committed to version control.

## Deliverable

A `.zip` named with the Kaggle user ID, containing:
1. the ~1 page report,
2. the completed notebook,
3. the best submission `.csv`.
