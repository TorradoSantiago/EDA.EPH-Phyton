# Income Inequality in Buenos Aires
**Annual Household Survey 2019 · Python**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

Buenos Aires is one of Latin America's most unequal cities. This project uses public microdata from the 2019 Annual Household Survey (EAH) to answer a simple question: **who earns more, and why?**

The analysis finds that where you live, how much you studied, how big your household is, and your gender all independently predict your income — and that a well-specified statistical model already captures almost all of that structure, with no need for machine learning.

**Four findings stand out:**

- A university graduate earns roughly **3× more** than someone with primary school only
- The commune you live in explains a **2× income gap** across the city (Palermo vs. Villa Soldati)
- Women earn **29% less** than men on average — and even after controlling for education, experience, and occupation, **35% of that gap remains unexplained**
- A Gradient Boosting model trained on the same data **doesn't beat the linear regression**

> For the full technical writeup — econometric specification, Gauss-Markov diagnostics, decomposition math, and model tables — see [`docs/household-survey-brief.pdf`](docs/household-survey-brief.pdf).

---

## Structure

```
├── household_survey_analysis.ipynb   ← start here
├── analysis_final.py                 ← regenerates all 14 figures
├── annual_household_survey_2019.csv  ← raw data (EAH 2019)
├── outputs/figures/                  ← 14 publication-quality charts
├── outputs/tables/                   ← model results as CSV
└── docs/household-survey-brief.pdf   ← 12-page technical brief (LaTeX)
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook household_survey_analysis.ipynb
# or: python analysis_final.py  ← figures only
```

## Methods

| Step | What | Why |
|------|------|-----|
| Descriptive EDA | 14 charts across geography, education, gender | Understand the data before modeling |
| Hypothesis tests | Kruskal-Wallis, Mann-Whitney U, Spearman ρ | Non-parametric — income is right-skewed |
| OLS regression | Log-linear earnings equation | Interpretable coefficients, economic theory |
| HC3 robust SEs | Correct for heteroskedasticity | Breusch-Pagan test rejects homoskedasticity |
| Decomposition | Two-fold gender gap breakdown | Separate explained vs. structural gap |
| ML benchmark | Random Forest + Gradient Boosting vs. OLS | Robustness check on the linear spec |

**Author:** Santiago Torrado — Applied economics, data analysis, public policy
