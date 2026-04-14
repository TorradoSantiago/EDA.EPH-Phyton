# Income Inequality in Buenos Aires
### Annual Household Survey 2019 — EDA, OLS Regression, and ML Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What this project does

This repository turns public microdata from the 2019 Buenos Aires Annual Household Survey into an applied case on labor and urban economics. The central question is not simply *who earns more* — it is **why income differences exist** across geography, education levels, household types, and gender, and whether a well-specified statistical model can capture those differences.

The analysis moves through four layers:

1. **Descriptive EDA** — documenting income gradients with 14 charts
2. **Formal hypothesis tests** — confirming each pattern is statistically real
3. **OLS earnings regression** — estimating the independent contribution of schooling, experience, and gender with full Gauss-Markov diagnostics
4. **ML benchmark** — testing whether non-linear models add meaningful predictive power over OLS

---

## Repository structure

```
EDA.EPH-Phyton/
├── household_survey_analysis.ipynb   ← Main notebook (start here)
├── analysis_final.py                 ← Standalone pipeline to regenerate all figures
├── annual_household_survey_2019.csv  ← Raw EAH 2019 dataset
├── outputs/
│   ├── figures/                      ← 14 publication-quality charts
│   └── tables/                       ← CSV exports of all model results
├── docs/
│   └── household-survey-brief.pdf    ← Academic brief with embedded figures (LaTeX)
└── archive/
    └── original_delivery/            ← Original Spanish academic submission
```

---

## Research questions and hypotheses

| # | Question | Null hypothesis (H0) | Test | Result |
|---|---|---|---|---|
| H1 | Does geography determine income? | All communes have identical distributions | Kruskal-Wallis | p = 1.36e-72 — **H0 rejected** |
| H2 | Do larger households have lower per-capita income? | Spearman rho = 0 | Spearman | rho = -0.388 — **H0 rejected** |
| H3 | Does education stratify income? | Education groups are identically distributed | Kruskal-Wallis | p << 0.001 — **H0 rejected** |
| H4 | Does a raw gender wage gap exist? | Male/female distributions are identical | Mann-Whitney U | p << 0.001 — **H0 rejected** |

Non-parametric tests are used because income data is right-skewed and does not satisfy the normality assumption required by ANOVA or Student's t-test.

---

## Earnings model

The core model is a log-linear OLS regression on 6,656 employed adults:

```
ln(labor_income) = b0 + b1*schooling + b2*experience_c + b3*experience_c^2 + b4*female + controls + error
```

Experience is **centered** before squaring to reduce multicollinearity between the linear and quadratic terms. This brings VIF values to the 1.01–1.15 range with no material change to the estimates.

Full Gauss-Markov diagnostics:

| Condition | Test | Result | Action |
|---|---|---|---|
| Homoskedasticity | Breusch-Pagan | Rejected | HC3 robust standard errors |
| No multicollinearity | VIF | VIF 1.01–1.15 after centering | Centered experience polynomial |
| Normal residuals | Jarque-Bera | Rejected | Expected; large-n inference valid |
| Correct functional form | RESET (Ramsey) | Not rejected | Specification is adequate |

**Main result:** each additional year of schooling is associated with a **+13.25% increase in monthly labor income**, controlling for experience, occupation, location, and gender. Adjusted R2 = 0.378.

---

## Gender wage gap decomposition

The 29.4% raw gender gap is split into two components:

- **Explained** (~-6.5%): attributable to differences in observable characteristics. Observable female characteristics are actually slightly favorable in this sample, so this component is negative.
- **Unexplained** (~38%): the gap in returns to those characteristics — what women earn less even after equalizing education, experience, and occupation. The 95% bootstrap confidence interval stays firmly above zero.

---

## ML benchmark

Four models compared with 5-fold cross-validation:

| Model | CV R2 |
|---|---|
| Linear Regression (OLS baseline) | 0.362 |
| Elastic Net | -0.007 |
| Random Forest | 0.331 |
| **Gradient Boosting** | **0.342** |

Gradient Boosting does not beat the OLS baseline in cross-validated performance. This is the analytically important result: the economic specification already captures the key structure of income determination. Non-linear complexity adds no meaningful predictive power.

---

## Main findings

- **Schooling premium:** +13.25% per additional year (OLS, HC3 errors)
- **Adjusted R2:** 0.378
- **Raw gender gap:** 29.4% (median, labor income)
- **Unexplained gender gap:** ~38% of the total, well above zero in bootstrap CI
- **Geographic inequality:** commune 14 (Palermo) has ~2x the median income of commune 8 (Villa Soldati)
- **Household dilution:** Spearman rho = -0.388 between household size and per-capita income

---

## Stack

Python 3 · pandas · numpy · statsmodels · scikit-learn · matplotlib · scipy

---

## Original submission

The original Spanish-language academic delivery is preserved in `archive/original_delivery/`. It coexists with this portfolio version without being replaced.

---

**Author:** Santiago Torrado — Applied economics, data analysis, public policy
