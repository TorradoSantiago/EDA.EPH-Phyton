# EDA.EPH-Phyton

Portfolio-grade socioeconomic analysis built on the 2019 Buenos Aires Annual Household Survey. The repository now combines descriptive diagnostics, labor-economics modeling, and a compact machine-learning benchmark so the case is useful both for policy conversations and for analytics hiring screens.

Start with:
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/household-survey-brief.pdf`

## Snapshot

- Main source: 2019 Annual Household Survey for Buenos Aires City
- Descriptive sample: `5,848` households across `15` communes
- Econometric sample: `6,656` employed adults with positive labor income
- Main econometric findings:
  - each additional year of schooling is associated with a `13.25%` labor-income premium;
  - the observed male-female labor-income gap is `29.35%`;
  - the unexplained component in the Oaxaca-Blinder decomposition reaches `40.25%`.
- Main ML finding:
  - a tuned Gradient Boosting model reaches `R^2 = 0.342` on log labor income, only modestly above the linear benchmark (`0.330`).

## Main questions

- How does income vary across Buenos Aires communes?
- How does household composition change the interpretation of income differences?
- What is the estimated return to schooling in the employed sample?
- How much of the gender income gap can be explained by measured characteristics?
- Do non-linear ML models materially outperform an interpretable labor-economics baseline?

## Analytical workflow

1. Clean and normalize the household survey microdata.
2. Aggregate the person-level survey into household summaries.
3. Export reusable descriptive tables and figures for communes, education, and marital status.
4. Estimate a semilog Mincer equation on employed adults with positive labor income.
5. Run a two-fold and three-fold Oaxaca-Blinder decomposition by gender.
6. Benchmark four out-of-sample ML models on log labor income:
   - Linear benchmark
   - Elastic Net
   - Random Forest
   - Gradient Boosting
7. Export recruiter-friendly figures and a PDF brief that explains the results in plain language.

## Key findings

- The descriptive layer still shows a sharp territorial pattern: Commune `14` doubles Commune `8` in median household income (`70,000` vs `35,000`).
- The econometric layer is strong and statistically significant: one additional year of schooling is associated with `13.25%` higher labor income after controls.
- The Oaxaca-Blinder result is especially interesting for interviews: women in the sample do not earn less because they bring weaker measured characteristics. The large unexplained component suggests the gap is mainly associated with different returns to similar characteristics.
- The ML layer adds a useful stress test. A tuned Gradient Boosting model does slightly better than the linear benchmark, but not by much. That is a good result, not a bad one: it suggests the strongest signal is already captured by a structured, interpretable specification.

## Main files

- `household_survey_analysis.ipynb`: original English notebook companion
- `archive/household_survey_analysis_legacy_es.ipynb`: preserved original Spanish notebook
- `annual_household_survey_2019.csv`: source dataset
- `BRIEF.md`: short explanation for non-technical readers
- `docs/executive-summary.md`: portfolio-facing summary with findings and interpretation
- `docs/household-survey-brief.html`: printable executive brief with charts
- `docs/household-survey-brief.pdf`: recruiter-friendly PDF version of the brief
- `scripts/household_analysis.py`: reproducible descriptive, econometric, and ML pipeline
- `outputs/tables/mincer_coefficients.csv`: key regression coefficients
- `outputs/tables/oaxaca_two_fold_summary.csv`: explained and unexplained gap components
- `outputs/tables/ml_model_comparison.csv`: out-of-sample ML benchmark table
- `outputs/tables/ml_best_params.csv`: chosen hyperparameters for the best ML model
- `outputs/figures/ml_model_performance.png`: visual comparison of ML benchmarks
- `outputs/figures/ml_feature_importance.png`: most important feature groups in the best ML model

## How to run the script

```bash
pip install -r requirements.txt
python scripts/household_analysis.py
```

The script writes descriptive, econometric, and ML outputs to `outputs/tables/` and `outputs/figures/`.

## Limitations

- This is a cross-sectional survey, so the econometric layer should be read as associative evidence, not causal proof.
- The ML benchmark is a robustness layer, not a substitute for interpretation.
- The Oaxaca decomposition would benefit from bootstrap inference in a next iteration.

## Why it works in a portfolio

This repository now shows three things at once:
- comfort with messy public microdata;
- ability to move from description to econometric reasoning;
- enough ML awareness to test whether more complexity actually improves the story.

That mix is especially strong for analytics consulting, labor economics, public policy, and strategy roles.
