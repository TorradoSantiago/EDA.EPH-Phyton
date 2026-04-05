# EDA.EPH-Phyton

Portfolio-grade socioeconomic analysis built on the 2019 Buenos Aires Annual Household Survey. The repository now combines descriptive diagnostics with a real econometric layer, so it reads less like an exploratory class exercise and more like an applied labor-economics case.

Start with:
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/household-survey-brief.pdf`

## Snapshot

- Main source: 2019 Annual Household Survey for Buenos Aires City
- Descriptive sample: `5,848` households across `15` communes
- Econometric sample: `6,656` employed adults with positive labor income
- Headline findings:
  - each additional year of schooling is associated with a `13.25%` labor-income premium;
  - the male-female labor-income gap is `29.35%` in the working sample;
  - the Oaxaca-Blinder decomposition shows a negative explained component and a large unexplained component, which suggests the gap is not driven by observed endowments alone;
  - Commune `14` doubles Commune `8` in median household income (`70,000` vs `35,000`).

## Main questions

- How does income vary across Buenos Aires communes?
- How does household composition relate to household and per-capita income?
- What is the estimated return to schooling in the survey's employed population?
- How much of the observed gender income gap can be explained by measured characteristics?

## Analytical workflow

1. Clean and normalize the household survey microdata.
2. Aggregate the person-level survey into household summaries for territorial diagnostics.
3. Export reusable descriptive tables and figures for communes, education, and marital status.
4. Estimate a semilog Mincer equation on employed adults with positive labor income.
5. Run a two-fold and three-fold Oaxaca-Blinder decomposition by gender.
6. Export charts and summary tables that can be reused in README files, interviews, or LinkedIn posts.

## Key findings

- The descriptive layer still shows a sharp territorial gradient: higher-income communes also tend to have smaller households and stronger per-capita income.
- In the econometric layer, one additional year of schooling is associated with roughly `13.25%` higher labor income, controlling for potential experience, commune, occupational category, and gender.
- Men in the labor subsample earn about `29.35%` more than women on average.
- The Oaxaca-Blinder results are especially interesting: the explained component is slightly negative (`-7.77%`), while the unexplained component is strongly positive (`40.25%`). In plain language, women in this sample do not appear to earn less because they bring weaker observable characteristics to the labor market. The gap is mainly associated with different returns to similar characteristics.

## Main files

- `household_survey_analysis.ipynb`: original English notebook companion
- `archive/household_survey_analysis_legacy_es.ipynb`: preserved original Spanish notebook
- `annual_household_survey_2019.csv`: source dataset
- `BRIEF.md`: short explanation for non-technical readers
- `docs/executive-summary.md`: portfolio-facing summary with findings and interpretation
- `docs/household-survey-brief.html`: print-ready brief for non-technical readers
- `docs/household-survey-brief.pdf`: portfolio-ready PDF version of the brief
- `scripts/household_analysis.py`: reproducible pipeline that now exports both descriptive and econometric outputs
- `outputs/tables/mincer_coefficients.csv`: key semilog regression coefficients
- `outputs/tables/oaxaca_two_fold_summary.csv`: explained and unexplained gap components
- `outputs/tables/oaxaca_three_fold_summary.csv`: endowment, coefficient, and interaction effects
- `outputs/figures/predicted_labor_income_by_schooling_gender.png`: predicted income profile by schooling and gender
- `outputs/figures/gender_income_gap_decomposition.png`: explained versus unexplained gender-gap visualization

## How to read the repository

1. Start with `BRIEF.md` for the non-technical version.
2. Continue with `docs/executive-summary.md` for the portfolio story.
3. Open `docs/household-survey-brief.pdf` if you want a polished presentation asset first.
4. Run `scripts/household_analysis.py` if you want to reproduce the tables and figures.
5. Open the notebook if you want to inspect the original exploratory path that led to the cleaned portfolio layer.

## How to run the script

```bash
pip install -r requirements.txt
python scripts/household_analysis.py
```

The script writes descriptive and econometric tables to `outputs/tables/` and figures to `outputs/figures/`.

## Limitations

- This is a cross-sectional survey, so the econometric layer should be read as associative evidence, not causal proof.
- Potential experience is a proxy derived from age and schooling, which is standard in Mincer-style models but still imperfect.
- The gender-gap decomposition is only as rich as the observed covariates in the survey.

## Why it works in a portfolio

This repository now shows three things at once:
- comfort with messy public microdata;
- ability to move from descriptive analysis to formal econometric reasoning;
- judgment in translating a statistical result into a credible socioeconomic argument.

That combination is especially strong for analytics consulting, labor economics, policy, and data-driven strategy roles.
