# EDA.EPH-Phyton

Portfolio-grade socioeconomic analysis built on the 2019 Buenos Aires Annual Household Survey. The repository now combines descriptive diagnostics, formal econometric tests, Oaxaca-Blinder decomposition, and a disciplined ML benchmark, while preserving the original Spanish delivery so the stronger theoretical framing is never lost.

Start with:
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/household-survey-brief.pdf`
- `archive/original_delivery/original_theoretical_notes_es.md`

## Recovered original delivery

The original project submission is preserved in `archive/original_delivery/`:

- `Datasets+Torrado.ipynb`: original notebook restored with its original filename from the first upload commit.
- `original_theoretical_notes_es.md`: extracted Markdown narrative from the original Spanish notebook.
- `recovery_manifest.md`: provenance note explaining what was recovered and from which commit.

This matters because the public-facing portfolio version can now coexist with the original academic delivery instead of replacing it.

## Research question

The project studies how household income, educational attainment, labor-market position, and gender interact inside Buenos Aires microdata. It is not only an inequality dashboard. The analytical purpose is to test whether the main gradients suggested by descriptive evidence survive formal modeling and whether more flexible machine-learning models materially improve what we learn from the survey.

## Why the dataset is theoretically useful

The Annual Household Survey is valuable because it combines territorial information, labor outcomes, educational attainment, household composition, and demographic structure in a single urban snapshot. In the original delivery, that context was one of the strongest parts of the project, and it remains central here.

The survey is theoretically useful for three reasons:

1. It lets us connect territorial inequality with labor-market inequality.
2. It lets us move from household outcomes to individual earnings equations.
3. It allows a gender-gap analysis that separates what is explained by observed characteristics from what remains unexplained.

## Analytical structure

The current version has four stacked layers.

1. Descriptive baseline
   - Household aggregation by commune.
   - Income gradients by education and marital status.
   - Household-size relationships with per-capita resources.

2. Statistical testing
   - Kruskal-Wallis across communes: `p = 1.36e-72`.
   - Spearman correlation between household members and per-capita income: `rho = -0.388`.
   - Kruskal-Wallis across education groups: overwhelmingly significant.
   - Mann-Whitney gender contrast in labor income: strongly significant.

3. Econometric layer
   - Semilog Mincer equation with schooling, centered experience, occupation, commune, and gender.
   - Oaxaca-Blinder decomposition of the male-female labor-income gap.
   - Bootstrap confidence intervals for the two-fold decomposition.

4. Machine-learning robustness layer
   - Linear benchmark.
   - Elastic Net.
   - Random Forest.
   - Gradient Boosting.
   - Five-fold cross-validation to compare out-of-sample stability.

## Core sample and results

- Descriptive sample: `5,848` households across `15` communes.
- Econometric sample: `6,656` employed adults with positive labor income.
- Main earnings result: each additional year of schooling is associated with a `13.25%` labor-income premium.
- Adjusted `R^2` of the Mincer equation: `0.378`.
- Raw gender labor-income gap: `29.35%`.
- Bootstrap unexplained component of the gender gap: `40.18%` with a 95% interval that stays well above zero.

## Diagnostic interpretation

The project is stronger now because it does not stop at coefficient reporting.

- Heteroskedasticity is present, so the main coefficient table is reported with `HC3` robust errors.
- The RESET test does not indicate a major omitted-nonlinearity problem in the Mincer equation once centered experience is included.
- Jarque-Bera rejects residual normality, which is common in earnings data and reinforces the decision to use robust inference.
- Centering the experience polynomial reduces VIF values to roughly `1.01` to `1.15`, which means the polynomial term no longer creates artificial collinearity.

## What the ML benchmark really says

The ML layer is intentionally framed as a robustness exercise rather than a replacement for theory.

- Best holdout model: `Gradient Boosting`, `R^2 = 0.342`.
- Best cross-validated model: `Gradient Boosting`, `CV R^2 = 0.346`.
- Linear benchmark: `CV R^2 = 0.339`.

That gap is small. This is an analytically important result: in this dataset, most of the useful structure is already captured by an interpretable specification. That is a better story than claiming that a black-box model changes everything.

## Theoretical proposal

The repository now supports a clearer theoretical claim than the original EDA alone.

Income inequality in Buenos Aires should be understood as a layered process:

- geography structures access to opportunities and cost of living;
- schooling shifts expected labor-market returns;
- household composition changes how gross income translates into effective welfare;
- gender differences remain even after conditioning on observed characteristics.

The project therefore moves from simple description to a compact theory of urban inequality: income differences are not only territorial and not only educational; they are reproduced through interacting labor-market, household, and gender mechanisms.

## Conclusion

This repository works as a portfolio case because it now shows judgment in addition to technique.

- It starts with a real public dataset.
- It formalizes the strongest descriptive patterns with tests.
- It estimates an interpretable earnings equation.
- It uses decomposition analysis to say something substantive about inequality.
- It adds ML only where ML helps clarify robustness.

That mix is useful for consulting, economic analysis, labor-market research, public-policy analytics, and strategy roles where explanation matters as much as prediction.
