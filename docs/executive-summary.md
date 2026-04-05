# Executive Summary

## Project purpose

This repository turns the 2019 Annual Household Survey of Buenos Aires into an applied case on inequality, labor income, and household welfare. The original submission already had a strong theoretical instinct: the survey mattered because it made it possible to connect demographic structure, labor-market conditions, education, and geography inside the same urban dataset. The current version keeps that spirit while making the methodology more rigorous.

## Conceptual base

The analytical base of the project is that income is not a single-dimensional outcome. In urban microdata, household wellbeing is shaped by at least four simultaneous forces:

1. Territorial segmentation across communes.
2. Human-capital differences captured by schooling.
3. Household composition, especially when total income must be translated into per-capita welfare.
4. Labor-market inequalities that persist across gender groups.

That means the project is not simply asking who earns more. It is asking how urban structure, labor-market rewards, and household demographics combine to produce unequal outcomes.

## Data and sample

- Source: 2019 Annual Household Survey for Buenos Aires City.
- Household aggregation sample: `5,848` households.
- Earnings-model sample: `6,656` employed adults aged `18` to `65` with positive labor income.

## Methodological upgrade

The repository now has four methodological layers.

### 1. Descriptive evidence

Income is summarized by commune, education level, marital status, and household structure. This preserves the original logic of starting from a readable social profile before fitting models.

### 2. Formal statistical tests

The descriptive story is now backed by explicit tests:

- Kruskal-Wallis confirms large commune-level income differences.
- Spearman correlation shows that larger households tend to have lower per-capita income.
- Kruskal-Wallis across education groups confirms strong income stratification by schooling.
- Mann-Whitney shows that male and female labor-income distributions differ materially before controls.

### 3. Econometric interpretation

A semilog Mincer equation estimates the return to schooling while controlling for centered experience, occupation, commune, and gender. Centering the experience polynomial reduces multicollinearity substantially and makes the coefficient interpretation cleaner.

Key diagnostics are also exported:

- heteroskedasticity tests,
- RESET functional-form check,
- Jarque-Bera residual check,
- VIF table for the core regressors.

### 4. ML robustness

The ML layer is treated as a benchmark, not as a substitute for interpretation. Linear regression, Elastic Net, Random Forest, and Gradient Boosting are compared both on a holdout split and under five-fold cross-validation.

## Main empirical findings

- Commune `14` shows roughly double the median household income of Commune `8`.
- Each additional year of schooling is associated with a `13.25%` labor-income premium.
- The male-female labor-income gap is `29.35%` in the working sample.
- The bootstrap mean of the unexplained component in the Oaxaca two-fold decomposition is `40.18%`.
- The best holdout ML model is `Gradient Boosting` with `R^2 = 0.342`.
- The best cross-validated ML result is also `Gradient Boosting` with `CV R^2 = 0.346`.

## Interpretation

The most useful analytical conclusion is not that ML wins. The useful conclusion is that ML wins only marginally. That means the citywide inequality story is already well captured by an interpretable combination of geography, education, labor-market position, and household structure. This is exactly the kind of result that is persuasive in consulting or policy settings, because it supports explanation rather than only prediction.

## Theoretical contribution

The repository now supports a stronger claim than the original EDA alone: urban inequality in Buenos Aires is reproduced through interacting mechanisms rather than a single source. Commune effects matter, but so do human capital, household dilution of resources, and a gender gap that remains large even after controls. The project therefore moves from description to a compact explanation of how inequality is generated and maintained inside a metropolitan labor market.

## Conclusion

This is now a stronger portfolio project because it combines social relevance, statistical discipline, econometric reasoning, and restrained use of machine learning. It keeps the better academic framing of the original delivery while making the outputs clearer for professional readers who need both rigor and narrative.
