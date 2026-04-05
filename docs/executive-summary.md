# Executive Summary

## Abstract

This project uses the 2019 Buenos Aires Annual Household Survey to study how income differences are structured across the city. The original delivery framed the dataset as a public-policy instrument for understanding labor, education, household composition, and wellbeing in Buenos Aires. That framing is preserved here because it makes the case stronger: the repository is not just about charts, but about extracting interpretable evidence from a socially relevant survey.

## Research objective

The core objective is to identify which variables appear most closely associated with family and labor income in Buenos Aires and to test whether those relationships remain informative once the analysis moves from descriptive exploration into econometrics and ML benchmarking.

## Research questions and hypotheses

- `H1`: income differences across Buenos Aires are structured by territory rather than randomly distributed.
- `H2`: education has a meaningful positive association with labor income in the employed sample.
- `H3`: household composition and demographic structure change the interpretation of income differences.
- `H4`: the observed gender labor-income gap is only partly explained by standard observed characteristics.
- `H5`: non-linear ML models may improve fit, but not necessarily enough to displace a strong interpretable baseline.

## Public-policy and business context

The survey is valuable because it helps answer practical questions that matter to policymakers, analysts, and consulting teams: where income pressure is strongest, how household size interacts with welfare, and whether apparently simple gaps still persist after controlling for the basics. In the original notebook, this was presented as a problem of improving social, educational, and labor policy design. That logic still holds and gives the repository a more serious purpose than a generic EDA exercise.

## Analytical approach

The project now combines three layers:

1. A descriptive household-level diagnostic that aggregates the microdata into `5,848` households across `15` communes.
2. A labor-economics layer that estimates a semilog Mincer equation and runs an Oaxaca-Blinder decomposition by gender on `6,656` employed adults with positive labor income.
3. A compact ML layer that benchmarks four out-of-sample models on log labor income to test whether more flexible prediction materially improves the story.

## What the evidence shows

- Commune `14` reaches a median household income of `70,000`, while Commune `8` sits at `35,000`, a clean `2.0x` gap.
- In the semilog Mincer equation, each additional year of schooling is associated with a `13.25%` labor-income premium.
- The estimated male-female labor-income gap in the working sample is `29.35%`.
- The Oaxaca-Blinder decomposition remains one of the strongest findings: the explained component is `-7.77%`, while the unexplained component is `40.25%`.
- The best ML benchmark is a tuned Gradient Boosting model with `R^2 = 0.342`, only modestly above the linear benchmark at `0.330`.

## Interpretation

That last point matters. The repository is strongest not because it has ML, but because it shows when ML does and does not add value. Here, the extra non-linearity slightly improves fit, but it does not overturn the main economic interpretation. The bulk of the signal is already captured by an interpretable framework built around schooling, experience, geography, household structure, and gender.

## Main conclusion

The portfolio-quality conclusion is that Buenos Aires shows both territorial inequality and a labor-market gap by gender that is not explained away by standard observed characteristics, while more complex predictive models add only modest incremental fit. That is a mature result because it combines theory, measurement, and methodological restraint.

## Original delivery preservation

The original Spanish notebook and its theoretical narrative are explicitly preserved in `archive/original_delivery/`. That archive matters because it keeps the first academic framing visible instead of replacing it with a thinner portfolio summary.
