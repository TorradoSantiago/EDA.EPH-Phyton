# Executive Brief

## In one sentence

Applied urban and labor-market case on Buenos Aires survey microdata, now strengthened with formal distribution tests, robust-error econometrics, bootstrap decomposition, and cross-validated ML benchmarks.

## What problem it addresses

The project asks where income differences appear across the city, how household composition changes welfare, how strongly schooling maps into labor income, and how much of the gender income gap remains after controlling for observable characteristics.

## Why it is stronger now

- The descriptive layer is no longer just visual; it is backed by formal statistical tests.
- The Mincer model now uses centered experience and diagnostic checks.
- The Oaxaca-Blinder result now has bootstrap confidence intervals.
- The ML comparison now includes five-fold cross-validation, not only one train/test split.

## What a non-technical reader can see

- A relevant public dataset used to study inequality in Buenos Aires.
- A clear education premium of `13.25%` per additional year of schooling.
- A large gender gap that remains materially unexplained after controls.
- A useful insight that ML improves fit only slightly over the interpretable baseline.

## Technical highlights

- Kruskal-Wallis across communes: `p = 1.36e-72`.
- Spearman household-size vs per-capita income: `rho = -0.388`.
- Adjusted `R^2` of the earnings equation: `0.378`.
- VIF after centering experience: approximately `1.01` to `1.15`.
- Bootstrap unexplained gender-gap component: `40.18%`.
- Best cross-validated ML benchmark: `Gradient Boosting`, `CV R^2 = 0.346`.

## Current conclusion

The project now tells a better story: urban inequality in Buenos Aires is territorial, educational, household-based, and gendered at the same time. The machine-learning layer does not overturn that conclusion; it confirms that the interpretable structure already captures most of the signal.

## Best next step

Extend the same logic with a second wave of data or add a subgroup comparison by age cohort to test whether schooling returns and the gender gap vary across generations.
