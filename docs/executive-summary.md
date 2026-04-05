# Executive Summary

## What this project studies

This repository uses the 2019 Buenos Aires Annual Household Survey to explain how income differences are distributed across the city and how those differences behave once the analysis moves from charts into econometrics. It now works as both a territorial inequality diagnostic and an applied labor-economics case.

## Core question

Which patterns appear most strongly when household income is read together with geography, education, and household composition, and how much of the observed gender labor-income gap can be explained by standard observable characteristics?

## Working hypotheses

- `H1`: income differences across Buenos Aires are structured by territory and household composition rather than randomly distributed.
- `H2`: schooling has a meaningful positive association with labor income in the employed sample.
- `H3`: the observed gender income gap is only partly explained by schooling, experience, and occupational sorting.

## What the upgraded analysis shows

- The descriptive pipeline aggregates the microdata into `5,848` households across `15` communes.
- Commune `14` reaches a median household income of `70,000`, while Commune `8` sits at `35,000`, a clean `2.0x` gap.
- The econometric sample includes `6,656` employed adults with positive labor income.
- In the semilog Mincer equation, each additional year of schooling is associated with a `13.25%` labor-income premium, holding potential experience, commune, occupational category, and gender constant.
- The estimated male-female labor-income gap in the working sample is `29.35%`.
- The Oaxaca-Blinder decomposition is the most interesting result: the explained component is `-7.77%`, while the unexplained component is `40.25%`. That means the raw gap is not coming from women in the sample having weaker measured endowments. If anything, the gap is driven by different returns to similar characteristics.

## Why this matters

That last point makes the repository materially stronger. A descriptive notebook can show that income differs across groups; an econometric layer can show whether those differences still persist after controlling for the basics. That shift matters for:

- urban analytics roles;
- public-policy and labor-market research teams;
- consulting work related to inequality, workforce diagnostics, or service targeting.

## Main conclusion

The strongest conclusion is no longer just that richer communes exist. The portfolio-quality conclusion is that Buenos Aires shows both territorial inequality and a labor-market gap by gender that is not explained away by standard observed characteristics. In other words, the project now says something more analytical than "the bars look different": it quantifies the schooling premium and separates the gender gap into explained and unexplained components.

## Important caveats

- The survey is cross-sectional, so the coefficients should be interpreted as associations rather than causal effects.
- Potential experience is constructed rather than directly observed.
- The decomposition is only as strong as the set of available covariates.

## Portfolio takeaway

This repository now works better as a professional case because it combines descriptive clarity, econometric discipline, and a conclusion that is both policy-relevant and easy to explain in an interview.
