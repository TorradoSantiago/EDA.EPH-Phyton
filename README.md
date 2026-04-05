# EDA.EPH-Phyton

Portfolio-grade socioeconomic analysis built on the 2019 Buenos Aires Annual Household Survey. The repository now combines descriptive diagnostics, labor-economics modeling, and a compact machine-learning benchmark, but it also preserves the original Spanish delivery so the stronger theoretical framing is not lost.

Start with:
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/household-survey-brief.pdf`
- `archive/original_delivery/original_theoretical_notes_es.md`

## Recovered original delivery

The original project submission is now explicitly preserved in `archive/original_delivery/`:

- `Datasets+Torrado.ipynb`: original notebook restored with its original filename from the first upload commit
- `original_theoretical_notes_es.md`: extracted Markdown narrative from the original Spanish notebook
- `recovery_manifest.md`: provenance note explaining what was recovered and from which commit

This means the current public-facing version and the original delivery can now coexist in the same repository.

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
6. Benchmark four out-of-sample ML models on log labor income.
7. Export recruiter-friendly figures and a PDF brief that explains the results in plain language while preserving the original theoretical notebook as archive material.

## Why it works in a portfolio

This repository now shows four things at once:
- comfort with messy public microdata;
- ability to move from description to econometric reasoning;
- enough ML awareness to test whether more complexity actually improves the story;
- respect for the original academic framing, hypotheses, and theoretical context of the first delivery.
