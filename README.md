# EDA.EPH-Phyton

Exploratory analytics case built on the 2019 Buenos Aires Annual Household Survey. The repository turns a high-value public dataset into an interpretable story about income, geography, education, and household structure.

Start with:
- `BRIEF.md`
- `docs/executive-summary.md`
- `docs/household-survey-brief.pdf`

## Snapshot

- Main source: 2019 Annual Household Survey for Buenos Aires City
- Core question: which variables appear most closely associated with household income differences?
- Portfolio value: public data, socially relevant questions, and an applied analytics angle

## Main questions

- How does income vary across communes?
- How does household composition relate to household and per-capita income?
- What patterns appear across marital status, education, and income?
- How useful is a regression-oriented approach for organizing these relationships?

## Why this project matters

This repository is more than a technical exercise. It works as a base for thinking about urban inequality, territorial differences, and socioeconomic conditions in Buenos Aires, which makes it a strong fit for analytics, policy, and applied economics positioning.

## Main files

- `household_survey_analysis.ipynb`: main exploration notebook
- `archive/household_survey_analysis_legacy_es.ipynb`: preserved original Spanish exploratory notebook
- `annual_household_survey_2019.csv`: source dataset
- `BRIEF.md`: short explanation for non-technical readers
- `docs/executive-summary.md`: executive reading of the case
- `docs/household-survey-brief.html`: print-ready brief for non-technical readers
- `docs/household-survey-brief.pdf`: portfolio-ready PDF version of the brief
- `scripts/household_analysis.py`: reusable pipeline for exportable tables and charts

## How to read the repository

1. Start with `BRIEF.md`.
2. Continue with `docs/executive-summary.md`.
3. Open `docs/household-survey-brief.pdf` if you want a presentation-ready version first.
4. Open `household_survey_analysis.ipynb` for the clean English notebook companion.
5. Review `archive/household_survey_analysis_legacy_es.ipynb` if you want the preserved original notebook.
6. Run the script if you want reusable outputs outside the notebook.

## How to run the script

```bash
python scripts/household_analysis.py
```

The script writes tables to `outputs/tables/` and figures to `outputs/figures/`.

## Dependencies

See `requirements.txt`.

## Current status

The repository now has a stronger English analysis layer, reproducible outputs, and a PDF brief for non-technical readers. The next improvement opportunity is a cleaner notebook structure that matches the quality of the exported analysis.
