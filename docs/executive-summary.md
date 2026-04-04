# Executive Summary

## What this project studies

This project uses the 2019 Buenos Aires Annual Household Survey to explain how income differences are distributed across the city. The repository now goes beyond exploratory charts and produces reusable English outputs that can support a portfolio case, an applied-policy conversation, or a consulting-style diagnostic.

## Core question

Which patterns appear most strongly when household income is read together with geography, household size, education, and family structure?

## Working hypothesis

- `H1`: income differences are not random across Buenos Aires; they are structured by territory and household composition.
- `H2`: higher educational attainment is associated with higher reported labor and household income, but the survey categories should be interpreted with care.

## What the upgraded analysis shows

- The pipeline aggregates the microdata into `5,848` households across `15` communes.
- The highest median household income appears in Commune `14` (`70,000`), while Commune `8` closes the distribution at `35,000`.
- That creates a clean `2.0x` gap in median household income between the top and bottom commune.
- The per-capita gap is even sharper: Commune `14` reaches `37,000`, while Commune `8` is close to `12,500`.
- Household size matters to that story. The highest-income commune averages `2.15` members per household, versus `3.21` in Commune `8`.

## Why this matters

The project becomes more credible once income is framed as a territorial and demographic pattern instead of a loose collection of charts. That makes the repository more useful for:

- urban analytics roles;
- public-policy or social-research teams;
- consulting work related to inequality, service access, or territorial diagnostics.

## Main conclusion

The strongest conclusion is not simply that some communes are richer than others. The more useful finding is that household income, per-capita income, and household size move together in a way that points to structural inequality across the city. Geography matters, but so does the burden of supporting larger households with lower income per person.

## Important caveat

The education breakdown is still informative, but the survey categories are narrow and somewhat administrative. They should be treated as directional evidence rather than as a full human-capital model.

## Portfolio takeaway

This repository now works better as a professional case because it has a clearer analytical question, exportable outputs, and a sharper conclusion. It reads less like a class notebook and more like an interpretable socioeconomic diagnostic.
