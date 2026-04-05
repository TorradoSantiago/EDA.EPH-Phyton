from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.oaxaca import OaxacaBlinder

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / 'annual_household_survey_2019.csv'
TABLES_DIR = BASE_DIR / 'outputs' / 'tables'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures'

EDUCATION_LABELS = {
    'egb (1o a 9o ano)': 'Basic general education (years 1-9)',
    'no corresponde': 'Not applicable',
    'otras escuelas especiales': 'Other special schools',
    'primario comun': 'General primary',
    'primario especial': 'Special primary',
    'sala de 5': 'Kindergarten age 5',
    'secundario/medio comun': 'General secondary',
}

MARITAL_STATUS_LABELS = {
    'casado/a': 'Married',
    'divorciado/a': 'Divorced',
    'no corresponde': 'Not applicable',
    'separado/a de union o matrimonio': 'Separated',
    'soltero/a': 'Single',
    'unido/a': 'Cohabiting',
    'viudo/a': 'Widowed',
}

EMPLOYMENT_STATUS_LABELS = {
    'desocupado': 'Unemployed',
    'inactivo': 'Inactive',
    'ocupado': 'Employed',
}

SEX_LABELS = {
    'mujer': 'Women',
    'varon': 'Men',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a household-level dataset and econometric summaries for the 2019 Household Survey.'
    )
    parser.add_argument(
        '--input-path',
        type=Path,
        default=DEFAULT_INPUT,
        help='Path to the Annual Household Survey CSV file.',
    )
    return parser.parse_args()


def normalize_column_name(name: str) -> str:
    text = unicodedata.normalize('NFKD', name)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.lower().replace(' ', '_')


def normalize_text_token(value: object) -> str:
    text = unicodedata.normalize('NFKD', str(value))
    text = text.encode('ascii', 'ignore').decode('ascii')
    return ' '.join(text.lower().strip().split())


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f'Input file was not found: {input_path}')

    dataframe = pd.read_csv(input_path, encoding='latin-1')
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
    return dataframe


def cast_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        'id',
        'nhogar',
        'miembro',
        'comuna',
        'edad',
        'ingreso_total_lab',
        'ingreso_total_no_lab',
        'ingresos_totales',
        'ingresos_familiares',
        'ingreso_per_capita_familiar',
        'anos_escolaridad',
        'cantidad_hijos_nac_vivos',
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')

    return dataframe


def series_mode(series: pd.Series) -> str | float:
    clean_series = series.dropna()
    if clean_series.empty:
        return float('nan')
    return str(clean_series.mode().iloc[0])


def translate_label(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return 'Unspecified'
    return mapping.get(normalize_text_token(value), str(value))


def build_household_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe['household_id'] = (
        dataframe['id'].astype('Int64').astype(str) + '-' + dataframe['nhogar'].astype('Int64').astype(str)
    )

    aggregations = {
        'comuna': 'first',
        'dominio': 'first',
        'miembro': 'count',
        'edad': 'mean',
        'ingresos_familiares': 'max',
        'ingreso_per_capita_familiar': 'max',
        'ingreso_total_lab': 'sum',
        'ingreso_total_no_lab': 'sum',
        'anos_escolaridad': 'mean',
        'situacion_conyugal': series_mode,
        'nivel_max_educativo': series_mode,
        'estado_ocupacional': series_mode,
    }

    households = (
        dataframe.groupby('household_id', dropna=False)
        .agg(aggregations)
        .rename(
            columns={
                'comuna': 'commune',
                'miembro': 'household_members',
                'edad': 'average_age',
                'ingresos_familiares': 'household_income',
                'ingreso_per_capita_familiar': 'per_capita_income',
                'ingreso_total_lab': 'total_labor_income',
                'ingreso_total_no_lab': 'total_non_labor_income',
                'anos_escolaridad': 'average_school_years',
                'situacion_conyugal': 'dominant_marital_status',
                'nivel_max_educativo': 'dominant_education_level',
                'estado_ocupacional': 'dominant_employment_status',
            }
        )
        .reset_index()
    )

    households['dominant_marital_status'] = households['dominant_marital_status'].map(
        lambda value: translate_label(value, MARITAL_STATUS_LABELS)
    )
    households['dominant_education_level'] = households['dominant_education_level'].map(
        lambda value: translate_label(value, EDUCATION_LABELS)
    )
    households['dominant_employment_status'] = households['dominant_employment_status'].map(
        lambda value: translate_label(value, EMPLOYMENT_STATUS_LABELS)
    )

    return households


def build_commune_summary(households: pd.DataFrame) -> pd.DataFrame:
    return (
        households.groupby('commune', dropna=False)
        .agg(
            households=('household_id', 'count'),
            median_household_income=('household_income', 'median'),
            median_per_capita_income=('per_capita_income', 'median'),
            average_members=('household_members', 'mean'),
        )
        .reset_index()
        .sort_values('median_household_income', ascending=False)
    )


def build_education_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    adults = dataframe.loc[dataframe['edad'] >= 18].copy()
    summary = (
        adults.groupby('nivel_max_educativo', dropna=False)
        .agg(
            people=('id', 'count'),
            median_total_income=('ingresos_totales', 'median'),
            average_total_income=('ingresos_totales', 'mean'),
        )
        .reset_index()
        .rename(columns={'nivel_max_educativo': 'education_level'})
        .sort_values('average_total_income', ascending=False)
    )
    summary['education_level'] = summary['education_level'].map(
        lambda value: translate_label(value, EDUCATION_LABELS)
    )
    return summary


def build_marital_status_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    adults = dataframe.loc[dataframe['edad'] >= 18].copy()
    summary = (
        adults.groupby('situacion_conyugal', dropna=False)
        .agg(
            people=('id', 'count'),
            median_per_capita_income=('ingreso_per_capita_familiar', 'median'),
        )
        .reset_index()
        .rename(columns={'situacion_conyugal': 'marital_status'})
        .sort_values('median_per_capita_income', ascending=False)
    )
    summary['marital_status'] = summary['marital_status'].map(
        lambda value: translate_label(value, MARITAL_STATUS_LABELS)
    )
    return summary


def build_individual_labor_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    labor_frame = dataframe.loc[
        dataframe['edad'].between(18, 65)
        & (dataframe['estado_ocupacional'] == 'Ocupado')
        & dataframe['sexo'].isin(['Mujer', 'Varon'])
        & dataframe['anos_escolaridad'].notna()
        & dataframe['ingreso_total_lab'].gt(0)
    ].copy()

    labor_frame['potential_experience'] = (labor_frame['edad'] - labor_frame['anos_escolaridad'] - 6).clip(lower=0)
    labor_frame['potential_experience_sq'] = labor_frame['potential_experience'] ** 2
    labor_frame['log_labor_income'] = np.log(labor_frame['ingreso_total_lab'])
    labor_frame['male'] = (labor_frame['sexo'] == 'Varon').astype(int)

    return labor_frame


def fit_mincer_model(labor_frame: pd.DataFrame):
    formula = (
        'log_labor_income ~ anos_escolaridad + potential_experience + '
        'I(potential_experience ** 2) + C(cat_ocupacional) + C(comuna) + C(sexo)'
    )
    return smf.ols(formula, data=labor_frame).fit(cov_type='HC3')


def build_mincer_summary(model) -> pd.DataFrame:
    key_terms = {
        'anos_escolaridad': 'Additional year of schooling',
        'potential_experience': 'Potential experience',
        'I(potential_experience ** 2)': 'Potential experience squared',
        'C(sexo)[T.Varon]': 'Male income premium',
    }

    rows: list[dict[str, float | str]] = []
    for term, label in key_terms.items():
        coefficient = float(model.params[term])
        rows.append(
            {
                'term': term,
                'label': label,
                'coefficient': coefficient,
                'std_error': float(model.bse[term]),
                'p_value': float(model.pvalues[term]),
                'approx_pct_effect': (np.exp(coefficient) - 1) * 100,
            }
        )

    return pd.DataFrame(rows)


def build_gender_income_summary(labor_frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        labor_frame.groupby('sexo', dropna=False)
        .agg(
            workers=('id', 'count'),
            average_labor_income=('ingreso_total_lab', 'mean'),
            median_labor_income=('ingreso_total_lab', 'median'),
            average_school_years=('anos_escolaridad', 'mean'),
            average_potential_experience=('potential_experience', 'mean'),
        )
        .reset_index()
        .rename(columns={'sexo': 'group'})
    )
    summary['group'] = summary['group'].map(lambda value: translate_label(value, SEX_LABELS))
    return summary


def run_oaxaca_decomposition(labor_frame: pd.DataFrame):
    exog = pd.concat(
        [
            labor_frame[['male', 'anos_escolaridad', 'potential_experience', 'potential_experience_sq']].astype(float),
            pd.get_dummies(labor_frame['comuna'], prefix='commune', drop_first=True, dtype=float),
            pd.get_dummies(labor_frame['cat_ocupacional'], prefix='occupation', drop_first=True, dtype=float),
        ],
        axis=1,
    )
    exog = sm.add_constant(exog, has_constant='add')

    model = OaxacaBlinder(labor_frame['log_labor_income'], exog, 'male', hasconst=True)
    return model.two_fold(), model.three_fold()


def build_oaxaca_summary(two_fold_result, three_fold_result) -> tuple[pd.DataFrame, pd.DataFrame]:
    two_fold = pd.DataFrame(
        [
            {
                'component': 'Unexplained effect',
                'log_points': float(two_fold_result.params[0]),
                'pct_gap': (np.exp(float(two_fold_result.params[0])) - 1) * 100,
            },
            {
                'component': 'Explained effect',
                'log_points': float(two_fold_result.params[1]),
                'pct_gap': (np.exp(float(two_fold_result.params[1])) - 1) * 100,
            },
            {
                'component': 'Total gap',
                'log_points': float(two_fold_result.params[2]),
                'pct_gap': (np.exp(float(two_fold_result.params[2])) - 1) * 100,
            },
        ]
    )

    three_fold = pd.DataFrame(
        [
            {
                'component': 'Endowment effect',
                'log_points': float(three_fold_result.params[0]),
                'pct_gap': (np.exp(float(three_fold_result.params[0])) - 1) * 100,
            },
            {
                'component': 'Coefficient effect',
                'log_points': float(three_fold_result.params[1]),
                'pct_gap': (np.exp(float(three_fold_result.params[1])) - 1) * 100,
            },
            {
                'component': 'Interaction effect',
                'log_points': float(three_fold_result.params[2]),
                'pct_gap': (np.exp(float(three_fold_result.params[2])) - 1) * 100,
            },
            {
                'component': 'Total gap',
                'log_points': float(three_fold_result.params[3]),
                'pct_gap': (np.exp(float(three_fold_result.params[3])) - 1) * 100,
            },
        ]
    )

    return two_fold, three_fold


def export_tables(
    households: pd.DataFrame,
    commune_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    marital_status_summary: pd.DataFrame,
    gender_income_summary: pd.DataFrame,
    mincer_summary: pd.DataFrame,
    oaxaca_two_fold_summary: pd.DataFrame,
    oaxaca_three_fold_summary: pd.DataFrame,
) -> None:
    households.to_csv(TABLES_DIR / 'household_dataset.csv', index=False)
    commune_summary.to_csv(TABLES_DIR / 'commune_income_summary.csv', index=False)
    education_summary.to_csv(TABLES_DIR / 'education_income_summary.csv', index=False)
    marital_status_summary.to_csv(TABLES_DIR / 'marital_status_income_summary.csv', index=False)
    gender_income_summary.to_csv(TABLES_DIR / 'gender_income_summary.csv', index=False)
    mincer_summary.to_csv(TABLES_DIR / 'mincer_coefficients.csv', index=False)
    oaxaca_two_fold_summary.to_csv(TABLES_DIR / 'oaxaca_two_fold_summary.csv', index=False)
    oaxaca_three_fold_summary.to_csv(TABLES_DIR / 'oaxaca_three_fold_summary.csv', index=False)


def export_figures(
    commune_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    labor_frame: pd.DataFrame,
    mincer_model,
    oaxaca_two_fold_summary: pd.DataFrame,
) -> None:
    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=commune_summary,
        x='commune',
        y='median_household_income',
        color='#1f4b6e',
    )
    plt.title('Median household income by commune')
    plt.xlabel('Commune')
    plt.ylabel('Median household income')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'median_household_income_by_commune.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=education_summary,
        x='education_level',
        y='average_total_income',
        color='#b36a32',
    )
    plt.title('Average total income by education level')
    plt.xlabel('Education level')
    plt.ylabel('Average total income')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'average_income_by_education.png', dpi=150)
    plt.close()

    reference_commune = int(labor_frame['comuna'].mode().iloc[0])
    reference_occupation = str(labor_frame['cat_ocupacional'].mode().iloc[0])
    reference_experience = float(labor_frame['potential_experience'].median())
    school_years = list(range(int(labor_frame['anos_escolaridad'].quantile(0.05)), int(labor_frame['anos_escolaridad'].quantile(0.95)) + 1))

    prediction_rows: list[dict[str, float | int | str]] = []
    for sex in ['Mujer', 'Varon']:
        for years in school_years:
            prediction_rows.append(
                {
                    'anos_escolaridad': years,
                    'potential_experience': reference_experience,
                    'cat_ocupacional': reference_occupation,
                    'comuna': reference_commune,
                    'sexo': sex,
                }
            )

    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame['predicted_labor_income'] = np.exp(mincer_model.predict(prediction_frame))
    prediction_frame['group'] = prediction_frame['sexo'].map(lambda value: translate_label(value, SEX_LABELS))

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=prediction_frame,
        x='anos_escolaridad',
        y='predicted_labor_income',
        hue='group',
        linewidth=2.4,
        palette=['#b94a48', '#1f4b6e'],
    )
    plt.title('Predicted labor income by schooling and gender')
    plt.xlabel('Years of schooling')
    plt.ylabel('Predicted labor income')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'predicted_labor_income_by_schooling_gender.png', dpi=150)
    plt.close()

    decomposition_plot = oaxaca_two_fold_summary.copy()
    decomposition_plot = decomposition_plot.loc[decomposition_plot['component'] != 'Total gap'].copy()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=decomposition_plot,
        x='component',
        y='pct_gap',
        hue='component',
        dodge=False,
        palette=['#c9772b', '#6b5ca5'],
    )
    plt.legend([], [], frameon=False)
    plt.axhline(0, color='#333333', linewidth=1)
    plt.title('Gender labor-income gap decomposition')
    plt.xlabel('Component')
    plt.ylabel('Percent gap')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'gender_income_gap_decomposition.png', dpi=150)
    plt.close()


def print_summary(
    households: pd.DataFrame,
    commune_summary: pd.DataFrame,
    labor_frame: pd.DataFrame,
    mincer_summary: pd.DataFrame,
    oaxaca_two_fold_summary: pd.DataFrame,
) -> None:
    schooling_effect = float(
        mincer_summary.loc[mincer_summary['term'] == 'anos_escolaridad', 'approx_pct_effect'].iloc[0]
    )
    total_gap = float(oaxaca_two_fold_summary.loc[oaxaca_two_fold_summary['component'] == 'Total gap', 'pct_gap'].iloc[0])
    unexplained_component = float(
        oaxaca_two_fold_summary.loc[oaxaca_two_fold_summary['component'] == 'Unexplained effect', 'pct_gap'].iloc[0]
    )

    print('=== Project summary ===')
    print(f'Households analyzed: {len(households):,}')
    print(f'Communes covered: {households["commune"].nunique()}')
    print(f'Workers in econometric sample: {len(labor_frame):,}')
    print(f'Estimated schooling premium per year: {schooling_effect:.2f}%')
    print(f'Gender labor-income gap: {total_gap:.2f}%')
    print(f'Unexplained component of the gap: {unexplained_component:.2f}%')

    print('\nTop 5 communes by median household income:')
    print(commune_summary.head(5).to_string(index=False))

    print('\nGenerated files:')
    print(f'- {TABLES_DIR / "household_dataset.csv"}')
    print(f'- {TABLES_DIR / "commune_income_summary.csv"}')
    print(f'- {TABLES_DIR / "education_income_summary.csv"}')
    print(f'- {TABLES_DIR / "marital_status_income_summary.csv"}')
    print(f'- {TABLES_DIR / "gender_income_summary.csv"}')
    print(f'- {TABLES_DIR / "mincer_coefficients.csv"}')
    print(f'- {TABLES_DIR / "oaxaca_two_fold_summary.csv"}')
    print(f'- {TABLES_DIR / "oaxaca_three_fold_summary.csv"}')
    print(f'- {FIGURES_DIR / "median_household_income_by_commune.png"}')
    print(f'- {FIGURES_DIR / "average_income_by_education.png"}')
    print(f'- {FIGURES_DIR / "predicted_labor_income_by_schooling_gender.png"}')
    print(f'- {FIGURES_DIR / "gender_income_gap_decomposition.png"}')


def main() -> None:
    args = parse_args()
    ensure_directories()
    dataframe = load_dataset(args.input_path)
    dataframe = cast_numeric_columns(dataframe)

    households = build_household_dataset(dataframe)
    commune_summary = build_commune_summary(households)
    education_summary = build_education_summary(dataframe)
    marital_status_summary = build_marital_status_summary(dataframe)

    labor_frame = build_individual_labor_frame(dataframe)
    mincer_model = fit_mincer_model(labor_frame)
    mincer_summary = build_mincer_summary(mincer_model)
    gender_income_summary = build_gender_income_summary(labor_frame)
    oaxaca_two_fold_result, oaxaca_three_fold_result = run_oaxaca_decomposition(labor_frame)
    oaxaca_two_fold_summary, oaxaca_three_fold_summary = build_oaxaca_summary(
        oaxaca_two_fold_result,
        oaxaca_three_fold_result,
    )

    export_tables(
        households,
        commune_summary,
        education_summary,
        marital_status_summary,
        gender_income_summary,
        mincer_summary,
        oaxaca_two_fold_summary,
        oaxaca_three_fold_summary,
    )
    export_figures(
        commune_summary,
        education_summary,
        labor_frame,
        mincer_model,
        oaxaca_two_fold_summary,
    )
    print_summary(
        households,
        commune_summary,
        labor_frame,
        mincer_summary,
        oaxaca_two_fold_summary,
    )


if __name__ == '__main__':
    main()
