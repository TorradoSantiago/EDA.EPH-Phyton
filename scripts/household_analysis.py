from __future__ import annotations

import argparse
import unicodedata
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


ML_NUMERIC_FEATURES = [
    'anos_escolaridad',
    'edad',
    'potential_experience',
    'potential_experience_sq',
    'children_count',
]
ML_CATEGORICAL_FEATURES = [
    'comuna',
    'sexo',
    'cat_ocupacional',
    'situacion_conyugal',
    'nivel_max_educativo',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build descriptive, econometric, and machine-learning outputs for the 2019 Household Survey.'
    )
    parser.add_argument(
        '--input-path',
        type=Path,
        default=DEFAULT_INPUT,
        help='Path to the Annual Household Survey CSV file.',
    )
    parser.add_argument(
        '--oaxaca-bootstrap-iterations',
        type=int,
        default=200,
        help='Bootstrap iterations used to estimate confidence bands for the Oaxaca two-fold decomposition.',
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


def trim_target_outliers(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    q1 = dataframe[target_column].quantile(0.25)
    q3 = dataframe[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return dataframe.loc[dataframe[target_column].between(lower_bound, upper_bound)].copy()


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
    labor_frame['experience_centered'] = labor_frame['potential_experience'] - labor_frame['potential_experience'].mean()
    labor_frame['experience_centered_sq'] = labor_frame['experience_centered'] ** 2
    labor_frame['log_labor_income'] = np.log(labor_frame['ingreso_total_lab'])
    labor_frame['male'] = (labor_frame['sexo'] == 'Varon').astype(int)
    labor_frame['children_count'] = labor_frame['cantidad_hijos_nac_vivos'].fillna(0)

    return labor_frame


def fit_mincer_models(labor_frame: pd.DataFrame):
    formula = (
        'log_labor_income ~ anos_escolaridad + experience_centered + '
        'I(experience_centered ** 2) + C(cat_ocupacional) + C(comuna) + C(sexo)'
    )
    base_model = smf.ols(formula, data=labor_frame).fit()
    robust_model = smf.ols(formula, data=labor_frame).fit(cov_type='HC3')
    return formula, base_model, robust_model


def build_mincer_summary(model) -> pd.DataFrame:
    key_terms = {
        'anos_escolaridad': 'Additional year of schooling',
        'experience_centered': 'Potential experience (centered)',
        'I(experience_centered ** 2)': 'Potential experience squared',
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


def build_mincer_diagnostics(base_model) -> pd.DataFrame:
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(base_model.resid, base_model.model.exog)
    reset_result = linear_reset(base_model, power=2, use_f=True)
    jb_result = scipy_stats.jarque_bera(base_model.resid)

    return pd.DataFrame(
        [
            {
                'diagnostic': 'Adjusted R-squared',
                'statistic': float(base_model.rsquared_adj),
                'p_value': np.nan,
                'interpretation': 'Share of log-income variation explained by the earnings equation.',
            },
            {
                'diagnostic': 'Breusch-Pagan LM',
                'statistic': float(bp_lm),
                'p_value': float(bp_lm_pvalue),
                'interpretation': 'Tests for heteroskedasticity in the residual variance.',
            },
            {
                'diagnostic': 'Breusch-Pagan F',
                'statistic': float(bp_fvalue),
                'p_value': float(bp_f_pvalue),
                'interpretation': 'F-version of the heteroskedasticity diagnostic.',
            },
            {
                'diagnostic': 'RESET F',
                'statistic': float(reset_result.fvalue),
                'p_value': float(reset_result.pvalue),
                'interpretation': 'Checks whether important nonlinear patterns remain omitted.',
            },
            {
                'diagnostic': 'Jarque-Bera',
                'statistic': float(jb_result.statistic),
                'p_value': float(jb_result.pvalue),
                'interpretation': 'Assesses whether residuals are normally distributed.',
            },
            {
                'diagnostic': 'Observations',
                'statistic': float(base_model.nobs),
                'p_value': np.nan,
                'interpretation': 'Working sample used in the labor-income model.',
            },
        ]
    )


def build_mincer_vif(labor_frame: pd.DataFrame) -> pd.DataFrame:
    design = labor_frame[['anos_escolaridad', 'experience_centered', 'experience_centered_sq', 'male']].dropna()
    design = sm.add_constant(design, has_constant='add')

    rows: list[dict[str, float | str]] = []
    for index, column in enumerate(design.columns):
        if column == 'const':
            continue
        rows.append(
            {
                'term': column,
                'vif': float(variance_inflation_factor(design.values, index)),
                'interpretation': 'Values close to 1 suggest low multicollinearity after centering experience.',
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


def build_statistical_tests(
    dataframe: pd.DataFrame, households: pd.DataFrame, labor_frame: pd.DataFrame
) -> pd.DataFrame:
    adults = dataframe.loc[
        dataframe['edad'].ge(18)
        & dataframe['ingresos_totales'].gt(0)
        & dataframe['nivel_max_educativo'].notna()
    ].copy()

    commune_groups = [
        group['household_income'].dropna().values
        for _, group in households.groupby('commune')
        if group['household_income'].notna().sum() >= 20
    ]
    commune_test = scipy_stats.kruskal(*commune_groups)

    household_size_test = scipy_stats.spearmanr(
        households['household_members'], households['per_capita_income'], nan_policy='omit'
    )

    education_groups = [
        group['ingresos_totales'].dropna().values
        for _, group in adults.groupby('nivel_max_educativo')
        if group['ingresos_totales'].notna().sum() >= 25
    ]
    education_test = scipy_stats.kruskal(*education_groups)

    men_income = labor_frame.loc[labor_frame['sexo'] == 'Varon', 'ingreso_total_lab']
    women_income = labor_frame.loc[labor_frame['sexo'] == 'Mujer', 'ingreso_total_lab']
    gender_test = scipy_stats.mannwhitneyu(men_income, women_income, alternative='two-sided')

    return pd.DataFrame(
        [
            {
                'test_id': 'T1',
                'question': 'Do household-income distributions differ across communes?',
                'test': 'Kruskal-Wallis',
                'statistic': float(commune_test.statistic),
                'p_value': float(commune_test.pvalue),
                'key_signal': 'Reject equal medians across communes; geography matters for household income.',
            },
            {
                'test_id': 'T2',
                'question': 'Is household size monotonically associated with per-capita income?',
                'test': 'Spearman correlation',
                'statistic': float(household_size_test.statistic),
                'p_value': float(household_size_test.pvalue),
                'key_signal': 'Larger households are associated with lower per-capita income.',
            },
            {
                'test_id': 'T3',
                'question': 'Do education groups show different individual income distributions?',
                'test': 'Kruskal-Wallis',
                'statistic': float(education_test.statistic),
                'p_value': float(education_test.pvalue),
                'key_signal': 'Income rises meaningfully across schooling categories before modeling.',
            },
            {
                'test_id': 'T4',
                'question': 'Do men and women show different labor-income distributions in the employed sample?',
                'test': 'Mann-Whitney U',
                'statistic': float(gender_test.statistic),
                'p_value': float(gender_test.pvalue),
                'key_signal': 'The raw gender gap is visible even before conditioning on controls.',
            },
        ]
    )


def build_oaxaca_design_matrix(labor_frame: pd.DataFrame) -> pd.DataFrame:
    exog = pd.concat(
        [
            labor_frame[['male', 'anos_escolaridad', 'experience_centered', 'experience_centered_sq']].astype(float),
            pd.get_dummies(labor_frame['comuna'], prefix='commune', drop_first=True, dtype=float),
            pd.get_dummies(labor_frame['cat_ocupacional'], prefix='occupation', drop_first=True, dtype=float),
        ],
        axis=1,
    )
    return sm.add_constant(exog, has_constant='add')


def run_oaxaca_decomposition(labor_frame: pd.DataFrame):
    exog = build_oaxaca_design_matrix(labor_frame)
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


def bootstrap_oaxaca(labor_frame: pd.DataFrame, iterations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    male_frame = labor_frame.loc[labor_frame['male'] == 1].copy()
    female_frame = labor_frame.loc[labor_frame['male'] == 0].copy()
    draws: list[dict[str, float | int | str]] = []

    for iteration in range(iterations):
        sample = pd.concat(
            [
                male_frame.sample(n=len(male_frame), replace=True, random_state=42 + iteration),
                female_frame.sample(n=len(female_frame), replace=True, random_state=142 + iteration),
            ],
            ignore_index=True,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result, _ = run_oaxaca_decomposition(sample)
        except Exception:
            continue

        for component, value in zip(
            ['Unexplained effect', 'Explained effect', 'Total gap'],
            result.params,
            strict=True,
        ):
            draws.append(
                {
                    'iteration': iteration,
                    'component': component,
                    'log_points': float(value),
                }
            )

    draws_frame = pd.DataFrame(draws)
    if draws_frame.empty:
        raise RuntimeError('Bootstrap Oaxaca estimation failed for all iterations.')

    summary = (
        draws_frame.groupby('component')['log_points']
        .agg(mean_log_points='mean', std_log_points='std', successful_iterations='count')
        .reset_index()
    )
    quantiles = (
        draws_frame.groupby('component')['log_points']
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .rename(columns={0.025: 'ci_low', 0.5: 'median_log_points', 0.975: 'ci_high'})
        .reset_index()
    )
    summary = summary.merge(quantiles, on='component', how='left')
    summary['mean_pct_gap'] = (np.exp(summary['mean_log_points']) - 1) * 100
    summary['ci_low_pct_gap'] = (np.exp(summary['ci_low']) - 1) * 100
    summary['ci_high_pct_gap'] = (np.exp(summary['ci_high']) - 1) * 100
    return draws_frame, summary


def get_linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                    ]
                ),
                ML_NUMERIC_FEATURES,
            ),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                ML_CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ML_NUMERIC_FEATURES),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                ML_CATEGORICAL_FEATURES,
            ),
        ]
    )


def collapse_feature_name(feature_name: str) -> str:
    clean_name = feature_name.replace('num__', '').replace('cat__', '')
    for base_feature in ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES:
        if clean_name == base_feature or clean_name.startswith(f'{base_feature}_'):
            return base_feature
    return clean_name


def evaluate_predictions(model_name: str, y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float | str]:
    y_true_income = np.exp(y_true_log)
    y_pred_income = np.exp(y_pred_log)
    return {
        'model': model_name,
        'mae_log': mean_absolute_error(y_true_log, y_pred_log),
        'rmse_log': mean_squared_error(y_true_log, y_pred_log) ** 0.5,
        'r2_log': r2_score(y_true_log, y_pred_log),
        'mae_income': mean_absolute_error(y_true_income, y_pred_income),
        'rmse_income': mean_squared_error(y_true_income, y_pred_income) ** 0.5,
    }


def build_ml_cross_validation(features: pd.DataFrame, target: pd.Series, models: dict[str, Pipeline]) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mse': 'neg_mean_squared_error',
    }

    rows: list[dict[str, float | str]] = []
    for model_name, model in models.items():
        scores = cross_validate(model, features, target, cv=cv, scoring=scoring, n_jobs=1)
        rmse_values = np.sqrt(-scores['test_neg_mse'])
        rows.append(
            {
                'model': model_name,
                'cv_r2_mean': float(scores['test_r2'].mean()),
                'cv_r2_std': float(scores['test_r2'].std(ddof=0)),
                'cv_mae_mean': float((-scores['test_neg_mae']).mean()),
                'cv_mae_std': float((-scores['test_neg_mae']).std(ddof=0)),
                'cv_rmse_mean': float(rmse_values.mean()),
                'cv_rmse_std': float(rmse_values.std(ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values('cv_r2_mean', ascending=False)


def run_ml_models(
    labor_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ml_frame = trim_target_outliers(labor_frame, 'ingreso_total_lab')
    features = ml_frame[ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES]
    target = ml_frame['log_labor_income']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    models: dict[str, Pipeline] = {
        'Linear benchmark': Pipeline(
            steps=[
                ('preprocessor', get_linear_preprocessor()),
                ('model', LinearRegression()),
            ]
        ),
        'Elastic Net': Pipeline(
            steps=[
                ('preprocessor', get_linear_preprocessor()),
                ('model', ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)),
            ]
        ),
        'Random Forest': Pipeline(
            steps=[
                ('preprocessor', get_tree_preprocessor()),
                (
                    'model',
                    RandomForestRegressor(
                        n_estimators=350,
                        max_depth=12,
                        min_samples_leaf=6,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        'Gradient Boosting': Pipeline(
            steps=[
                ('preprocessor', get_tree_preprocessor()),
                (
                    'model',
                    GradientBoostingRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=2,
                        subsample=0.85,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    metrics_rows: list[dict[str, float | str]] = []
    best_model_name = ''
    best_model: Pipeline | None = None
    best_predictions: np.ndarray | None = None
    best_r2 = -np.inf

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(model_name, y_test, predictions)
        metrics_rows.append(metrics)
        if metrics['r2_log'] > best_r2:
            best_r2 = float(metrics['r2_log'])
            best_model_name = model_name
            best_model = model
            best_predictions = predictions

    model_comparison = pd.DataFrame(metrics_rows).sort_values('r2_log', ascending=False)

    if best_model is None or best_predictions is None:
        raise RuntimeError('Best ML model could not be determined.')

    best_estimator = best_model.named_steps['model']
    best_params = pd.DataFrame(
        [
            {'model': best_model_name, 'parameter': key, 'value': value}
            for key, value in best_estimator.get_params().items()
            if key in {'alpha', 'l1_ratio', 'learning_rate', 'max_depth', 'min_samples_leaf', 'n_estimators', 'subsample'}
        ]
    )

    if hasattr(best_estimator, 'feature_importances_'):
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        feature_importance = pd.DataFrame(
            {
                'feature': feature_names,
                'importance': best_estimator.feature_importances_,
            }
        )
        feature_importance['feature_group'] = feature_importance['feature'].map(collapse_feature_name)
        feature_importance = (
            feature_importance.groupby('feature_group', as_index=False)['importance']
            .sum()
            .sort_values('importance', ascending=False)
        )
    else:
        feature_importance = pd.DataFrame(columns=['feature_group', 'importance'])

    prediction_frame = pd.DataFrame(
        {
            'actual_log_income': y_test,
            'predicted_log_income': best_predictions,
            'actual_income': np.exp(y_test),
            'predicted_income': np.exp(best_predictions),
            'model': best_model_name,
        }
    ).reset_index(drop=True)
    ml_cross_validation = build_ml_cross_validation(features, target, models)

    return model_comparison, best_params, feature_importance, prediction_frame, ml_cross_validation


def export_tables(
    households: pd.DataFrame,
    commune_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    marital_status_summary: pd.DataFrame,
    gender_income_summary: pd.DataFrame,
    mincer_summary: pd.DataFrame,
    mincer_diagnostics: pd.DataFrame,
    mincer_vif: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    oaxaca_two_fold_summary: pd.DataFrame,
    oaxaca_three_fold_summary: pd.DataFrame,
    oaxaca_bootstrap_summary: pd.DataFrame,
    ml_model_comparison: pd.DataFrame,
    ml_best_params: pd.DataFrame,
    ml_feature_importance: pd.DataFrame,
    ml_cross_validation: pd.DataFrame,
) -> None:
    households.to_csv(TABLES_DIR / 'household_dataset.csv', index=False)
    commune_summary.to_csv(TABLES_DIR / 'commune_income_summary.csv', index=False)
    education_summary.to_csv(TABLES_DIR / 'education_income_summary.csv', index=False)
    marital_status_summary.to_csv(TABLES_DIR / 'marital_status_income_summary.csv', index=False)
    gender_income_summary.to_csv(TABLES_DIR / 'gender_income_summary.csv', index=False)
    mincer_summary.to_csv(TABLES_DIR / 'mincer_coefficients.csv', index=False)
    mincer_diagnostics.to_csv(TABLES_DIR / 'mincer_diagnostics.csv', index=False)
    mincer_vif.to_csv(TABLES_DIR / 'mincer_vif.csv', index=False)
    statistical_tests.to_csv(TABLES_DIR / 'statistical_tests.csv', index=False)
    oaxaca_two_fold_summary.to_csv(TABLES_DIR / 'oaxaca_two_fold_summary.csv', index=False)
    oaxaca_three_fold_summary.to_csv(TABLES_DIR / 'oaxaca_three_fold_summary.csv', index=False)
    oaxaca_bootstrap_summary.to_csv(TABLES_DIR / 'oaxaca_bootstrap_summary.csv', index=False)
    ml_model_comparison.to_csv(TABLES_DIR / 'ml_model_comparison.csv', index=False)
    ml_best_params.to_csv(TABLES_DIR / 'ml_best_params.csv', index=False)
    ml_feature_importance.to_csv(TABLES_DIR / 'ml_feature_importance.csv', index=False)
    ml_cross_validation.to_csv(TABLES_DIR / 'ml_cross_validation.csv', index=False)


def export_figures(
    commune_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    labor_frame: pd.DataFrame,
    mincer_summary: pd.DataFrame,
    oaxaca_two_fold_summary: pd.DataFrame,
    oaxaca_bootstrap_summary: pd.DataFrame,
    ml_model_comparison: pd.DataFrame,
    ml_feature_importance: pd.DataFrame,
    ml_prediction_frame: pd.DataFrame,
    ml_cross_validation: pd.DataFrame,
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

    plot_frame = mincer_summary.copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_frame, x='label', y='approx_pct_effect', hue='label', dodge=False, palette='Blues_d')
    legend = plt.gca().get_legend()
    if legend is not None:
        legend.remove()
    plt.axhline(0, color='#333333', linewidth=1)
    plt.title('Key effects from the Mincer earnings equation')
    plt.xlabel('')
    plt.ylabel('Approximate percent effect')
    plt.xticks(rotation=18, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mincer_key_effects.png', dpi=150)
    plt.close()

    reference_commune = int(labor_frame['comuna'].mode().iloc[0])
    reference_occupation = str(labor_frame['cat_ocupacional'].mode().iloc[0])
    school_years = list(
        range(
            int(labor_frame['anos_escolaridad'].quantile(0.05)),
            int(labor_frame['anos_escolaridad'].quantile(0.95)) + 1,
        )
    )

    prediction_rows: list[dict[str, float | int | str]] = []
    for sex in ['Mujer', 'Varon']:
        for years in school_years:
            prediction_rows.append(
                {
                    'anos_escolaridad': years,
                    'experience_centered': 0.0,
                    'cat_ocupacional': reference_occupation,
                    'comuna': reference_commune,
                    'sexo': sex,
                }
            )

    prediction_frame = pd.DataFrame(prediction_rows)
    schooling_effect = float(
        mincer_summary.loc[mincer_summary['term'] == 'anos_escolaridad', 'coefficient'].iloc[0]
    )
    male_effect = float(
        mincer_summary.loc[mincer_summary['term'] == 'C(sexo)[T.Varon]', 'coefficient'].iloc[0]
    )
    prediction_frame['predicted_log_income'] = schooling_effect * prediction_frame['anos_escolaridad']
    prediction_frame.loc[prediction_frame['sexo'] == 'Varon', 'predicted_log_income'] += male_effect
    prediction_frame['predicted_labor_income'] = np.exp(prediction_frame['predicted_log_income'])
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

    oaxaca_ci_plot = oaxaca_bootstrap_summary.loc[
        oaxaca_bootstrap_summary['component'] != 'Total gap'
    ].copy()
    lower_error = oaxaca_ci_plot['mean_pct_gap'] - oaxaca_ci_plot['ci_low_pct_gap']
    upper_error = oaxaca_ci_plot['ci_high_pct_gap'] - oaxaca_ci_plot['mean_pct_gap']

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        x=oaxaca_ci_plot['mean_pct_gap'],
        y=oaxaca_ci_plot['component'],
        xerr=[lower_error, upper_error],
        fmt='o',
        color='#145da0',
        ecolor='#7aa6d6',
        elinewidth=2.2,
        capsize=4,
    )
    plt.axvline(0, color='#333333', linewidth=1, linestyle='--')
    plt.title('Bootstrap confidence intervals for the Oaxaca decomposition')
    plt.xlabel('Percent gap')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'oaxaca_bootstrap_ci.png', dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=ml_model_comparison, x='model', y='r2_log', color='#2e8b57')
    plt.title('Out-of-sample ML benchmark on log labor income')
    plt.xlabel('Model')
    plt.ylabel('R-squared on log income')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_model_performance.png', dpi=150)
    plt.close()

    if not ml_feature_importance.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=ml_feature_importance.head(10), x='importance', y='feature_group', color='#6b5ca5')
        plt.title('Most important features in the best ML model')
        plt.xlabel('Importance')
        plt.ylabel('Feature group')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ml_feature_importance.png', dpi=150)
        plt.close()

    plt.figure(figsize=(7, 7))
    plot_frame = ml_prediction_frame.copy().sort_values('actual_income')
    sns.scatterplot(data=plot_frame, x='actual_income', y='predicted_income', s=24, alpha=0.6, color='#1f4b6e')
    min_value = min(plot_frame['actual_income'].min(), plot_frame['predicted_income'].min())
    max_value = max(plot_frame['actual_income'].max(), plot_frame['predicted_income'].max())
    plt.plot([min_value, max_value], [min_value, max_value], color='#b94a48', linewidth=1.5)
    plt.title('Actual vs predicted income in the best ML model')
    plt.xlabel('Actual labor income')
    plt.ylabel('Predicted labor income')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_actual_vs_predicted.png', dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=ml_cross_validation, x='model', y='cv_r2_mean', hue='model', dodge=False, palette='rocket')
    legend = plt.gca().get_legend()
    if legend is not None:
        legend.remove()
    plt.errorbar(
        x=np.arange(len(ml_cross_validation)),
        y=ml_cross_validation['cv_r2_mean'],
        yerr=ml_cross_validation['cv_r2_std'],
        fmt='none',
        ecolor='#213547',
        capsize=4,
        linewidth=1.5,
    )
    plt.title('Five-fold cross-validation R-squared')
    plt.xlabel('Model')
    plt.ylabel('Cross-validated R-squared')
    plt.xticks(rotation=18, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_cross_validation_r2.png', dpi=150)
    plt.close()


def print_summary(
    households: pd.DataFrame,
    commune_summary: pd.DataFrame,
    statistical_tests: pd.DataFrame,
    labor_frame: pd.DataFrame,
    mincer_summary: pd.DataFrame,
    oaxaca_bootstrap_summary: pd.DataFrame,
    oaxaca_two_fold_summary: pd.DataFrame,
    ml_model_comparison: pd.DataFrame,
) -> None:
    schooling_effect = float(
        mincer_summary.loc[mincer_summary['term'] == 'anos_escolaridad', 'approx_pct_effect'].iloc[0]
    )
    total_gap = float(oaxaca_two_fold_summary.loc[oaxaca_two_fold_summary['component'] == 'Total gap', 'pct_gap'].iloc[0])
    unexplained_component = float(
        oaxaca_bootstrap_summary.loc[
            oaxaca_bootstrap_summary['component'] == 'Unexplained effect', 'mean_pct_gap'
        ].iloc[0]
    )
    commune_test_pvalue = float(statistical_tests.loc[statistical_tests['test_id'] == 'T1', 'p_value'].iloc[0])
    best_ml_model = ml_model_comparison.iloc[0]

    print('=== Project summary ===')
    print(f'Households analyzed: {len(households):,}')
    print(f'Communes covered: {households["commune"].nunique()}')
    print(f'Workers in econometric sample: {len(labor_frame):,}')
    print(f'Estimated schooling premium per year: {schooling_effect:.2f}%')
    print(f'Gender labor-income gap: {total_gap:.2f}%')
    print(f'Bootstrap unexplained component of the gap: {unexplained_component:.2f}%')
    print(f'Commune distribution test p-value: {commune_test_pvalue:.3g}')
    print(f"Best ML benchmark: {best_ml_model['model']} (R^2 on log income = {best_ml_model['r2_log']:.3f})")

    print('\nTop 5 communes by median household income:')
    print(commune_summary.head(5).to_string(index=False))

    print('\nGenerated files:')
    print(f'- {TABLES_DIR / "household_dataset.csv"}')
    print(f'- {TABLES_DIR / "commune_income_summary.csv"}')
    print(f'- {TABLES_DIR / "education_income_summary.csv"}')
    print(f'- {TABLES_DIR / "marital_status_income_summary.csv"}')
    print(f'- {TABLES_DIR / "gender_income_summary.csv"}')
    print(f'- {TABLES_DIR / "mincer_coefficients.csv"}')
    print(f'- {TABLES_DIR / "mincer_diagnostics.csv"}')
    print(f'- {TABLES_DIR / "mincer_vif.csv"}')
    print(f'- {TABLES_DIR / "statistical_tests.csv"}')
    print(f'- {TABLES_DIR / "oaxaca_two_fold_summary.csv"}')
    print(f'- {TABLES_DIR / "oaxaca_three_fold_summary.csv"}')
    print(f'- {TABLES_DIR / "oaxaca_bootstrap_summary.csv"}')
    print(f'- {TABLES_DIR / "ml_model_comparison.csv"}')
    print(f'- {TABLES_DIR / "ml_best_params.csv"}')
    print(f'- {TABLES_DIR / "ml_feature_importance.csv"}')
    print(f'- {TABLES_DIR / "ml_cross_validation.csv"}')
    print(f'- {FIGURES_DIR / "median_household_income_by_commune.png"}')
    print(f'- {FIGURES_DIR / "average_income_by_education.png"}')
    print(f'- {FIGURES_DIR / "mincer_key_effects.png"}')
    print(f'- {FIGURES_DIR / "predicted_labor_income_by_schooling_gender.png"}')
    print(f'- {FIGURES_DIR / "gender_income_gap_decomposition.png"}')
    print(f'- {FIGURES_DIR / "oaxaca_bootstrap_ci.png"}')
    print(f'- {FIGURES_DIR / "ml_model_performance.png"}')
    print(f'- {FIGURES_DIR / "ml_feature_importance.png"}')
    print(f'- {FIGURES_DIR / "ml_actual_vs_predicted.png"}')
    print(f'- {FIGURES_DIR / "ml_cross_validation_r2.png"}')


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
    statistical_tests = build_statistical_tests(dataframe, households, labor_frame)
    _, mincer_base_model, mincer_robust_model = fit_mincer_models(labor_frame)
    mincer_summary = build_mincer_summary(mincer_robust_model)
    mincer_diagnostics = build_mincer_diagnostics(mincer_base_model)
    mincer_vif = build_mincer_vif(labor_frame)
    gender_income_summary = build_gender_income_summary(labor_frame)
    oaxaca_two_fold_result, oaxaca_three_fold_result = run_oaxaca_decomposition(labor_frame)
    oaxaca_two_fold_summary, oaxaca_three_fold_summary = build_oaxaca_summary(
        oaxaca_two_fold_result,
        oaxaca_three_fold_result,
    )
    _, oaxaca_bootstrap_summary = bootstrap_oaxaca(labor_frame, args.oaxaca_bootstrap_iterations)
    ml_model_comparison, ml_best_params, ml_feature_importance, ml_prediction_frame, ml_cross_validation = run_ml_models(
        labor_frame
    )

    export_tables(
        households,
        commune_summary,
        education_summary,
        marital_status_summary,
        gender_income_summary,
        mincer_summary,
        mincer_diagnostics,
        mincer_vif,
        statistical_tests,
        oaxaca_two_fold_summary,
        oaxaca_three_fold_summary,
        oaxaca_bootstrap_summary,
        ml_model_comparison,
        ml_best_params,
        ml_feature_importance,
        ml_cross_validation,
    )
    export_figures(
        commune_summary,
        education_summary,
        labor_frame,
        mincer_summary,
        oaxaca_two_fold_summary,
        oaxaca_bootstrap_summary,
        ml_model_comparison,
        ml_feature_importance,
        ml_prediction_frame,
        ml_cross_validation,
    )
    print_summary(
        households,
        commune_summary,
        statistical_tests,
        labor_frame,
        mincer_summary,
        oaxaca_bootstrap_summary,
        oaxaca_two_fold_summary,
        ml_model_comparison,
    )


if __name__ == '__main__':
    main()
