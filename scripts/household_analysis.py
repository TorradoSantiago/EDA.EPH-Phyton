from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "annual_household_survey_2019.csv"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

EDUCATION_LABELS = {
    "EGB (1° a 9° año)": "Basic general education (years 1-9)",
    "No corresponde": "Not applicable",
    "Otras escuelas especiales": "Other special schools",
    "Primario comun": "General primary",
    "Primario especial": "Special primary",
    "Sala de 5": "Kindergarten age 5",
    "Secundario/medio comun": "General secondary",
}

MARITAL_STATUS_LABELS = {
    "Casado/a": "Married",
    "Divorciado/a": "Divorced",
    "No corresponde": "Not applicable",
    "Separado/a de unión o matrimonio": "Separated",
    "Soltero/a": "Single",
    "Unido/a": "Cohabiting",
    "Viudo/a": "Widowed",
}

EMPLOYMENT_STATUS_LABELS = {
    "Desocupado": "Unemployed",
    "Inactivo": "Inactive",
    "Ocupado": "Employed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a household-level dataset and executive summaries for the 2019 Household Survey."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the Annual Household Survey CSV file.",
    )
    return parser.parse_args()


def normalize_column_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower().replace(" ", "_")


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    dataframe = pd.read_csv(input_path, encoding="latin-1")
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
    return dataframe


def cast_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "id",
        "nhogar",
        "miembro",
        "comuna",
        "edad",
        "ingreso_total_lab",
        "ingreso_total_no_lab",
        "ingresos_totales",
        "ingresos_familiares",
        "ingreso_per_capita_familiar",
        "anos_escolaridad",
        "cantidad_hijos_nac_vivos",
    ]

    for column in numeric_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    return dataframe


def series_mode(series: pd.Series) -> str | float:
    clean_series = series.dropna()
    if clean_series.empty:
        return float("nan")
    return str(clean_series.mode().iloc[0])


def translate_label(value: object, mapping: dict[str, str]) -> object:
    if pd.isna(value):
        return "Unspecified"
    return mapping.get(str(value), str(value))


def build_household_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["household_id"] = (
        dataframe["id"].astype("Int64").astype(str) + "-" + dataframe["nhogar"].astype("Int64").astype(str)
    )

    aggregations = {
        "comuna": "first",
        "dominio": "first",
        "miembro": "count",
        "edad": "mean",
        "ingresos_familiares": "max",
        "ingreso_per_capita_familiar": "max",
        "ingreso_total_lab": "sum",
        "ingreso_total_no_lab": "sum",
        "anos_escolaridad": "mean",
        "situacion_conyugal": series_mode,
        "nivel_max_educativo": series_mode,
        "estado_ocupacional": series_mode,
    }

    households = (
        dataframe.groupby("household_id", dropna=False)
        .agg(aggregations)
        .rename(
            columns={
                "comuna": "commune",
                "miembro": "household_members",
                "edad": "average_age",
                "ingresos_familiares": "household_income",
                "ingreso_per_capita_familiar": "per_capita_income",
                "ingreso_total_lab": "total_labor_income",
                "ingreso_total_no_lab": "total_non_labor_income",
                "anos_escolaridad": "average_school_years",
                "situacion_conyugal": "dominant_marital_status",
                "nivel_max_educativo": "dominant_education_level",
                "estado_ocupacional": "dominant_employment_status",
            }
        )
        .reset_index()
    )

    households["dominant_marital_status"] = households["dominant_marital_status"].map(
        lambda value: translate_label(value, MARITAL_STATUS_LABELS)
    )
    households["dominant_education_level"] = households["dominant_education_level"].map(
        lambda value: translate_label(value, EDUCATION_LABELS)
    )
    households["dominant_employment_status"] = households["dominant_employment_status"].map(
        lambda value: translate_label(value, EMPLOYMENT_STATUS_LABELS)
    )

    return households


def build_commune_summary(households: pd.DataFrame) -> pd.DataFrame:
    summary = (
        households.groupby("commune", dropna=False)
        .agg(
            households=("household_id", "count"),
            median_household_income=("household_income", "median"),
            median_per_capita_income=("per_capita_income", "median"),
            average_members=("household_members", "mean"),
        )
        .reset_index()
        .sort_values("median_household_income", ascending=False)
    )
    return summary


def build_education_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    adults = dataframe.loc[dataframe["edad"] >= 18].copy()
    summary = (
        adults.groupby("nivel_max_educativo", dropna=False)
        .agg(
            people=("id", "count"),
            median_total_income=("ingresos_totales", "median"),
            average_total_income=("ingresos_totales", "mean"),
        )
        .reset_index()
        .rename(columns={"nivel_max_educativo": "education_level"})
        .sort_values("average_total_income", ascending=False)
    )
    summary["education_level"] = summary["education_level"].map(lambda value: translate_label(value, EDUCATION_LABELS))
    return summary


def build_marital_status_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    adults = dataframe.loc[dataframe["edad"] >= 18].copy()
    summary = (
        adults.groupby("situacion_conyugal", dropna=False)
        .agg(
            people=("id", "count"),
            median_per_capita_income=("ingreso_per_capita_familiar", "median"),
        )
        .reset_index()
        .rename(columns={"situacion_conyugal": "marital_status"})
        .sort_values("median_per_capita_income", ascending=False)
    )
    summary["marital_status"] = summary["marital_status"].map(
        lambda value: translate_label(value, MARITAL_STATUS_LABELS)
    )
    return summary


def export_tables(
    households: pd.DataFrame,
    commune_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    marital_status_summary: pd.DataFrame,
) -> None:
    households.to_csv(TABLES_DIR / "household_dataset.csv", index=False)
    commune_summary.to_csv(TABLES_DIR / "commune_income_summary.csv", index=False)
    education_summary.to_csv(TABLES_DIR / "education_income_summary.csv", index=False)
    marital_status_summary.to_csv(TABLES_DIR / "marital_status_income_summary.csv", index=False)


def export_figures(commune_summary: pd.DataFrame, education_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=commune_summary,
        x="commune",
        y="median_household_income",
        color="#1f4b6e",
    )
    plt.title("Median household income by commune")
    plt.xlabel("Commune")
    plt.ylabel("Median household income")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "median_household_income_by_commune.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=education_summary,
        x="education_level",
        y="average_total_income",
        color="#b36a32",
    )
    plt.title("Average total income by education level")
    plt.xlabel("Education level")
    plt.ylabel("Average total income")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "average_income_by_education.png", dpi=150)
    plt.close()


def print_summary(households: pd.DataFrame, commune_summary: pd.DataFrame) -> None:
    print("=== Project summary ===")
    print(f"Households analyzed: {len(households):,}")
    print(f"Communes covered: {households['commune'].nunique()}")

    print("\nTop 5 communes by median household income:")
    print(commune_summary.head(5).to_string(index=False))

    print("\nGenerated files:")
    print(f"- {TABLES_DIR / 'household_dataset.csv'}")
    print(f"- {TABLES_DIR / 'commune_income_summary.csv'}")
    print(f"- {TABLES_DIR / 'education_income_summary.csv'}")
    print(f"- {TABLES_DIR / 'marital_status_income_summary.csv'}")
    print(f"- {FIGURES_DIR / 'median_household_income_by_commune.png'}")
    print(f"- {FIGURES_DIR / 'average_income_by_education.png'}")


def main() -> None:
    args = parse_args()
    ensure_directories()
    dataframe = load_dataset(args.input_path)
    dataframe = cast_numeric_columns(dataframe)

    households = build_household_dataset(dataframe)
    commune_summary = build_commune_summary(households)
    education_summary = build_education_summary(dataframe)
    marital_status_summary = build_marital_status_summary(dataframe)

    export_tables(households, commune_summary, education_summary, marital_status_summary)
    export_figures(commune_summary, education_summary)
    print_summary(households, commune_summary)


if __name__ == "__main__":
    main()
