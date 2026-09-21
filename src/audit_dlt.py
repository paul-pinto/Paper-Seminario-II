from pathlib import Path
import hashlib
import json

import pandas as pd


INPUT = Path("data/raw/dlt_sentiment_news.parquet")
RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 80)
    print("EC3 - AUDITORÍA DEL DATASET DLT-SENTIMENT-NEWS")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset esperado en: {INPUT}"
        )

    # ------------------------------------------------------------------
    # 1. Carga
    # ------------------------------------------------------------------
    df = pd.read_parquet(INPUT)

    print("\n[1] DIMENSIONES ORIGINALES")
    print(f"Filas:    {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    print("\nColumnas disponibles:")
    for column in df.columns:
        print(f"  - {column}: {df[column].dtype}")

    # ------------------------------------------------------------------
    # 2. Conversión temporal
    # ------------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    print("\n[2] COBERTURA TEMPORAL")
    print(f"Fecha mínima: {df['timestamp'].min()}")
    print(f"Fecha máxima: {df['timestamp'].max()}")

    invalid_dates = int(df["timestamp"].isna().sum())

    print(
        f"Timestamps inválidos/nulos: "
        f"{invalid_dates:,}"
    )

    # ------------------------------------------------------------------
    # 3. Valores faltantes
    # ------------------------------------------------------------------
    print("\n[3] VALORES FALTANTES")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing")
        .to_frame()
    )

    missing["pct"] = (
        missing["missing"]
        / len(df)
        * 100
    )

    print(missing)

    missing.to_csv(
        RESULTS_DIR / "missing_values.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 4. Distribución de la variable objetivo
    # ------------------------------------------------------------------
    target = "market_direction"

    print("\n[4] DISTRIBUCIÓN DE CLASES")

    class_distribution = (
        df[target]
        .value_counts(dropna=False)
        .rename("count")
        .to_frame()
    )

    class_distribution["pct"] = (
        class_distribution["count"]
        / len(df)
        * 100
    )

    print(class_distribution)

    class_distribution.to_csv(
        RESULTS_DIR / "class_distribution.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 5. Normalización mínima SOLO PARA AUDITORÍA
    # ------------------------------------------------------------------
    #
    # OJO metodológico:
    # No hacemos stemming, lemmatization ni eliminación de stopwords.
    # En Transformers eso podría destruir información contextual.
    # Aquí solo normalizamos espacios para detectar duplicados exactos.
    #
    df["text_clean_audit"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    # ------------------------------------------------------------------
    # 6. Textos vacíos
    # ------------------------------------------------------------------
    empty_text = int(
        df["text_clean_audit"]
        .eq("")
        .sum()
    )

    print("\n[5] CALIDAD DEL TEXTO")
    print(
        f"Textos vacíos después de normalización mínima: "
        f"{empty_text:,}"
    )

    # ------------------------------------------------------------------
    # 7. Duplicados exactos
    # ------------------------------------------------------------------
    duplicated_mask = df.duplicated(
        subset=["text_clean_audit"],
        keep=False,
    )

    duplicated_rows = df.loc[
        duplicated_mask
    ].copy()

    duplicated_unique_texts = (
        duplicated_rows[
            "text_clean_audit"
        ]
        .nunique()
    )

    print("\n[6] DUPLICADOS EXACTOS")
    print(
        f"Filas involucradas en duplicados exactos: "
        f"{len(duplicated_rows):,}"
    )

    print(
        f"Textos únicos que aparecen repetidos: "
        f"{duplicated_unique_texts:,}"
    )

    if len(duplicated_rows) > 0:
        duplicated_rows[
            [
                "timestamp",
                "title",
                "text",
                target,
                "url",
            ]
        ].sort_values(
            "text"
        ).to_csv(
            RESULTS_DIR / "exact_duplicates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ------------------------------------------------------------------
    # 8. Duplicados con etiquetas conflictivas
    # ------------------------------------------------------------------
    label_counts_per_text = (
        df.groupby(
            "text_clean_audit"
        )[target]
        .nunique(dropna=True)
    )

    conflicting_texts = (
        label_counts_per_text[
            label_counts_per_text > 1
        ]
    )

    print(
        f"Textos repetidos con etiquetas conflictivas: "
        f"{len(conflicting_texts):,}"
    )

    if len(conflicting_texts) > 0:
        conflict_df = df[
            df["text_clean_audit"].isin(
                conflicting_texts.index
            )
        ].copy()

        conflict_df[
            [
                "timestamp",
                "title",
                "text",
                target,
                "url",
            ]
        ].sort_values(
            "text"
        ).to_csv(
            RESULTS_DIR
            / "duplicate_label_conflicts.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ------------------------------------------------------------------
    # 9. Longitud textual
    # ------------------------------------------------------------------
    df["char_length"] = (
        df["text_clean_audit"]
        .str.len()
    )

    df["word_length"] = (
        df["text_clean_audit"]
        .str.split()
        .str.len()
    )

    length_columns = [
        "char_length",
        "word_length",
    ]

    if "total_tokens" in df.columns:
        length_columns.append(
            "total_tokens"
        )

    length_stats = (
        df[length_columns]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\n[7] LONGITUD DE LOS TEXTOS")
    print(length_stats)

    length_stats.to_csv(
        RESULTS_DIR
        / "text_length_stats.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 10. Distribución temporal
    # ------------------------------------------------------------------
    df["year"] = (
        df["timestamp"]
        .dt.year
    )

    df["month"] = (
        df["timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    print("\n[8] DISTRIBUCIÓN DE CLASES POR AÑO")

    year_class = pd.crosstab(
        df["year"],
        df[target],
        margins=True,
    )

    print(year_class)

    year_class.to_csv(
        RESULTS_DIR
        / "class_distribution_by_year.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 11. Distribución mensual
    # ------------------------------------------------------------------
    month_class = pd.crosstab(
        df["month"],
        df[target],
    )

    month_class.to_csv(
        RESULTS_DIR
        / "class_distribution_by_month.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 12. Fuentes
    # ------------------------------------------------------------------
    if "source_url" in df.columns:
        unique_sources = int(
            df["source_url"]
            .nunique(dropna=True)
        )
    else:
        unique_sources = None

    if "url" in df.columns:
        unique_urls = int(
            df["url"]
            .nunique(dropna=True)
        )
    else:
        unique_urls = None

    print("\n[9] FUENTES")
    print(
        f"source_url únicos: "
        f"{unique_sources}"
    )
    print(
        f"url únicos: "
        f"{unique_urls}"
    )

    # ------------------------------------------------------------------
    # 13. Auditoría de etiquetas
    # ------------------------------------------------------------------
    valid_labels = {
        "bullish",
        "bearish",
        "neutral",
    }

    observed_labels = set(
        df[target]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected_labels = (
        observed_labels
        - valid_labels
    )

    print("\n[10] AUDITORÍA DE ETIQUETAS")
    print(
        f"Etiquetas observadas: "
        f"{sorted(observed_labels)}"
    )

    print(
        f"Etiquetas inesperadas: "
        f"{sorted(unexpected_labels)}"
    )

    # ------------------------------------------------------------------
    # 14. Dataset limpio candidato
    # ------------------------------------------------------------------
    #
    # IMPORTANTE:
    # Todavía NO hacemos train/validation/test.
    # Primero auditamos la distribución temporal.
    #
    clean = df[
        df["timestamp"].notna()
        & df[target].notna()
        & df["text_clean_audit"].ne("")
    ].copy()

    before_dedup = len(clean)

    clean = (
        clean
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "text_clean_audit"
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    after_dedup = len(clean)

    removed_duplicates = (
        before_dedup
        - after_dedup
    )

    print("\n[11] DATASET LIMPIO CANDIDATO")
    print(
        f"Filas antes de deduplicación: "
        f"{before_dedup:,}"
    )

    print(
        f"Filas después de deduplicación: "
        f"{after_dedup:,}"
    )

    print(
        f"Filas eliminadas por duplicación: "
        f"{removed_duplicates:,}"
    )

    clean_output = (
        PROCESSED_DIR
        / "dlt_sentiment_news_clean.parquet"
    )

    clean.to_parquet(
        clean_output,
        index=False,
    )

    print(
        f"Guardado: {clean_output}"
    )

    # ------------------------------------------------------------------
    # 15. Distribución final limpia
    # ------------------------------------------------------------------
    clean_distribution = (
        clean[target]
        .value_counts()
        .rename("count")
        .to_frame()
    )

    clean_distribution["pct"] = (
        clean_distribution["count"]
        / len(clean)
        * 100
    )

    clean_distribution.to_csv(
        RESULTS_DIR
        / "clean_class_distribution.csv",
        encoding="utf-8-sig",
    )

    clean_year_class = pd.crosstab(
        clean["year"],
        clean[target],
        margins=True,
    )

    clean_year_class.to_csv(
        RESULTS_DIR
        / "clean_class_distribution_by_year.csv",
        encoding="utf-8-sig",
    )

    print("\nDistribución final limpia:")
    print(clean_distribution)

    # ------------------------------------------------------------------
    # 16. Hash SHA-256 del raw
    # ------------------------------------------------------------------
    print("\n[12] REPRODUCIBILIDAD")

    sha256 = hashlib.sha256(
        INPUT.read_bytes()
    ).hexdigest()

    print(
        f"SHA256 dataset raw:\n{sha256}"
    )

    metadata = {
        "dataset": (
            "ExponentialScience/"
            "DLT-Sentiment-News"
        ),
        "raw_file": str(INPUT),
        "raw_rows": int(len(df)),
        "raw_columns": int(
            len(df.columns)
        ),
        "clean_rows": int(
            len(clean)
        ),
        "removed_duplicates": int(
            removed_duplicates
        ),
        "timestamp_min": str(
            clean["timestamp"].min()
        ),
        "timestamp_max": str(
            clean["timestamp"].max()
        ),
        "target": target,
        "labels": sorted(
            observed_labels
        ),
        "unique_sources": (
            unique_sources
        ),
        "unique_urls": (
            unique_urls
        ),
        "sha256_raw": sha256,
    }

    with open(
        RESULTS_DIR
        / "dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # 17. Resumen
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AUDITORÍA COMPLETADA")
    print("=" * 80)

    print(
        "\nArchivos generados:"
    )

    outputs = [
        "results/missing_values.csv",
        "results/class_distribution.csv",
        "results/class_distribution_by_year.csv",
        "results/class_distribution_by_month.csv",
        "results/text_length_stats.csv",
        "results/exact_duplicates.csv",
        "results/duplicate_label_conflicts.csv",
        "results/clean_class_distribution.csv",
        "results/clean_class_distribution_by_year.csv",
        "results/dataset_metadata.json",
        "data/processed/dlt_sentiment_news_clean.parquet",
    ]

    for output in outputs:
        print(f"  - {output}")

    print(
        "\nTodavía NO se ha realizado "
        "el split train/validation/test."
    )

    print(
        "El corte temporal se definirá "
        "después de revisar esta auditoría."
    )


if __name__ == "__main__":
    main()