from pathlib import Path
import json

import pandas as pd


INPUT = Path(
    "data/processed/"
    "dlt_sentiment_news_clean.parquet"
)

OUTPUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LABEL_MAP = {
    0: "neutral",
    1: "bearish",
    2: "bullish",
}


def main() -> None:
    print("=" * 80)
    print("EC3 - PREPARACIÓN DEL SPLIT TEMPORAL")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe: {INPUT}"
        )

    df = pd.read_parquet(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    # ------------------------------------------------------------
    # 1. Etiquetas
    # ------------------------------------------------------------
    print("\n[1] MAPEO DE ETIQUETAS")

    observed = set(
        df["market_direction"]
        .dropna()
        .astype(int)
        .unique()
    )

    expected = set(LABEL_MAP.keys())

    if not observed.issubset(expected):
        raise ValueError(
            "Se encontraron etiquetas "
            f"inesperadas: {observed - expected}"
        )

    df["label_id"] = (
        df["market_direction"]
        .astype(int)
    )

    df["label_name"] = (
        df["label_id"]
        .map(LABEL_MAP)
    )

    print(
        df[
            [
                "label_id",
                "label_name",
            ]
        ]
        .drop_duplicates()
        .sort_values("label_id")
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # 2. Split temporal
    # ------------------------------------------------------------
    print("\n[2] SPLIT TEMPORAL")

    year = df["timestamp"].dt.year

    df["split"] = pd.NA

    df.loc[
        year <= 2023,
        "split",
    ] = "train"

    df.loc[
        year == 2024,
        "split",
    ] = "validation"

    df.loc[
        year == 2025,
        "split",
    ] = "test"

    unassigned = int(
        df["split"]
        .isna()
        .sum()
    )

    if unassigned != 0:
        raise ValueError(
            f"Hay {unassigned} filas "
            "sin split temporal."
        )

    # ------------------------------------------------------------
    # 3. Validación temporal
    # ------------------------------------------------------------
    train = (
        df[df["split"] == "train"]
        .copy()
    )

    validation = (
        df[
            df["split"]
            == "validation"
        ]
        .copy()
    )

    test = (
        df[df["split"] == "test"]
        .copy()
    )

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print(
        "\nRangos temporales:"
    )

    for name, split_df in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        print(
            f"{name:10s}: "
            f"{split_df['timestamp'].min()} "
            f"-> "
            f"{split_df['timestamp'].max()}"
        )

    # ------------------------------------------------------------
    # 4. Verificación de leakage temporal
    # ------------------------------------------------------------
    print(
        "\n[3] VERIFICACIÓN TEMPORAL"
    )

    assert (
        train["timestamp"].max()
        <
        validation["timestamp"].min()
    )

    assert (
        validation["timestamp"].max()
        <
        test["timestamp"].min()
    )

    print(
        "OK: train < validation < test"
    )

    # ------------------------------------------------------------
    # 5. Distribución de clases
    # ------------------------------------------------------------
    print(
        "\n[4] DISTRIBUCIÓN DE CLASES"
    )

    distribution = pd.crosstab(
        df["split"],
        df["label_name"],
    )

    distribution = distribution.reindex(
        [
            "train",
            "validation",
            "test",
        ]
    )

    print(distribution)

    distribution.to_csv(
        RESULTS_DIR
        / "split_class_distribution.csv",
        encoding="utf-8-sig",
    )

    distribution_pct = pd.crosstab(
        df["split"],
        df["label_name"],
        normalize="index",
    ) * 100

    distribution_pct = (
        distribution_pct
        .reindex(
            [
                "train",
                "validation",
                "test",
            ]
        )
    )

    print(
        "\nPorcentajes por split:"
    )

    print(
        distribution_pct.round(2)
    )

    distribution_pct.to_csv(
        RESULTS_DIR
        / "split_class_distribution_pct.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 6. Guardado
    # ------------------------------------------------------------
    print(
        "\n[5] GUARDANDO SPLITS"
    )

    train.to_parquet(
        OUTPUT_DIR
        / "train.parquet",
        index=False,
    )

    validation.to_parquet(
        OUTPUT_DIR
        / "validation.parquet",
        index=False,
    )

    test.to_parquet(
        OUTPUT_DIR
        / "test.parquet",
        index=False,
    )

    df.to_parquet(
        OUTPUT_DIR
        / "dlt_sentiment_news_split.parquet",
        index=False,
    )

    metadata = {
        "strategy": (
            "strict chronological "
            "year-based split"
        ),
        "train_period": {
            "start": str(
                train[
                    "timestamp"
                ].min()
            ),
            "end": str(
                train[
                    "timestamp"
                ].max()
            ),
            "rows": int(
                len(train)
            ),
        },
        "validation_period": {
            "start": str(
                validation[
                    "timestamp"
                ].min()
            ),
            "end": str(
                validation[
                    "timestamp"
                ].max()
            ),
            "rows": int(
                len(validation)
            ),
        },
        "test_period": {
            "start": str(
                test[
                    "timestamp"
                ].min()
            ),
            "end": str(
                test[
                    "timestamp"
                ].max()
            ),
            "rows": int(
                len(test)
            ),
        },
        "label_mapping": {
            str(key): value
            for key, value
            in LABEL_MAP.items()
        },
        "random_split": False,
        "test_used_for_tuning": False,
    }

    with open(
        RESULTS_DIR
        / "split_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nArchivos generados:"
    )

    for filename in [
        "data/processed/train.parquet",
        "data/processed/validation.parquet",
        "data/processed/test.parquet",
        (
            "data/processed/"
            "dlt_sentiment_news_split.parquet"
        ),
        (
            "results/"
            "split_class_distribution.csv"
        ),
        (
            "results/"
            "split_class_distribution_pct.csv"
        ),
        (
            "results/"
            "split_metadata.json"
        ),
    ]:
        print(
            f"  - {filename}"
        )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "El conjunto test queda "
        "congelado desde este punto."
    )

    print(
        "No debe utilizarse para "
        "selección de hiperparámetros."
    )


if __name__ == "__main__":
    main()