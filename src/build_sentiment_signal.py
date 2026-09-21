from pathlib import Path
import json

import numpy as np
import pandas as pd


INPUT = Path(
    "results/distilbert_test_predictions.csv"
)

RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> None:
    print("=" * 80)
    print("EC3 -> EC2: CONSTRUCCIÓN DE SEÑAL DE SENTIMIENTO")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    print("\n[1] DATASET DE PREDICCIONES")

    print(
        f"Filas: {len(df):,}"
    )

    print(
        f"Columnas: {len(df.columns)}"
    )

    required_columns = {
        "timestamp",
        "prob_neutral",
        "prob_bearish",
        "prob_bullish",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}"
        )

    # ------------------------------------------------------------
    # 2. Timestamp
    # ------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df["date"] = (
        df["timestamp"]
        .dt.floor("D")
    )

    # ------------------------------------------------------------
    # 3. Validación de probabilidades
    # ------------------------------------------------------------

    probability_columns = [
        "prob_neutral",
        "prob_bearish",
        "prob_bullish",
    ]

    probability_sum = (
        df[probability_columns]
        .sum(axis=1)
    )

    max_probability_error = float(
        np.abs(
            probability_sum - 1.0
        ).max()
    )

    print("\n[2] VALIDACIÓN DE PROBABILIDADES")

    print(
        "Máximo error respecto a suma=1: "
        f"{max_probability_error:.8f}"
    )

    # ------------------------------------------------------------
    # 4. Señal individual por noticia
    # ------------------------------------------------------------

    df["sentiment_score"] = (
        df["prob_bullish"]
        - df["prob_bearish"]
    )

    df["sentiment_strength"] = (
        df["sentiment_score"]
        .abs()
    )

    df["sentiment_entropy"] = -(
        df[probability_columns]
        .clip(
            lower=1e-12
        )
        .mul(
            np.log(
                df[probability_columns]
                .clip(
                    lower=1e-12
                )
            )
        )
        .sum(axis=1)
    )

    print("\n[3] SEÑAL POR DOCUMENTO")

    print(
        df[
            "sentiment_score"
        ]
        .describe()
    )

    # ------------------------------------------------------------
    # 5. Agregación diaria
    # ------------------------------------------------------------

    daily = (
        df.groupby("date")
        .agg(
            sentiment_mean=(
                "sentiment_score",
                "mean",
            ),

            sentiment_median=(
                "sentiment_score",
                "median",
            ),

            sentiment_std=(
                "sentiment_score",
                "std",
            ),

            sentiment_min=(
                "sentiment_score",
                "min",
            ),

            sentiment_max=(
                "sentiment_score",
                "max",
            ),

            sentiment_strength_mean=(
                "sentiment_strength",
                "mean",
            ),

            sentiment_entropy_mean=(
                "sentiment_entropy",
                "mean",
            ),

            prob_neutral_mean=(
                "prob_neutral",
                "mean",
            ),

            prob_bearish_mean=(
                "prob_bearish",
                "mean",
            ),

            prob_bullish_mean=(
                "prob_bullish",
                "mean",
            ),

            news_count=(
                "sentiment_score",
                "size",
            ),
        )
        .reset_index()
        .sort_values("date")
    )

    daily[
        "sentiment_std"
    ] = (
        daily[
            "sentiment_std"
        ]
        .fillna(0.0)
    )

    # ------------------------------------------------------------
    # 6. Indicadores derivados
    # ------------------------------------------------------------

    daily[
        "bullish_share_proxy"
    ] = (
        daily[
            "prob_bullish_mean"
        ]
    )

    daily[
        "bearish_share_proxy"
    ] = (
        daily[
            "prob_bearish_mean"
        ]
    )

    daily[
        "neutral_share_proxy"
    ] = (
        daily[
            "prob_neutral_mean"
        ]
    )

    daily[
        "sentiment_direction"
    ] = np.select(
        [
            daily[
                "sentiment_mean"
            ] > 0,

            daily[
                "sentiment_mean"
            ] < 0,
        ],
        [
            "bullish",
            "bearish",
        ],
        default="neutral",
    )

    # ------------------------------------------------------------
    # 7. Cobertura temporal
    # ------------------------------------------------------------

    print("\n[4] COBERTURA TEMPORAL")

    print(
        f"Fecha mínima: "
        f"{daily['date'].min()}"
    )

    print(
        f"Fecha máxima: "
        f"{daily['date'].max()}"
    )

    print(
        f"Días con noticias: "
        f"{len(daily):,}"
    )

    calendar_days = (
        (
            daily["date"].max()
            - daily["date"].min()
        ).days
        + 1
    )

    coverage_pct = (
        len(daily)
        / calendar_days
        * 100
    )

    print(
        f"Días calendario: "
        f"{calendar_days:,}"
    )

    print(
        f"Cobertura: "
        f"{coverage_pct:.2f}%"
    )

    # ------------------------------------------------------------
    # 8. Resumen de señal
    # ------------------------------------------------------------

    print("\n[5] RESUMEN DIARIO")

    print(
        daily[
            [
                "sentiment_mean",
                "sentiment_median",
                "sentiment_std",
                "news_count",
            ]
        ].describe()
    )

    # ------------------------------------------------------------
    # 9. Guardar documento-level
    # ------------------------------------------------------------

    document_output = (
        PROCESSED_DIR
        / "distilbert_sentiment_documents.parquet"
    )

    df.to_parquet(
        document_output,
        index=False,
    )

    # ------------------------------------------------------------
    # 10. Guardar señal diaria
    # ------------------------------------------------------------

    daily_csv = (
        RESULTS_DIR
        / "contextual_sentiment_daily.csv"
    )

    daily_parquet = (
        PROCESSED_DIR
        / "contextual_sentiment_daily.parquet"
    )

    daily.to_csv(
        daily_csv,
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_parquet(
        daily_parquet,
        index=False,
    )

    # ------------------------------------------------------------
    # 11. Metadata
    # ------------------------------------------------------------

    metadata = {
        "source_model": "DistilBERT",

        "source_file": str(INPUT),

        "signal_definition": (
            "P(bullish) - P(bearish)"
        ),

        "aggregation_level": "daily",

        "documents": int(
            len(df)
        ),

        "days_with_news": int(
            len(daily)
        ),

        "calendar_days": int(
            calendar_days
        ),

        "coverage_pct": float(
            coverage_pct
        ),

        "date_min": str(
            daily["date"].min()
        ),

        "date_max": str(
            daily["date"].max()
        ),

        "mean_sentiment": float(
            daily[
                "sentiment_mean"
            ].mean()
        ),

        "median_sentiment": float(
            daily[
                "sentiment_median"
            ].median()
        ),

        "mean_news_per_day": float(
            daily[
                "news_count"
            ].mean()
        ),

        "max_probability_sum_error": (
            max_probability_error
        ),
    }

    with open(
        RESULTS_DIR
        / "sentiment_signal_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[6] ARCHIVOS GENERADOS")

    outputs = [
        document_output,
        daily_csv,
        daily_parquet,
        (
            RESULTS_DIR
            / "sentiment_signal_metadata.json"
        ),
    ]

    for output in outputs:
        print(
            f"  - {output}"
        )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "La señal contextual diaria "
        "ya puede integrarse con variables "
        "de mercado por fecha."
    )


if __name__ == "__main__":
    main()