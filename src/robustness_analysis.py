from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


RESULTS_DIR = Path("results")

METRICS_FILE = (
    RESULTS_DIR
    / "recommender_daily_metrics.csv"
)

COEFFICIENTS_FILE = (
    RESULTS_DIR
    / "recommender_model_coefficients.csv"
)

BOOTSTRAP_ITERATIONS = 10000
SEED = 42


def paired_bootstrap(
    values: np.ndarray,
    iterations: int = 10000,
    seed: int = 42,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    rng = np.random.default_rng(
        seed
    )

    observed = float(
        values.mean()
    )

    samples = np.empty(
        iterations,
        dtype=float,
    )

    n = len(values)

    for i in range(iterations):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        samples[i] = (
            values[idx]
            .mean()
        )

    lower = float(
        np.percentile(
            samples,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            samples,
            97.5,
        )
    )

    return (
        observed,
        lower,
        upper,
    )


def compare_by_profile(
    metrics: pd.DataFrame,
):

    rows = []

    for profile in sorted(
        metrics["profile"].unique()
    ):

        subset = metrics[
            metrics["profile"]
            == profile
        ]

        pivot = subset.pivot_table(
            index="date",
            columns="treatment",
            values="ndcg_at_10",
        ).dropna()

        pivot[
            "difference"
        ] = (
            pivot["T1"]
            - pivot["T0"]
        )

        (
            mean_diff,
            ci_low,
            ci_high,
        ) = paired_bootstrap(
            pivot[
                "difference"
            ].to_numpy(),
            iterations=(
                BOOTSTRAP_ITERATIONS
            ),
            seed=SEED,
        )

        nonzero = pivot[
            "difference"
        ][
            pivot[
                "difference"
            ] != 0
        ]

        if len(nonzero) > 0:

            result = wilcoxon(
                nonzero,
                alternative="two-sided",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        else:

            statistic = 0.0
            p_value = 1.0

        improved = int(
            (
                pivot[
                    "difference"
                ] > 0
            ).sum()
        )

        worsened = int(
            (
                pivot[
                    "difference"
                ] < 0
            ).sum()
        )

        tied = int(
            (
                pivot[
                    "difference"
                ] == 0
            ).sum()
        )

        rows.append(
            {
                "profile":
                    profile,

                "n_dates":
                    len(pivot),

                "mean_delta_ndcg":
                    mean_diff,

                "ci95_lower":
                    ci_low,

                "ci95_upper":
                    ci_high,

                "wilcoxon_statistic":
                    statistic,

                "wilcoxon_p":
                    p_value,

                "days_T1_better":
                    improved,

                "days_equal":
                    tied,

                "days_T1_worse":
                    worsened,

                "pct_days_T1_better":
                    (
                        improved
                        / len(pivot)
                        * 100
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_by_regime(
    metrics: pd.DataFrame,
):

    rows = []

    for (
        regime,
        profile,
    ), subset in metrics.groupby(
        [
            "regime",
            "profile",
        ]
    ):

        pivot = subset.pivot_table(
            index="date",
            columns="treatment",
            values="ndcg_at_10",
        ).dropna()

        if pivot.empty:
            continue

        pivot[
            "difference"
        ] = (
            pivot["T1"]
            - pivot["T0"]
        )

        values = pivot[
            "difference"
        ].to_numpy()

        mean_diff = float(
            np.mean(values)
        )

        median_diff = float(
            np.median(values)
        )

        improved = int(
            np.sum(
                values > 0
            )
        )

        worsened = int(
            np.sum(
                values < 0
            )
        )

        tied = int(
            np.sum(
                values == 0
            )
        )

        rows.append(
            {
                "regime":
                    regime,

                "profile":
                    profile,

                "n_dates":
                    len(values),

                "mean_delta_ndcg":
                    mean_diff,

                "median_delta_ndcg":
                    median_diff,

                "days_T1_better":
                    improved,

                "days_equal":
                    tied,

                "days_T1_worse":
                    worsened,
            }
        )

    return pd.DataFrame(
        rows
    )


def global_daily_distribution(
    metrics: pd.DataFrame,
):

    pivot = metrics.pivot_table(
        index=[
            "date",
            "profile",
        ],
        columns="treatment",
        values="ndcg_at_10",
    ).dropna()

    pivot[
        "difference"
    ] = (
        pivot["T1"]
        - pivot["T0"]
    )

    daily = (
        pivot.reset_index()
        .groupby("date")
        ["difference"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "difference":
                    "mean_delta_ndcg"
            }
        )
    )

    values = daily[
        "mean_delta_ndcg"
    ]

    summary = pd.DataFrame(
        [
            {
                "n_dates":
                    len(values),

                "mean":
                    float(
                        values.mean()
                    ),

                "median":
                    float(
                        values.median()
                    ),

                "std":
                    float(
                        values.std()
                    ),

                "min":
                    float(
                        values.min()
                    ),

                "q25":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),

                "q75":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),

                "max":
                    float(
                        values.max()
                    ),

                "days_positive":
                    int(
                        (
                            values > 0
                        ).sum()
                    ),

                "days_zero":
                    int(
                        (
                            values == 0
                        ).sum()
                    ),

                "days_negative":
                    int(
                        (
                            values < 0
                        ).sum()
                    ),
            }
        ]
    )

    return (
        daily,
        summary,
    )


def analyze_coefficients(
    coefficients: pd.DataFrame,
):

    sentiment = coefficients[
        (
            coefficients[
                "treatment"
            ] == "T1"
        )
        &
        (
            coefficients[
                "feature"
            ]
            == "sentiment_exposure"
        )
    ].copy()

    return sentiment


def main():

    print("=" * 80)
    print("ANÁLISIS DE ROBUSTEZ FINAL")
    print("=" * 80)

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            METRICS_FILE
        )

    if not COEFFICIENTS_FILE.exists():
        raise FileNotFoundError(
            COEFFICIENTS_FILE
        )

    metrics = pd.read_csv(
        METRICS_FILE
    )

    coefficients = pd.read_csv(
        COEFFICIENTS_FILE
    )

    metrics[
        "date"
    ] = pd.to_datetime(
        metrics["date"]
    )

    # ========================================================
    # 1. Por perfil
    # ========================================================

    profile_stats = (
        compare_by_profile(
            metrics
        )
    )

    print(
        "\n[1] ROBUSTEZ POR PERFIL"
    )

    print(
        profile_stats.to_string(
            index=False
        )
    )

    profile_stats.to_csv(
        RESULTS_DIR
        / "robustness_by_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 2. Por régimen
    # ========================================================

    regime_stats = (
        compare_by_regime(
            metrics
        )
    )

    print(
        "\n[2] RESULTADOS DESCRIPTIVOS POR RÉGIMEN"
    )

    print(
        regime_stats.to_string(
            index=False
        )
    )

    regime_stats.to_csv(
        RESULTS_DIR
        / "robustness_by_regime.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 3. Distribución diaria
    # ========================================================

    (
        daily_delta,
        daily_summary,
    ) = (
        global_daily_distribution(
            metrics
        )
    )

    print(
        "\n[3] DISTRIBUCIÓN DIARIA DEL DELTA NDCG@10"
    )

    print(
        daily_summary.to_string(
            index=False
        )
    )

    daily_delta.to_csv(
        RESULTS_DIR
        / "daily_delta_ndcg.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily_summary.to_csv(
        RESULTS_DIR
        / "daily_delta_ndcg_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 4. Coeficiente de sentimiento
    # ========================================================

    sentiment_coefficients = (
        analyze_coefficients(
            coefficients
        )
    )

    print(
        "\n[4] COEFICIENTE DE SENTIMENT_EXPOSURE"
    )

    print(
        sentiment_coefficients.to_string(
            index=False
        )
    )

    sentiment_coefficients.to_csv(
        RESULTS_DIR
        / "sentiment_exposure_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 5. Archivos
    # ========================================================

    print(
        "\n[5] ARCHIVOS GENERADOS"
    )

    outputs = [
        "robustness_by_profile.csv",
        "robustness_by_regime.csv",
        "daily_delta_ndcg.csv",
        "daily_delta_ndcg_summary.csv",
        "sentiment_exposure_coefficients.csv",
    ]

    for filename in outputs:
        print(
            f"  - results/{filename}"
        )

    print(
        "\nANÁLISIS COMPLETADO"
    )


if __name__ == "__main__":
    main()