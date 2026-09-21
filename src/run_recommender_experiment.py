from pathlib import Path
import json

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN
# ============================================================

MARKET_FILE = Path(
    "data/processed/crypto_market_selected.parquet"
)

SENTIMENT_FILE = Path(
    "data/processed/contextual_sentiment_daily.parquet"
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CALIBRATION_START = pd.Timestamp(
    "2025-02-01"
)

CALIBRATION_END = pd.Timestamp(
    "2025-03-08"
)

EVALUATION_START = pd.Timestamp(
    "2025-03-16"
)

EVALUATION_END = pd.Timestamp(
    "2025-05-16"
)

FORWARD_HORIZON = 7

BETA_WINDOW = 30
BETA_MIN_PERIODS = 20

K = 10

RIDGE_ALPHA = 1.0

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


PROFILES = {
    "conservative": 1.00,
    "moderate": 0.50,
    "aggressive": 0.15,
}


BASE_FEATURES = [
    "momentum_7",
    "momentum_30",
    "volatility_14",
]

CONTEXT_FEATURES = [
    "momentum_7",
    "momentum_30",
    "volatility_14",
    "sentiment_exposure",
]


# ============================================================
# UTILIDADES
# ============================================================

def forward_volatility(
    returns: np.ndarray,
    horizon: int,
) -> np.ndarray:

    n = len(returns)

    result = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for i in range(n):

        start = i + 1
        end = i + 1 + horizon

        if end <= n:

            window = returns[
                start:end
            ]

            if np.isfinite(
                window
            ).all():

                result[i] = np.std(
                    window,
                    ddof=1,
                )

    return result


def dcg(
    relevance: np.ndarray,
) -> float:

    relevance = np.asarray(
        relevance,
        dtype=float,
    )

    if len(relevance) == 0:
        return np.nan

    positions = np.arange(
        2,
        len(relevance) + 2,
    )

    gains = (
        np.power(
            2.0,
            relevance,
        )
        - 1.0
    )

    discounts = np.log2(
        positions
    )

    return float(
        np.sum(
            gains / discounts
        )
    )


def ndcg_at_k(
    relevance: np.ndarray,
    predicted_score: np.ndarray,
    k: int,
) -> float:

    order = np.argsort(
        -predicted_score
    )

    ideal_order = np.argsort(
        -relevance
    )

    predicted_rel = relevance[
        order[:k]
    ]

    ideal_rel = relevance[
        ideal_order[:k]
    ]

    predicted_dcg = dcg(
        predicted_rel
    )

    ideal_dcg = dcg(
        ideal_rel
    )

    if ideal_dcg == 0:
        return np.nan

    return (
        predicted_dcg
        / ideal_dcg
    )


def recall_at_k(
    future_utility: np.ndarray,
    predicted_score: np.ndarray,
    k: int,
    truth_k: int = 5,
) -> float:

    predicted_order = np.argsort(
        -predicted_score
    )[:k]

    true_order = np.argsort(
        -future_utility
    )[:truth_k]

    hits = len(
        set(predicted_order)
        & set(true_order)
    )

    return hits / truth_k


def map_at_k(
    future_utility: np.ndarray,
    predicted_score: np.ndarray,
    k: int,
    truth_k: int = 5,
) -> float:

    predicted_order = np.argsort(
        -predicted_score
    )[:k]

    true_relevant = set(
        np.argsort(
            -future_utility
        )[:truth_k]
    )

    hits = 0
    precision_sum = 0.0

    for rank, idx in enumerate(
        predicted_order,
        start=1,
    ):

        if idx in true_relevant:

            hits += 1

            precision_sum += (
                hits / rank
            )

    denominator = min(
        truth_k,
        k,
    )

    if denominator == 0:
        return np.nan

    return (
        precision_sum
        / denominator
    )


def graded_relevance(
    utility: np.ndarray,
) -> np.ndarray:

    n = len(
        utility
    )

    order = np.argsort(
        -utility
    )

    relevance = np.zeros(
        n,
        dtype=float,
    )

    # Para 15 activos:
    # 1-3   -> 4
    # 4-6   -> 3
    # 7-9   -> 2
    # 10-12 -> 1
    # 13-15 -> 0

    grades = [
        4,
        3,
        2,
        1,
        0,
    ]

    groups = np.array_split(
        order,
        5,
    )

    for grade, indices in zip(
        grades,
        groups,
    ):
        relevance[
            indices
        ] = grade

    return relevance


def paired_bootstrap(
    differences: np.ndarray,
    iterations: int = 10000,
    seed: int = 42,
):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    rng = np.random.default_rng(
        seed
    )

    observed = float(
        differences.mean()
    )

    bootstrap_means = np.empty(
        iterations,
        dtype=float,
    )

    n = len(
        differences
    )

    for i in range(
        iterations
    ):

        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[
            i
        ] = sample.mean()

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return (
        observed,
        lower,
        upper,
    )


# ============================================================
# CARGA
# ============================================================

def load_data():

    if not MARKET_FILE.exists():
        raise FileNotFoundError(
            MARKET_FILE
        )

    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            SENTIMENT_FILE
        )

    market = pd.read_parquet(
        MARKET_FILE
    )

    sentiment = pd.read_parquet(
        SENTIMENT_FILE
    )

    market["date"] = pd.to_datetime(
        market["date"]
    )

    sentiment["date"] = pd.to_datetime(
        sentiment["date"]
    )

    return market, sentiment


# ============================================================
# PREPARACIÓN
# ============================================================

def prepare_panel(
    market: pd.DataFrame,
    sentiment: pd.DataFrame,
):

    print(
        "\n[1] PREPARACIÓN DEL PANEL"
    )

    sentiment = (
        sentiment[
            [
                "date",
                "sentiment_mean",
                "news_count",
            ]
        ]
        .drop_duplicates(
            subset="date"
        )
        .sort_values(
            "date"
        )
    )

    # --------------------------------------------------------
    # Calendario completo
    # --------------------------------------------------------

    calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                market["date"].min(),
                market["date"].max(),
                freq="D",
            )
        }
    )

    calendar = calendar.merge(
        sentiment,
        on="date",
        how="left",
    )

    calendar[
        "sentiment_observed"
    ] = (
        calendar[
            "sentiment_mean"
        ]
        .notna()
        .astype(int)
    )

    # Máximo 1 día de arrastre.
    calendar[
        "sentiment_mean"
    ] = (
        calendar[
            "sentiment_mean"
        ]
        .ffill(
            limit=1
        )
        .fillna(0.0)
    )

    calendar[
        "news_count"
    ] = (
        calendar[
            "news_count"
        ]
        .fillna(0)
    )

    panel = market.merge(
        calendar,
        on="date",
        how="left",
    )

    panel = panel.sort_values(
        [
            "asset",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Features financieros
    # --------------------------------------------------------

    panel[
        "return_1d"
    ] = (
        panel.groupby(
            "asset"
        )["close"]
        .pct_change()
    )

    panel[
        "momentum_7"
    ] = (
        panel.groupby(
            "asset"
        )["close"]
        .pct_change(
            periods=7
        )
    )

    panel[
        "momentum_30"
    ] = (
        panel.groupby(
            "asset"
        )["close"]
        .pct_change(
            periods=30
        )
    )

    panel[
        "volatility_14"
    ] = (
        panel.groupby(
            "asset"
        )["return_1d"]
        .transform(
            lambda x:
            x.rolling(
                14,
                min_periods=14,
            ).std()
        )
    )

    # --------------------------------------------------------
    # Sentimiento rezagado para estimar beta
    # --------------------------------------------------------

    panel[
        "sentiment_lag1"
    ] = (
        panel[
            "sentiment_mean"
        ]
        .shift(1)
    )

    # --------------------------------------------------------
    # Beta histórica:
    #
    # r_i,t = alpha + beta_i * S_(t-1)
    #
    # Se estima rolling y se desplaza un día.
    # Por tanto beta usada en t solo conoce <= t-1.
    # --------------------------------------------------------

    beta_frames = []

    for asset, group in panel.groupby(
        "asset",
        sort=False,
    ):

        group = group.copy()

        x = group[
            "sentiment_lag1"
        ]

        y = group[
            "return_1d"
        ]

        covariance = (
            y.rolling(
                BETA_WINDOW,
                min_periods=BETA_MIN_PERIODS,
            )
            .cov(x)
        )

        variance = (
            x.rolling(
                BETA_WINDOW,
                min_periods=BETA_MIN_PERIODS,
            )
            .var()
        )

        beta = (
            covariance
            / variance.replace(
                0,
                np.nan,
            )
        )

        group[
            "sentiment_beta"
        ] = beta.shift(
            1
        )

        beta_frames.append(
            group
        )

    panel = pd.concat(
        beta_frames,
        ignore_index=True,
    )

    panel[
        "sentiment_exposure"
    ] = (
        panel[
            "sentiment_beta"
        ]
        * panel[
            "sentiment_mean"
        ]
    )

    # --------------------------------------------------------
    # Forward return + forward volatility
    # --------------------------------------------------------

    forward_frames = []

    for asset, group in panel.groupby(
        "asset",
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "date"
            )
            .copy()
        )

        group[
            "forward_return_7"
        ] = (
            group[
                "close"
            ]
            .shift(
                -FORWARD_HORIZON
            )
            / group[
                "close"
            ]
            - 1.0
        )

        group[
            "forward_volatility_7"
        ] = forward_volatility(
            group[
                "return_1d"
            ].to_numpy(
                dtype=float
            ),
            FORWARD_HORIZON,
        )

        forward_frames.append(
            group
        )

    panel = pd.concat(
        forward_frames,
        ignore_index=True,
    )

    panel = panel.sort_values(
        [
            "date",
            "asset",
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Filas panel: "
        f"{len(panel):,}"
    )

    print(
        f"Activos: "
        f"{panel['asset'].nunique()}"
    )

    print(
        f"Fechas: "
        f"{panel['date'].min().date()} "
        f"-> "
        f"{panel['date'].max().date()}"
    )

    return panel


# ============================================================
# REGÍMENES
# ============================================================

def build_regimes(
    panel: pd.DataFrame,
):

    btc = (
        panel[
            panel[
                "asset"
            ] == "BTC"
        ][
            [
                "date",
                "close",
            ]
        ]
        .copy()
        .sort_values(
            "date"
        )
    )

    btc[
        "btc_momentum_14"
    ] = (
        btc[
            "close"
        ]
        .pct_change(
            14
        )
    )

    calibration_btc = btc[
        btc[
            "date"
        ].between(
            CALIBRATION_START,
            CALIBRATION_END,
        )
    ]

    lower = float(
        calibration_btc[
            "btc_momentum_14"
        ].quantile(
            1 / 3
        )
    )

    upper = float(
        calibration_btc[
            "btc_momentum_14"
        ].quantile(
            2 / 3
        )
    )

    btc[
        "regime"
    ] = np.select(
        [
            btc[
                "btc_momentum_14"
            ] < lower,

            btc[
                "btc_momentum_14"
            ] > upper,
        ],
        [
            "bear",
            "bull",
        ],
        default="sideways",
    )

    print(
        "\n[2] UMBRALES DE RÉGIMEN"
    )

    print(
        f"Bear < {lower:.4f}"
    )

    print(
        f"Bull > {upper:.4f}"
    )

    return (
        btc[
            [
                "date",
                "btc_momentum_14",
                "regime",
            ]
        ],
        lower,
        upper,
    )


# ============================================================
# EXPERIMENTO
# ============================================================

def run_experiment(
    panel: pd.DataFrame,
    regimes: pd.DataFrame,
):

    all_metrics = []
    coefficients = []
    ranking_rows = []

    print(
        "\n[3] EXPERIMENTO POR PERFIL"
    )

    for (
        profile,
        risk_aversion,
    ) in PROFILES.items():

        print(
            f"\nPerfil: {profile}"
        )

        data = panel.copy()

        data[
            "future_utility"
        ] = (
            data[
                "forward_return_7"
            ]
            - risk_aversion
            * data[
                "forward_volatility_7"
            ]
            * np.sqrt(
                FORWARD_HORIZON
            )
        )

        required = list(
            set(
                CONTEXT_FEATURES
                + [
                    "future_utility",
                ]
            )
        )

        data = data.dropna(
            subset=required
        )

        calibration = data[
            data[
                "date"
            ].between(
                CALIBRATION_START,
                CALIBRATION_END,
            )
        ].copy()

        evaluation = data[
            data[
                "date"
            ].between(
                EVALUATION_START,
                EVALUATION_END,
            )
        ].copy()

        print(
            f"  calibración: "
            f"{len(calibration):,}"
        )

        print(
            f"  evaluación: "
            f"{len(evaluation):,}"
        )

        # ----------------------------------------------------
        # Modelos
        # ----------------------------------------------------

        base_model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=RIDGE_ALPHA
                    ),
                ),
            ]
        )

        context_model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=RIDGE_ALPHA
                    ),
                ),
            ]
        )

        base_model.fit(
            calibration[
                BASE_FEATURES
            ],
            calibration[
                "future_utility"
            ],
        )

        context_model.fit(
            calibration[
                CONTEXT_FEATURES
            ],
            calibration[
                "future_utility"
            ],
        )

        evaluation[
            "score_T0"
        ] = base_model.predict(
            evaluation[
                BASE_FEATURES
            ]
        )

        evaluation[
            "score_T1"
        ] = context_model.predict(
            evaluation[
                CONTEXT_FEATURES
            ]
        )

        # ----------------------------------------------------
        # Coeficientes estandarizados
        # ----------------------------------------------------

        for treatment, model, features in [
            (
                "T0",
                base_model,
                BASE_FEATURES,
            ),
            (
                "T1",
                context_model,
                CONTEXT_FEATURES,
            ),
        ]:

            ridge = model.named_steps[
                "ridge"
            ]

            for feature, coefficient in zip(
                features,
                ridge.coef_,
            ):

                coefficients.append(
                    {
                        "profile":
                            profile,

                        "treatment":
                            treatment,

                        "feature":
                            feature,

                        "coefficient":
                            float(
                                coefficient
                            ),
                    }
                )

        # ----------------------------------------------------
        # Evaluación por fecha
        # ----------------------------------------------------

        for date, day in evaluation.groupby(
            "date"
        ):

            day = (
                day
                .sort_values(
                    "asset"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(day) < K:
                continue

            utility = day[
                "future_utility"
            ].to_numpy(
                dtype=float
            )

            relevance = (
                graded_relevance(
                    utility
                )
            )

            for treatment in [
                "T0",
                "T1",
            ]:

                score_column = (
                    "score_T0"
                    if treatment == "T0"
                    else "score_T1"
                )

                scores = day[
                    score_column
                ].to_numpy(
                    dtype=float
                )

                ndcg = ndcg_at_k(
                    relevance,
                    scores,
                    K,
                )

                recall = recall_at_k(
                    utility,
                    scores,
                    K,
                    truth_k=5,
                )

                map_k = map_at_k(
                    utility,
                    scores,
                    K,
                    truth_k=5,
                )

                predicted_order = np.argsort(
                    -scores
                )

                top_assets = (
                    day.iloc[
                        predicted_order[:K]
                    ]["asset"]
                    .tolist()
                )

                all_metrics.append(
                    {
                        "date":
                            date,

                        "profile":
                            profile,

                        "risk_aversion":
                            risk_aversion,

                        "treatment":
                            treatment,

                        "ndcg_at_10":
                            ndcg,

                        "recall_at_10":
                            recall,

                        "map_at_10":
                            map_k,
                    }
                )

                for rank, asset in enumerate(
                    top_assets,
                    start=1,
                ):

                    ranking_rows.append(
                        {
                            "date":
                                date,

                            "profile":
                                profile,

                            "treatment":
                                treatment,

                            "rank":
                                rank,

                            "asset":
                                asset,
                        }
                    )

    metrics = pd.DataFrame(
        all_metrics
    )

    rankings = pd.DataFrame(
        ranking_rows
    )

    coefficients = pd.DataFrame(
        coefficients
    )

    metrics = metrics.merge(
        regimes,
        on="date",
        how="left",
    )

    return (
        metrics,
        rankings,
        coefficients,
    )


# ============================================================
# ESTABILIDAD
# ============================================================

def calculate_stability(
    rankings: pd.DataFrame,
):

    rows = []

    for (
        profile,
        treatment,
    ), group in rankings.groupby(
        [
            "profile",
            "treatment",
        ]
    ):

        dates = sorted(
            group[
                "date"
            ].unique()
        )

        previous = None

        jaccards = []

        for date in dates:

            current = set(
                group[
                    group[
                        "date"
                    ] == date
                ][
                    "asset"
                ]
            )

            if previous is not None:

                intersection = len(
                    current
                    & previous
                )

                union = len(
                    current
                    | previous
                )

                jaccards.append(
                    intersection
                    / union
                )

            previous = current

        rows.append(
            {
                "profile":
                    profile,

                "treatment":
                    treatment,

                "top10_jaccard":
                    float(
                        np.mean(
                            jaccards
                        )
                    ),

                "top10_turnover":
                    float(
                        1.0
                        - np.mean(
                            jaccards
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESÚMENES
# ============================================================

def summarize_results(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
):

    summary = (
        metrics.groupby(
            [
                "profile",
                "treatment",
            ]
        )
        .agg(
            ndcg_at_10=(
                "ndcg_at_10",
                "mean",
            ),

            recall_at_10=(
                "recall_at_10",
                "mean",
            ),

            map_at_10=(
                "map_at_10",
                "mean",
            ),

            n_dates=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
        .merge(
            stability,
            on=[
                "profile",
                "treatment",
            ],
            how="left",
        )
    )

    regime_summary = (
        metrics.groupby(
            [
                "regime",
                "profile",
                "treatment",
            ]
        )
        .agg(
            ndcg_at_10=(
                "ndcg_at_10",
                "mean",
            ),

            recall_at_10=(
                "recall_at_10",
                "mean",
            ),

            map_at_10=(
                "map_at_10",
                "mean",
            ),

            n_dates=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
    )

    return (
        summary,
        regime_summary,
    )


# ============================================================
# PRUEBA ESTADÍSTICA
# ============================================================

def statistical_test(
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

    # Para bootstrap, agrupamos perfiles por fecha
    # para no tratar los tres perfiles del mismo día
    # como observaciones completamente independientes.

    by_date = (
        pivot.reset_index()
        .groupby(
            "date"
        )["difference"]
        .mean()
    )

    (
        delta,
        lower,
        upper,
    ) = paired_bootstrap(
        by_date.to_numpy(),
        iterations=(
            BOOTSTRAP_ITERATIONS
        ),
        seed=RANDOM_SEED,
    )

    nonzero = by_date[
        by_date != 0
    ]

    if len(nonzero) > 0:

        wilcoxon_result = wilcoxon(
            nonzero,
            alternative="two-sided",
        )

        p_value = float(
            wilcoxon_result.pvalue
        )

        statistic = float(
            wilcoxon_result.statistic
        )

    else:

        p_value = 1.0
        statistic = 0.0

    result = pd.DataFrame(
        [
            {
                "metric":
                    "NDCG@10",

                "comparison":
                    "T1 - T0",

                "mean_difference":
                    delta,

                "ci95_lower":
                    lower,

                "ci95_upper":
                    upper,

                "wilcoxon_statistic":
                    statistic,

                "wilcoxon_p":
                    p_value,

                "n_dates":
                    len(
                        by_date
                    ),
            }
        ]
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("EC2 + EC3 - EXPERIMENTO INTEGRADOR DEL RECOMENDADOR")
    print("=" * 80)

    market, sentiment = load_data()

    print(
        f"\nMercado: "
        f"{market['asset'].nunique()} activos"
    )

    print(
        f"Sentimiento: "
        f"{len(sentiment):,} días observados"
    )

    panel = prepare_panel(
        market,
        sentiment,
    )

    (
        regimes,
        regime_lower,
        regime_upper,
    ) = build_regimes(
        panel
    )

    (
        metrics,
        rankings,
        coefficients,
    ) = run_experiment(
        panel,
        regimes,
    )

    stability = calculate_stability(
        rankings
    )

    (
        summary,
        regime_summary,
    ) = summarize_results(
        metrics,
        stability,
    )

    stats = statistical_test(
        metrics
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    panel.to_parquet(
        RESULTS_DIR
        / "recommender_panel.parquet",
        index=False,
    )

    metrics.to_csv(
        RESULTS_DIR
        / "recommender_daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rankings.to_csv(
        RESULTS_DIR
        / "recommender_rankings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficients.to_csv(
        RESULTS_DIR
        / "recommender_model_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        RESULTS_DIR
        / "recommender_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    regime_summary.to_csv(
        RESULTS_DIR
        / "recommender_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stats.to_csv(
        RESULTS_DIR
        / "recommender_statistical_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "primary_metric":
            "NDCG@10",

        "n_assets":
            int(
                market[
                    "asset"
                ].nunique()
            ),

        "k":
            K,

        "forward_horizon_days":
            FORWARD_HORIZON,

        "calibration_start":
            str(
                CALIBRATION_START.date()
            ),

        "calibration_end":
            str(
                CALIBRATION_END.date()
            ),

        "embargo_days":
            7,

        "evaluation_start":
            str(
                EVALUATION_START.date()
            ),

        "evaluation_end":
            str(
                EVALUATION_END.date()
            ),

        "beta_window":
            BETA_WINDOW,

        "beta_min_periods":
            BETA_MIN_PERIODS,

        "ridge_alpha":
            RIDGE_ALPHA,

        "profiles":
            PROFILES,

        "T0_features":
            BASE_FEATURES,

        "T1_features":
            CONTEXT_FEATURES,

        "regime_definition":
            "BTC trailing 14-day momentum; tertile "
            "thresholds estimated on calibration only",

        "regime_lower_threshold":
            regime_lower,

        "regime_upper_threshold":
            regime_upper,
    }

    with open(
        RESULTS_DIR
        / "recommender_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "[4] RESULTADOS GENERALES"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "[5] RESULTADOS POR RÉGIMEN"
    )

    print(
        "=" * 80
    )

    print(
        regime_summary.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "[6] PRUEBA ESTADÍSTICA"
    )

    print(
        "=" * 80
    )

    print(
        stats.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXPERIMENTO COMPLETADO"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()