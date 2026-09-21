from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido: {path}"
        )


# ============================================================
# 1. TABLA PLN
# ============================================================

def build_pln_table():
    path = RESULTS_DIR / "pln_model_comparison.csv"
    require(path)

    df = pd.read_csv(path)

    df = df.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    df.to_csv(
        TABLES_DIR / "table_pln_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ============================================================
# 2. FIGURA PLN
# ============================================================

def plot_pln_comparison(df):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        df["model"],
        df["macro_f1"],
    )

    ax.set_ylabel("Macro-F1")
    ax.set_title(
        "Comparación de desempeño de modelos PLN"
    )

    ax.set_ylim(
        0,
        max(df["macro_f1"]) * 1.18
    )

    for i, value in enumerate(
        df["macro_f1"]
    ):
        ax.text(
            i,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
        )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "fig_pln_macro_f1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 3. TABLA ESTADÍSTICA PLN
# ============================================================

def build_pln_statistics_table():
    path = RESULTS_DIR / "pln_pairwise_statistics.csv"
    require(path)

    df = pd.read_csv(path)

    df.to_csv(
        TABLES_DIR / "table_pln_pairwise_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 4. SEÑAL DE SENTIMIENTO
# ============================================================

def plot_sentiment_signal():
    path = RESULTS_DIR / "contextual_sentiment_daily.csv"
    require(path)

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["date"],
        df["sentiment_mean"],
        linewidth=1.5,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Señal diaria de sentimiento financiero"
    )

    ax.set_xlabel("Fecha")
    ax.set_ylabel(
        "P(bullish) - P(bearish)"
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig_daily_sentiment_signal.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 5. TABLA PRINCIPAL DEL RECOMENDADOR
# ============================================================

def build_recommender_summary():
    path = RESULTS_DIR / "recommender_summary.csv"
    require(path)

    df = pd.read_csv(path)

    df.to_csv(
        TABLES_DIR / "table_recommender_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ============================================================
# 6. FIGURA NDCG T0 VS T1
# ============================================================

def plot_ndcg_by_profile(df):
    pivot = df.pivot(
        index="profile",
        columns="treatment",
        values="ndcg_at_10",
    )

    preferred_order = [
        "conservador",
        "moderado",
        "agresivo",
    ]

    pivot = pivot.reindex(
        [
            x
            for x in preferred_order
            if x in pivot.index
        ]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        "NDCG@10 por perfil y tratamiento"
    )

    ax.set_ylabel("NDCG@10")
    ax.set_xlabel("Perfil")

    ax.legend(
        title="Tratamiento"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "fig_ndcg_by_profile.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 7. FIGURA TURNOVER
# ============================================================

def plot_turnover(df):
    pivot = df.pivot(
        index="profile",
        columns="treatment",
        values="top10_turnover",
    )

    preferred_order = [
        "conservador",
        "moderado",
        "agresivo",
    ]

    pivot = pivot.reindex(
        [
            x
            for x in preferred_order
            if x in pivot.index
        ]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title(
        "Turnover del Top-10 por perfil"
    )

    ax.set_ylabel(
        "Top-10 turnover"
    )

    ax.set_xlabel("Perfil")

    ax.legend(
        title="Tratamiento"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "fig_turnover_by_profile.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 8. TABLA POR RÉGIMEN
# ============================================================

def build_regime_table():
    path = RESULTS_DIR / "recommender_regime_summary.csv"
    require(path)

    df = pd.read_csv(path)

    rename_map = {
        "bear": "lower-momentum",
        "sideways": "middle-momentum",
        "bull": "upper-momentum",
    }

    df["regime_label"] = (
        df["regime"]
        .map(rename_map)
        .fillna(df["regime"])
    )

    df.to_csv(
        TABLES_DIR / "table_recommender_by_regime.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


# ============================================================
# 9. FIGURA DELTA POR RÉGIMEN
# ============================================================

def plot_delta_by_regime():
    path = RESULTS_DIR / "robustness_by_regime.csv"
    require(path)

    df = pd.read_csv(path)

    rename_map = {
        "bear": "lower-momentum",
        "sideways": "middle-momentum",
        "bull": "upper-momentum",
    }

    df["regime_label"] = (
        df["regime"]
        .map(rename_map)
        .fillna(df["regime"])
    )

    pivot = df.pivot(
        index="regime_label",
        columns="profile",
        values="mean_delta_ndcg",
    )

    preferred_order = [
        "lower-momentum",
        "middle-momentum",
        "upper-momentum",
    ]

    pivot = pivot.reindex(
        [
            x
            for x in preferred_order
            if x in pivot.index
        ]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 5),
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Cambio medio en NDCG@10 por régimen"
    )

    ax.set_ylabel(
        "Δ NDCG@10 (T1 - T0)"
    )

    ax.set_xlabel(
        "Régimen de momentum"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.legend(
        title="Perfil"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "fig_delta_ndcg_by_regime.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 10. DELTA DIARIO
# ============================================================

def plot_daily_delta():
    path = RESULTS_DIR / "daily_delta_ndcg.csv"
    require(path)

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["date"],
        df["mean_delta_ndcg"],
        linewidth=1.4,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Diferencia diaria de NDCG@10"
    )

    ax.set_xlabel("Fecha")
    ax.set_ylabel(
        "Δ NDCG@10 (T1 - T0)"
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig_daily_delta_ndcg.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 11. ROBUSTEZ POR PERFIL
# ============================================================

def build_robustness_table():
    path = RESULTS_DIR / "robustness_by_profile.csv"
    require(path)

    df = pd.read_csv(path)

    df.to_csv(
        TABLES_DIR
        / "table_robustness_by_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 12. COEFICIENTE DE SENTIMIENTO
# ============================================================

def plot_sentiment_coefficients():
    path = RESULTS_DIR / "sentiment_exposure_coefficients.csv"
    require(path)

    df = pd.read_csv(path)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        df["profile"],
        df["coefficient"],
    )

    ax.set_title(
        "Coeficiente estandarizado de la exposición al sentimiento"
    )

    ax.set_xlabel("Perfil")
    ax.set_ylabel(
        "Coeficiente Ridge"
    )

    for i, value in enumerate(
        df["coefficient"]
    ):
        ax.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig_sentiment_exposure_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 13. TABLA METODOLÓGICA RESUMIDA
# ============================================================

def build_methodological_summary():
    rows = [
        {
            "component": "PLN split",
            "definition": (
                "Train 2021-2023; validation 2024; test 2025"
            ),
        },
        {
            "component": "PLN primary metric",
            "definition": "Macro-F1",
        },
        {
            "component": "Selected NLP model",
            "definition": "DistilBERT",
        },
        {
            "component": "Sentiment signal",
            "definition": (
                "P(bullish) - P(bearish)"
            ),
        },
        {
            "component": "Asset universe",
            "definition": "15 cryptoassets",
        },
        {
            "component": "Recommendation calibration",
            "definition": (
                "2025-02-01 to 2025-03-08"
            ),
        },
        {
            "component": "Embargo",
            "definition": "7 days",
        },
        {
            "component": "Recommendation evaluation",
            "definition": (
                "2025-03-16 to 2025-05-16"
            ),
        },
        {
            "component": "Primary ranking metric",
            "definition": "NDCG@10",
        },
        {
            "component": "Treatments",
            "definition": (
                "T0 quantitative; T1 quantitative + sentiment exposure"
            ),
        },
    ]

    df = pd.DataFrame(rows)

    df.to_csv(
        TABLES_DIR
        / "table_methodological_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("GENERACIÓN DE FIGURAS Y TABLAS FINALES")
    print("=" * 80)

    print("\n[1] PLN")
    pln = build_pln_table()
    plot_pln_comparison(pln)
    build_pln_statistics_table()

    print("[2] Sentimiento")
    plot_sentiment_signal()

    print("[3] Recomendador")
    recommender = build_recommender_summary()
    plot_ndcg_by_profile(recommender)
    plot_turnover(recommender)

    print("[4] Regímenes")
    build_regime_table()
    plot_delta_by_regime()

    print("[5] Robustez")
    plot_daily_delta()
    build_robustness_table()

    print("[6] Coeficientes")
    plot_sentiment_coefficients()

    print("[7] Metodología")
    build_methodological_summary()

    print("\nFIGURAS GENERADAS:")

    for path in sorted(
        FIGURES_DIR.glob("*.png")
    ):
        print(
            f"  - {path}"
        )

    print("\nTABLAS GENERADAS:")

    for path in sorted(
        TABLES_DIR.glob("*.csv")
    ):
        print(
            f"  - {path}"
        )

    print("\n" + "=" * 80)
    print("GENERACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()