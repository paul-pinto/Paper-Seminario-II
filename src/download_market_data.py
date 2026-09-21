from pathlib import Path
import json
import time

import pandas as pd
import yfinance as yf


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Universo candidato
# ============================================================

ASSETS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
    "DOT": "DOT-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
    "XLM": "XLM-USD",
    "TRX": "TRX-USD",
    "ETC": "ETC-USD",
}


# Descargamos algunos días adicionales para:
# - construir features rezagadas;
# - calcular retornos forward después del 23/05.
START_DATE = "2024-11-01"
END_DATE = "2025-06-15"


def download_asset(
    asset: str,
    ticker: str,
) -> pd.DataFrame:

    print(
        f"Descargando {asset:5s} "
        f"({ticker})..."
    )

    data = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        print(
            f"  WARNING: sin datos para {asset}"
        )
        return pd.DataFrame()

    # yfinance puede devolver MultiIndex incluso
    # al solicitar un único ticker.
    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = (
        data.reset_index()
        .rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
    )

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
        errors="coerce",
    ).dt.tz_localize(None)

    data["asset"] = asset
    data["ticker"] = ticker

    keep = [
        "date",
        "asset",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    existing = [
        column
        for column in keep
        if column in data.columns
    ]

    data = data[
        existing
    ].copy()

    data = data.dropna(
        subset=[
            "date",
            "close",
        ]
    )

    data = data.sort_values(
        "date"
    )

    print(
        f"  filas={len(data):,} | "
        f"{data['date'].min().date()} "
        f"-> "
        f"{data['date'].max().date()}"
    )

    return data


def main() -> None:

    print("=" * 80)
    print("EC2 - DESCARGA Y AUDITORÍA DE MERCADO")
    print("=" * 80)

    frames = []

    failures = []

    for asset, ticker in ASSETS.items():

        try:
            df = download_asset(
                asset,
                ticker,
            )

            if df.empty:
                failures.append(asset)
            else:
                frames.append(df)

        except Exception as exc:
            print(
                f"  ERROR {asset}: {exc}"
            )
            failures.append(asset)

        time.sleep(0.5)

    if not frames:
        raise RuntimeError(
            "No se descargó ningún activo."
        )

    market = pd.concat(
        frames,
        ignore_index=True,
    )

    market = market.sort_values(
        [
            "date",
            "asset",
        ]
    ).reset_index(
        drop=True
    )

    raw_output = (
        RAW_DIR
        / "crypto_market_2025.parquet"
    )

    market.to_parquet(
        raw_output,
        index=False,
    )

    print("\n" + "=" * 80)
    print("[1] COBERTURA POR ACTIVO")
    print("=" * 80)

    coverage = (
        market.groupby("asset")
        .agg(
            observations=(
                "date",
                "size",
            ),
            date_min=(
                "date",
                "min",
            ),
            date_max=(
                "date",
                "max",
            ),
            close_missing=(
                "close",
                lambda x: int(
                    x.isna().sum()
                ),
            ),
        )
        .reset_index()
    )

    coverage[
        "date_min"
    ] = (
        coverage["date_min"]
        .dt.date
    )

    coverage[
        "date_max"
    ] = (
        coverage["date_max"]
        .dt.date
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    coverage.to_csv(
        RESULTS_DIR
        / "market_asset_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Cobertura del periodo experimental
    # ========================================================

    experiment_start = pd.Timestamp(
        "2025-01-01"
    )

    experiment_end = pd.Timestamp(
        "2025-05-23"
    )

    experiment = market[
        market["date"].between(
            experiment_start,
            experiment_end,
        )
    ].copy()

    pivot = experiment.pivot_table(
        index="date",
        columns="asset",
        values="close",
        aggfunc="last",
    )

    expected_days = pd.date_range(
        experiment_start,
        experiment_end,
        freq="D",
    )

    pivot = pivot.reindex(
        expected_days
    )

    asset_coverage = (
        pivot.notna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 80)
    print("[2] COBERTURA EN EL PERIODO EXPERIMENTAL")
    print("=" * 80)

    print(
        asset_coverage
        .round(2)
        .to_string()
    )

    asset_coverage.rename(
        "coverage_pct"
    ).to_csv(
        RESULTS_DIR
        / "market_experiment_coverage.csv",
        encoding="utf-8-sig",
    )

    # ========================================================
    # Selección automática
    # ========================================================

    # Exigimos 95% o más de cobertura.
    selected_assets = (
        asset_coverage[
            asset_coverage >= 95.0
        ]
        .index
        .tolist()
    )

    print("\n[3] ACTIVOS SELECCIONADOS")

    for asset in selected_assets:
        print(
            f"  - {asset}: "
            f"{asset_coverage[asset]:.2f}%"
        )

    print(
        f"\nTotal seleccionados: "
        f"{len(selected_assets)}"
    )

    if len(selected_assets) < 10:
        print(
            "\nWARNING: "
            "quedaron menos de 10 activos. "
            "No congelaremos todavía NDCG@10."
        )

    selected_market = market[
        market["asset"].isin(
            selected_assets
        )
    ].copy()

    selected_output = (
        PROCESSED_DIR
        / "crypto_market_selected.parquet"
    )

    selected_market.to_parquet(
        selected_output,
        index=False,
    )

    metadata = {
        "source": "Yahoo Finance via yfinance",
        "download_start": START_DATE,
        "download_end_exclusive": END_DATE,

        "experiment_start":
            "2025-01-01",

        "experiment_end":
            "2025-05-23",

        "candidate_assets":
            list(ASSETS.keys()),

        "selected_assets":
            selected_assets,

        "selection_rule":
            ">=95% close-price coverage",

        "failed_assets":
            failures,
    }

    with open(
        RESULTS_DIR
        / "market_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[4] ARCHIVOS GENERADOS")

    for path in [
        raw_output,
        selected_output,
        RESULTS_DIR
        / "market_asset_coverage.csv",
        RESULTS_DIR
        / "market_experiment_coverage.csv",
        RESULTS_DIR
        / "market_metadata.json",
    ]:
        print(
            f"  - {path}"
        )


if __name__ == "__main__":
    main()
    