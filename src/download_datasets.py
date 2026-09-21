from pathlib import Path

import pandas as pd
from datasets import load_dataset


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_dlt_sentiment() -> None:
    """
    Descarga el dataset principal DLT-Sentiment-News desde Hugging Face
    y lo guarda localmente en formato Parquet.
    """
    print("=" * 70)
    print("Downloading DLT-Sentiment-News...")
    print("=" * 70)

    ds = load_dataset(
        "ExponentialScience/DLT-Sentiment-News",
        split="train",
    )

    df = ds.to_pandas()

    output = RAW_DIR / "dlt_sentiment_news.parquet"

    df.to_parquet(
        output,
        index=False,
    )

    print(f"\nSaved: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst rows:")
    print(df.head())


def download_financial_phrasebank() -> None:
    """
    Descarga una versión moderna en Parquet de Financial PhraseBank
    desde Hugging Face y la guarda localmente.

    Se usa el repositorio:
    lmassaron/FinancialPhraseBank

    para evitar el loader antiguo basado en scripts, que ya no está
    soportado por las versiones recientes de datasets.
    """
    print("\n" + "=" * 70)
    print("Downloading Financial PhraseBank...")
    print("=" * 70)

    ds = load_dataset(
        "lmassaron/FinancialPhraseBank"
    )

    print("\nAvailable splits:")
    print(ds)

    frames = []

    for split_name, split_ds in ds.items():
        tmp = split_ds.to_pandas()

        tmp["original_split"] = split_name

        frames.append(tmp)

        print(
            f"Split '{split_name}': "
            f"{len(tmp):,} rows"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    output = RAW_DIR / "financial_phrasebank.parquet"

    df.to_parquet(
        output,
        index=False,
    )

    print(f"\nSaved: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst rows:")
    print(df.head())


def main() -> None:
    print("\nEC3 - DATASET INGESTION")
    print("=" * 70)

    download_dlt_sentiment()

    try:
        download_financial_phrasebank()
    except Exception as exc:
        print("\n" + "!" * 70)
        print("WARNING:")
        print(
            "Financial PhraseBank could not be downloaded."
        )
        print(
            "The main DLT-Sentiment-News dataset "
            "was already saved successfully."
        )
        print("\nError:")
        print(repr(exc))
        print("!" * 70)


if __name__ == "__main__":
    main()