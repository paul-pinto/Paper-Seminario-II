from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import f1_score


RESULTS_DIR = Path("results")

FILES = {
    "TF-IDF + LinearSVC":
        RESULTS_DIR / "tfidf_svm_test_predictions.csv",

    "DistilBERT":
        RESULTS_DIR / "distilbert_test_predictions.csv",

    "FinBERT":
        RESULTS_DIR / "finbert_test_predictions.csv",
}

SEED = 42
N_BOOTSTRAP = 5000


def load_predictions():
    models = {}

    for model_name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No existe: {path}"
            )

        df = pd.read_csv(path)

        required = {
            "label_id",
            "predicted_label_id",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{model_name}: faltan columnas "
                f"{missing}"
            )

        models[model_name] = df

    return models


def bootstrap_macro_f1_difference(
    y_true,
    pred_a,
    pred_b,
    n_bootstrap=5000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n = len(y_true)

    observed_a = f1_score(
        y_true,
        pred_a,
        average="macro",
    )

    observed_b = f1_score(
        y_true,
        pred_b,
        average="macro",
    )

    observed_difference = (
        observed_a - observed_b
    )

    differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample_true = y_true[indices]

        sample_a = pred_a[indices]
        sample_b = pred_b[indices]

        score_a = f1_score(
            sample_true,
            sample_a,
            average="macro",
            zero_division=0,
        )

        score_b = f1_score(
            sample_true,
            sample_b,
            average="macro",
            zero_division=0,
        )

        differences[i] = (
            score_a - score_b
        )

    lower = np.percentile(
        differences,
        2.5,
    )

    upper = np.percentile(
        differences,
        97.5,
    )

    return {
        "macro_f1_a": observed_a,
        "macro_f1_b": observed_b,
        "difference": observed_difference,
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def mcnemar_exact(
    y_true,
    pred_a,
    pred_b,
):
    correct_a = (
        pred_a == y_true
    )

    correct_b = (
        pred_b == y_true
    )

    a_correct_b_wrong = int(
        np.sum(
            correct_a
            & ~correct_b
        )
    )

    a_wrong_b_correct = int(
        np.sum(
            ~correct_a
            & correct_b
        )
    )

    discordant = (
        a_correct_b_wrong
        + a_wrong_b_correct
    )

    if discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(
            a_correct_b_wrong,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = result.pvalue

    return {
        "a_correct_b_wrong":
            a_correct_b_wrong,

        "a_wrong_b_correct":
            a_wrong_b_correct,

        "discordant":
            discordant,

        "mcnemar_exact_p":
            p_value,
    }


def main():
    print("=" * 80)
    print("EC3 - COMPARACIÓN ESTADÍSTICA PLN")
    print("=" * 80)

    models = load_predictions()

    reference = next(
        iter(models.values())
    )

    y_true = (
        reference["label_id"]
        .to_numpy()
        .astype(int)
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    for model_name, df in models.items():
        current_true = (
            df["label_id"]
            .to_numpy()
            .astype(int)
        )

        if not np.array_equal(
            y_true,
            current_true,
        ):
            raise ValueError(
                "Los archivos no tienen "
                "las mismas observaciones "
                f"o el mismo orden: {model_name}"
            )

    # --------------------------------------------------------
    # Global scores
    # --------------------------------------------------------

    global_results = []

    predictions = {}

    for model_name, df in models.items():
        pred = (
            df["predicted_label_id"]
            .to_numpy()
            .astype(int)
        )

        predictions[
            model_name
        ] = pred

        macro_f1 = f1_score(
            y_true,
            pred,
            average="macro",
        )

        accuracy = np.mean(
            pred == y_true
        )

        global_results.append(
            {
                "model": model_name,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
            }
        )

    global_df = pd.DataFrame(
        global_results
    ).sort_values(
        "macro_f1",
        ascending=False,
    )

    print("\n[1] RESULTADOS GLOBALES")
    print(
        global_df.to_string(
            index=False
        )
    )

    global_df.to_csv(
        RESULTS_DIR
        / "pln_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------------

    names = list(
        predictions.keys()
    )

    pairwise_results = []

    for i in range(len(names)):
        for j in range(
            i + 1,
            len(names),
        ):
            name_a = names[i]
            name_b = names[j]

            pred_a = predictions[
                name_a
            ]

            pred_b = predictions[
                name_b
            ]

            bootstrap = (
                bootstrap_macro_f1_difference(
                    y_true,
                    pred_a,
                    pred_b,
                    n_bootstrap=N_BOOTSTRAP,
                    seed=SEED,
                )
            )

            mcnemar = mcnemar_exact(
                y_true,
                pred_a,
                pred_b,
            )

            significant_bootstrap = (
                bootstrap[
                    "ci95_lower"
                ] > 0
                or
                bootstrap[
                    "ci95_upper"
                ] < 0
            )

            significant_mcnemar = (
                mcnemar[
                    "mcnemar_exact_p"
                ] < 0.05
            )

            result = {
                "model_a": name_a,
                "model_b": name_b,

                **bootstrap,
                **mcnemar,

                "bootstrap_significant_95":
                    significant_bootstrap,

                "mcnemar_significant_005":
                    significant_mcnemar,
            }

            pairwise_results.append(
                result
            )

    pairwise_df = pd.DataFrame(
        pairwise_results
    )

    print(
        "\n[2] COMPARACIONES PAREADAS"
    )

    display_columns = [
        "model_a",
        "model_b",
        "difference",
        "ci95_lower",
        "ci95_upper",
        "mcnemar_exact_p",
    ]

    print(
        pairwise_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    pairwise_df.to_csv(
        RESULTS_DIR
        / "pln_pairwise_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nArchivos generados:"
    )

    print(
        "  - results/"
        "pln_model_comparison.csv"
    )

    print(
        "  - results/"
        "pln_pairwise_statistics.csv"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Un IC95% del delta Macro-F1 "
        "que no cruza cero aporta "
        "evidencia de diferencia."
    )

    print(
        "McNemar p < 0.05 aporta "
        "evidencia de diferencia en "
        "el patrón de aciertos/errores."
    )


if __name__ == "__main__":
    main()