from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TRAIN_PATH = Path("data/processed/train.parquet")
VAL_PATH = Path("data/processed/validation.parquet")
TEST_PATH = Path("data/processed/test.parquet")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


TEXT_COLUMN = "text"
LABEL_COLUMN = "label_id"

LABEL_NAMES = [
    "neutral",
    "bearish",
    "bullish",
]


def evaluate_model(
    model,
    X,
    y,
    split_name: str,
):
    start = time.perf_counter()

    predictions = model.predict(X)

    inference_time = (
        time.perf_counter()
        - start
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision_macro = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2],
    )

    metrics = {
        "split": split_name,
        "accuracy": float(accuracy),
        "precision_macro": float(
            precision_macro
        ),
        "recall_macro": float(
            recall_macro
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "inference_seconds": float(
            inference_time
        ),
        "inference_ms_per_doc": float(
            inference_time
            / len(X)
            * 1000
        ),
    }

    return (
        predictions,
        metrics,
        report,
        matrix,
    )


def main() -> None:
    print("=" * 80)
    print("EC3 - TF-IDF + LinearSVC")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Carga
    # ------------------------------------------------------------
    train = pd.read_parquet(
        TRAIN_PATH
    )

    validation = pd.read_parquet(
        VAL_PATH
    )

    test = pd.read_parquet(
        TEST_PATH
    )

    print("\n[1] DATASETS")

    print(
        f"Train:      {len(train):,}"
    )
    print(
        f"Validation: {len(validation):,}"
    )
    print(
        f"Test:       {len(test):,}"
    )

    X_train = train[
        TEXT_COLUMN
    ].astype(str)

    y_train = train[
        LABEL_COLUMN
    ].astype(int)

    X_validation = validation[
        TEXT_COLUMN
    ].astype(str)

    y_validation = validation[
        LABEL_COLUMN
    ].astype(int)

    X_test = test[
        TEXT_COLUMN
    ].astype(str)

    y_test = test[
        LABEL_COLUMN
    ].astype(int)

    # ------------------------------------------------------------
    # 2. Grid manual
    # ------------------------------------------------------------
    #
    # No usamos GridSearchCV porque el split es temporal.
    # Entrenamos en train y seleccionamos usando validation.
    #
    configurations = []

    config_id = 0

    for ngram_range in [
        (1, 1),
        (1, 2),
    ]:
        for min_df in [
            2,
            5,
        ]:
            for max_features in [
                30000,
                50000,
            ]:
                for C in [
                    0.5,
                    1.0,
                    2.0,
                ]:
                    for class_weight in [
                        None,
                        "balanced",
                    ]:
                        config_id += 1

                        configurations.append(
                            {
                                "config_id": (
                                    config_id
                                ),
                                "ngram_range": (
                                    ngram_range
                                ),
                                "min_df": (
                                    min_df
                                ),
                                "max_features": (
                                    max_features
                                ),
                                "C": C,
                                "class_weight": (
                                    class_weight
                                ),
                            }
                        )

    print(
        f"\n[2] CONFIGURACIONES: "
        f"{len(configurations)}"
    )

    validation_results = []

    best_macro_f1 = -np.inf
    best_model = None
    best_config = None

    # ------------------------------------------------------------
    # 3. Tuning SOLO en validation
    # ------------------------------------------------------------
    print(
        "\n[3] TUNING SOBRE VALIDATION 2024"
    )

    for config in configurations:
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(
                            config[
                                "ngram_range"
                            ]
                        ),
                        min_df=(
                            config[
                                "min_df"
                            ]
                        ),
                        max_features=(
                            config[
                                "max_features"
                            ]
                        ),
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "svm",
                    LinearSVC(
                        C=config["C"],
                        class_weight=(
                            config[
                                "class_weight"
                            ]
                        ),
                        random_state=42,
                    ),
                ),
            ]
        )

        train_start = (
            time.perf_counter()
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        train_seconds = (
            time.perf_counter()
            - train_start
        )

        (
            val_predictions,
            val_metrics,
            _,
            _,
        ) = evaluate_model(
            pipeline,
            X_validation,
            y_validation,
            "validation",
        )

        result = {
            **config,
            "train_seconds": float(
                train_seconds
            ),
            **val_metrics,
        }

        validation_results.append(
            result
        )

        print(
            f"Config "
            f"{config['config_id']:02d} | "
            f"ngrams="
            f"{config['ngram_range']} | "
            f"min_df="
            f"{config['min_df']} | "
            f"max_features="
            f"{config['max_features']} | "
            f"C={config['C']} | "
            f"class_weight="
            f"{config['class_weight']} | "
            f"Macro-F1="
            f"{val_metrics['macro_f1']:.4f}"
        )

        if (
            val_metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):
            best_macro_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            best_model = pipeline
            best_config = config

    # ------------------------------------------------------------
    # 4. Guardar tuning
    # ------------------------------------------------------------
    tuning_df = pd.DataFrame(
        validation_results
    )

    tuning_df = (
        tuning_df
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    tuning_df.to_csv(
        RESULTS_DIR
        / "tfidf_svm_validation_tuning.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[4] MEJOR CONFIGURACIÓN"
    )

    print(
        json.dumps(
            {
                **best_config,
                "validation_macro_f1": (
                    best_macro_f1
                ),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    # ------------------------------------------------------------
    # 5. Guardar modelo seleccionado
    # ------------------------------------------------------------
    model_path = (
        MODELS_DIR
        / "tfidf_linearsvc_best.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        f"\nModelo guardado: "
        f"{model_path}"
    )

    # ------------------------------------------------------------
    # 6. Evaluación final en test congelado
    # ------------------------------------------------------------
    print(
        "\n[5] EVALUACIÓN FINAL "
        "SOBRE TEST 2025"
    )

    (
        test_predictions,
        test_metrics,
        test_report,
        test_matrix,
    ) = evaluate_model(
        best_model,
        X_test,
        y_test,
        "test",
    )

    print(
        "\nMétricas:"
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    print(
        "\nClassification report:"
    )

    report_df = (
        pd.DataFrame(
            test_report
        )
        .transpose()
    )

    print(
        report_df
    )

    report_df.to_csv(
        RESULTS_DIR
        / "tfidf_svm_test_classification_report.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 7. Matriz de confusión
    # ------------------------------------------------------------
    matrix_df = pd.DataFrame(
        test_matrix,
        index=[
            f"true_{name}"
            for name in LABEL_NAMES
        ],
        columns=[
            f"pred_{name}"
            for name in LABEL_NAMES
        ],
    )

    print(
        "\nMatriz de confusión:"
    )

    print(
        matrix_df
    )

    matrix_df.to_csv(
        RESULTS_DIR
        / "tfidf_svm_test_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 8. Predicciones
    # ------------------------------------------------------------
    predictions_df = test[
        [
            "timestamp",
            "title",
            "text",
            "label_id",
            "label_name",
        ]
    ].copy()

    predictions_df[
        "predicted_label_id"
    ] = test_predictions

    predictions_df[
        "predicted_label_name"
    ] = (
        predictions_df[
            "predicted_label_id"
        ]
        .map(
            {
                0: "neutral",
                1: "bearish",
                2: "bullish",
            }
        )
    )

    predictions_df[
        "correct"
    ] = (
        predictions_df[
            "label_id"
        ]
        ==
        predictions_df[
            "predicted_label_id"
        ]
    )

    predictions_df.to_csv(
        RESULTS_DIR
        / "tfidf_svm_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 9. Métricas generales
    # ------------------------------------------------------------
    metrics_df = pd.DataFrame(
        [
            {
                "model": (
                    "TF-IDF + LinearSVC"
                ),
                **test_metrics,
            }
        ]
    )

    metrics_df.to_csv(
        RESULTS_DIR
        / "pln_metrics_tfidf_svm.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # 10. Metadata experimental
    # ------------------------------------------------------------
    metadata = {
        "model": (
            "TF-IDF + LinearSVC"
        ),
        "selection_metric": (
            "validation_macro_f1"
        ),
        "random_state": 42,
        "train_period": (
            "2021-01-01 to "
            "2023-12-31"
        ),
        "validation_period": (
            "2024-01-01 to "
            "2024-12-31"
        ),
        "test_period": (
            "2025-01-01 to "
            "2025-05-23"
        ),
        "best_config": {
            **best_config,
            "ngram_range": list(
                best_config[
                    "ngram_range"
                ]
            ),
        },
        "validation_macro_f1": float(
            best_macro_f1
        ),
        "test_metrics": (
            test_metrics
        ),
    }

    with open(
        RESULTS_DIR
        / "tfidf_svm_metadata.json",
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
        "\n" + "=" * 80
    )

    print(
        "TF-IDF + LinearSVC COMPLETADO"
    )

    print(
        "=" * 80
    )

    print(
        "\nArchivos principales:"
    )

    for filename in [
        (
            "results/"
            "tfidf_svm_validation_tuning.csv"
        ),
        (
            "results/"
            "tfidf_svm_test_classification_report.csv"
        ),
        (
            "results/"
            "tfidf_svm_test_confusion_matrix.csv"
        ),
        (
            "results/"
            "tfidf_svm_test_predictions.csv"
        ),
        (
            "results/"
            "pln_metrics_tfidf_svm.csv"
        ),
        (
            "results/"
            "tfidf_svm_metadata.json"
        ),
        (
            "models/"
            "tfidf_linearsvc_best.joblib"
        ),
    ]:
        print(
            f"  - {filename}"
        )


if __name__ == "__main__":
    main()