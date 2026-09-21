from pathlib import Path
import json
import time
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path("data/processed/train.parquet")
VAL_PATH = Path("data/processed/validation.parquet")
TEST_PATH = Path("data/processed/test.parquet")

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label_id"

LABEL_NAMES = [
    "neutral",
    "bearish",
    "bullish",
]

ID2LABEL = {
    0: "neutral",
    1: "bearish",
    2: "bullish",
}

LABEL2ID = {
    value: key
    for key, value in ID2LABEL.items()
}

SEED = 42

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 4

WARMUP_STEPS = 466

EARLY_STOPPING_PATIENCE = 1


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision_macro = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("EC3 - DISTILBERT")
    print("=" * 80)

    set_seed(SEED)

    # --------------------------------------------------------
    # 1. Hardware
    # --------------------------------------------------------

    print("\n[1] HARDWARE")

    print(
        f"Torch: {torch.__version__}"
    )

    print(
        f"CUDA disponible: "
        f"{torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. "
            "Este experimento está configurado "
            "para ejecutarse en GPU."
        )

    device_name = torch.cuda.get_device_name(0)

    total_memory_gb = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"GPU: {device_name}"
    )

    print(
        f"VRAM total: "
        f"{total_memory_gb:.2f} GB"
    )

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------

    print("\n[2] DATASETS")

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    val_df = pd.read_parquet(
        VAL_PATH
    )

    test_df = pd.read_parquet(
        TEST_PATH
    )

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    print(
        f"Test:       {len(test_df):,}"
    )

    # --------------------------------------------------------
    # 3. Hugging Face datasets
    # --------------------------------------------------------

    train_hf = Dataset.from_pandas(
        train_df[
            [
                TEXT_COLUMN,
                LABEL_COLUMN,
            ]
        ].rename(
            columns={
                LABEL_COLUMN: "labels",
            }
        ),
        preserve_index=False,
    )

    val_hf = Dataset.from_pandas(
        val_df[
            [
                TEXT_COLUMN,
                LABEL_COLUMN,
            ]
        ].rename(
            columns={
                LABEL_COLUMN: "labels",
            }
        ),
        preserve_index=False,
    )

    test_hf = Dataset.from_pandas(
        test_df[
            [
                TEXT_COLUMN,
                LABEL_COLUMN,
            ]
        ].rename(
            columns={
                LABEL_COLUMN: "labels",
            }
        ),
        preserve_index=False,
    )

    # --------------------------------------------------------
    # 4. Tokenizer
    # --------------------------------------------------------

    print("\n[3] TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_hf = train_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            TEXT_COLUMN,
        ],
    )

    val_hf = val_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            TEXT_COLUMN,
        ],
    )

    test_hf = test_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=[
            TEXT_COLUMN,
        ],
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
        )
    )

    # --------------------------------------------------------
    # 5. Model
    # --------------------------------------------------------

    print("\n[4] MODELO")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    # --------------------------------------------------------
    # 6. Training arguments
    # --------------------------------------------------------

    output_dir = (
        MODELS_DIR
        / "distilbert_checkpoints"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,

        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        num_train_epochs=NUM_EPOCHS,

        weight_decay=WEIGHT_DECAY,

        warmup_steps=WARMUP_STEPS,

        lr_scheduler_type="linear",

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        fp16=True,

        logging_strategy="steps",

        logging_steps=100,

        save_total_limit=2,

        report_to="none",

        seed=SEED,

        data_seed=SEED,

        dataloader_num_workers=0,

        optim="adamw_torch",

        gradient_checkpointing=True,
    )

    # --------------------------------------------------------
    # 7. Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_hf,

        eval_dataset=val_hf,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    EARLY_STOPPING_PATIENCE
                )
            )
        ],
    )

    # --------------------------------------------------------
    # 8. Train
    # --------------------------------------------------------

    print("\n[5] ENTRENAMIENTO")

    torch.cuda.empty_cache()

    start_train = (
        time.perf_counter()
    )

    train_result = trainer.train()

    train_seconds = (
        time.perf_counter()
        - start_train
    )

    print(
        f"\nTiempo total entrenamiento: "
        f"{train_seconds:.2f} s"
    )

    print(
        f"Tiempo total entrenamiento: "
        f"{train_seconds / 60:.2f} min"
    )

    # --------------------------------------------------------
    # 9. Validation result of selected checkpoint
    # --------------------------------------------------------

    print(
        "\n[6] EVALUACIÓN "
        "DEL MEJOR CHECKPOINT EN VALIDATION"
    )

    validation_metrics = (
        trainer.evaluate(
            eval_dataset=val_hf
        )
    )

    print(
        json.dumps(
            validation_metrics,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 10. Frozen test
    # --------------------------------------------------------

    print(
        "\n[7] EVALUACIÓN FINAL "
        "SOBRE TEST 2025"
    )

    test_start = (
        time.perf_counter()
    )

    test_output = trainer.predict(
        test_hf
    )

    test_seconds = (
        time.perf_counter()
        - test_start
    )

    logits = (
        test_output.predictions
    )

    true_labels = (
        test_output.label_ids
    )

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    # --------------------------------------------------------
    # 11. Test metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    precision_macro = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    test_metrics = {
        "model": (
            "DistilBERT"
        ),
        "split": "test",
        "accuracy": float(
            accuracy
        ),
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
        "training_seconds": float(
            train_seconds
        ),
        "inference_seconds": float(
            test_seconds
        ),
        "inference_ms_per_doc": float(
            test_seconds
            / len(test_df)
            * 1000
        ),
    }

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 12. Classification report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(
            report
        )
        .transpose()
    )

    print(
        "\nClassification report:"
    )

    print(
        report_df
    )

    report_df.to_csv(
        RESULTS_DIR
        / (
            "distilbert_test_"
            "classification_report.csv"
        ),
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 13. Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[
            0,
            1,
            2,
        ],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "true_neutral",
            "true_bearish",
            "true_bullish",
        ],
        columns=[
            "pred_neutral",
            "pred_bearish",
            "pred_bullish",
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
        / (
            "distilbert_test_"
            "confusion_matrix.csv"
        ),
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 14. Probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    predictions_df = test_df[
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
    ] = predictions

    predictions_df[
        "predicted_label_name"
    ] = (
        predictions_df[
            "predicted_label_id"
        ]
        .map(ID2LABEL)
    )

    predictions_df[
        "prob_neutral"
    ] = probabilities[:, 0]

    predictions_df[
        "prob_bearish"
    ] = probabilities[:, 1]

    predictions_df[
        "prob_bullish"
    ] = probabilities[:, 2]

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
        / (
            "distilbert_test_"
            "predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 15. Main metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            test_metrics
        ]
    )

    metrics_df.to_csv(
        RESULTS_DIR
        / "pln_metrics_distilbert.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 16. Save best model
    # --------------------------------------------------------

    final_model_dir = (
        MODELS_DIR
        / "distilbert_best"
    )

    trainer.save_model(
        str(
            final_model_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            final_model_dir
        )
    )

    # --------------------------------------------------------
    # 17. Metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": MODEL_NAME,

        "seed": SEED,

        "max_length": MAX_LENGTH,

        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),

        "eval_batch_size": (
            EVAL_BATCH_SIZE
        ),

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),

        "effective_train_batch_size": (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "epochs_max": NUM_EPOCHS,

        "warmup_steps": WARMUP_STEPS,

        "fp16": True,

        "gradient_checkpointing": True,

        "selection_metric": (
            "validation_macro_f1"
        ),

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

        "gpu": device_name,

        "vram_gb": float(
            total_memory_gb
        ),

        "training_seconds": float(
            train_seconds
        ),

        "validation_metrics": (
            validation_metrics
        ),

        "test_metrics": (
            test_metrics
        ),
    }

    with open(
        RESULTS_DIR
        / "distilbert_metadata.json",
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
        "DISTILBERT COMPLETADO"
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
            "pln_metrics_distilbert.csv"
        ),
        (
            "results/"
            "distilbert_test_"
            "predictions.csv"
        ),
        (
            "results/"
            "distilbert_test_"
            "classification_report.csv"
        ),
        (
            "results/"
            "distilbert_test_"
            "confusion_matrix.csv"
        ),
        (
            "results/"
            "distilbert_metadata.json"
        ),
        (
            "models/"
            "distilbert_best/"
        ),
    ]:
        print(
            f"  - {filename}"
        )


if __name__ == "__main__":
    main()