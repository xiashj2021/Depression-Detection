import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# =========================
# Optional environment config
# =========================
# tf.config.set_visible_devices([], "GPU")
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# =========================
# Global config
# =========================
DATA_PATH = "./Dataset.xlsx"

SUBJECT_COL = "id"
ANSWER_COL = "text"
LABEL_COL = "class"

# Prefer seq as the question index if available (1-9).
# If seq does not exist, the script will fall back to question.
QUESTION_ID_COL_CANDIDATES = ["seq", "question"]
QUESTION_TEXT_COL = "question"

MAX_LEN = 150
N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]

SAVE_RESULTS = True
DETAILS_CSV = "single_question_repeated_cv_details.csv"
SUMMARY_CSV = "single_question_repeated_cv_summary.csv"
REPORT_TXT = "single_question_results.txt"

# If QUESTION_TEXT_COL stores only codes like Q1/Q2,
# fill this dictionary. Otherwise keep it empty.
QUESTION_TEXT_MAP = {
    # "Q1": "完整题目文本1",
    # "Q2": "完整题目文本2",
}

# =========================
# Reproducibility
# =========================
BASE_SEED = 42
tf.random.set_seed(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

# =========================
# Helper functions
# =========================
def normalize_question_text(q):
    q = str(q)
    return QUESTION_TEXT_MAP.get(q, q)

def choose_question_id_col(df):
    for col in QUESTION_ID_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Dataset is missing a question identifier column. "
        f"Tried: {QUESTION_ID_COL_CANDIDATES}"
    )

def extract_sentence_embedding(text, tokenizer, model, max_length=100):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        max_length=max_length,
        padding=True,
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    return acc, pre, rec, f1, auc

def get_valid_n_splits(y, requested_splits):
    class_counts = pd.Series(y).value_counts()
    min_class_count = class_counts.min()
    return int(min(requested_splits, min_class_count))

def sort_question_values(values):
    values = list(values)
    try:
        return sorted(values, key=lambda x: float(x))
    except (TypeError, ValueError):
        return sorted(values, key=lambda x: str(x))

def pretty_pm(mean_val, std_val):
    if pd.isna(mean_val):
        return "nan"
    if pd.isna(std_val):
        return f"{mean_val:.4f} ± nan"
    return f"{mean_val:.4f} ± {std_val:.4f}"

# =========================
# Load data
# =========================
data = pd.read_excel(DATA_PATH)

QUESTION_ID_COL = choose_question_id_col(data)

required_cols = [SUBJECT_COL, ANSWER_COL, LABEL_COL, QUESTION_ID_COL]
missing_cols = [c for c in required_cols if c not in data.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

if QUESTION_TEXT_COL not in data.columns:
    data[QUESTION_TEXT_COL] = data[QUESTION_ID_COL].astype(str)

data[SUBJECT_COL] = data[SUBJECT_COL].astype(str)
data[ANSWER_COL] = data[ANSWER_COL].astype(str)
data[QUESTION_ID_COL] = data[QUESTION_ID_COL]
data[QUESTION_TEXT_COL] = data[QUESTION_TEXT_COL].astype(str)

print("Dataset loaded.")
print(f"Total rows: {len(data)}")
print(f"Unique subjects: {data[SUBJECT_COL].nunique()}")
print(f"Question id column: {QUESTION_ID_COL}")
print("Sample-level label distribution:")
print(data[LABEL_COL].value_counts())

print("\nQuestions per subject:")
print(data.groupby(SUBJECT_COL)[QUESTION_ID_COL].nunique().value_counts().sort_index())

# =========================
# Prepare question text
# =========================
data["question_text_normalized"] = data[QUESTION_TEXT_COL].apply(normalize_question_text)

# =========================
# Load ERNIE once
# =========================
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
model.eval()

# =========================
# Extract answer embeddings once
# =========================
print("\nExtracting answer-only embeddings...")
data["answer_embedding"] = data[ANSWER_COL].apply(
    lambda x: extract_sentence_embedding(x, tokenizer, model, max_length=MAX_LEN)
)

# =========================
# Run one-question-at-a-time experiments
# =========================
question_values = sort_question_values(data[QUESTION_ID_COL].dropna().unique())
all_results = []
skipped_questions = []

print("\n" + "=" * 80)
print("Running single-question repeated CV experiments")
print("=" * 80)

for question_value in question_values:
    question_df = data[data[QUESTION_ID_COL] == question_value].copy()

    # Keep one row per subject for this specific question if duplicates exist.
    dup_count = question_df.duplicated(subset=[SUBJECT_COL], keep=False).sum()
    if dup_count > 0:
        print(
            f"Warning: question {question_value} has duplicated subject rows. "
            "Keeping the first row per subject."
        )
        question_df = question_df.drop_duplicates(subset=[SUBJECT_COL], keep="first").copy()

    question_df = question_df.reset_index(drop=True)

    question_texts = question_df["question_text_normalized"].dropna().unique().tolist()
    question_text = question_texts[0] if len(question_texts) > 0 else str(question_value)

    y = question_df[LABEL_COL].values
    X = np.stack(question_df["answer_embedding"].values)
    groups = question_df[SUBJECT_COL].values

    n_subjects = len(question_df)
    class_counts = pd.Series(y).value_counts().to_dict()
    valid_n_splits = get_valid_n_splits(y, N_SPLITS)

    print("\n" + "-" * 80)
    print(f"Question: {question_value}")
    print(f"Question text: {question_text}")
    print(f"Subjects: {n_subjects}")
    print(f"Class counts: {class_counts}")
    print(f"Using n_splits = {valid_n_splits}")

    if len(np.unique(y)) < 2 or valid_n_splits < 2:
        print(
            f"Skipping question {question_value}: not enough class diversity "
            "for cross-validation."
        )
        skipped_questions.append({
            "question_id": question_value,
            "question_text": question_text,
            "reason": "not enough class diversity for cross-validation"
        })
        continue

    for repeat_id, seed in enumerate(SEEDS, start=1):
        sgkf = StratifiedGroupKFold(
            n_splits=valid_n_splits,
            shuffle=True,
            random_state=seed
        )

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(X, y, groups), start=1
        ):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            train_groups = groups[train_idx]
            val_groups = groups[val_idx]

            train_subjects = set(train_groups)
            val_subjects = set(val_groups)
            assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            acc, pre, rec, f1, auc = compute_metrics(y_val, y_pred, y_prob)

            all_results.append({
                "question_id": question_value,
                "question_text": question_text,
                "repeat": repeat_id,
                "seed": seed,
                "fold": fold,
                "n_splits": valid_n_splits,
                "n_subjects": n_subjects,
                "accuracy": acc,
                "precision": pre,
                "recall": rec,
                "f1": f1,
                "auc": auc
            })

# =========================
# Final result tables
# =========================
results_df = pd.DataFrame(all_results)

if results_df.empty:
    raise ValueError(
        "No valid experiment results were produced. "
        "Please check class balance for each question."
    )

summary_df = (
    results_df.groupby(["question_id", "question_text"], as_index=False)
    .agg(
        n_subjects=("n_subjects", "first"),
        n_splits=("n_splits", "first"),
        runs=("fold", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
    )
)

pretty_summary = summary_df.copy()
for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    pretty_summary[metric] = pretty_summary.apply(
        lambda row: pretty_pm(row[f"{metric}_mean"], row[f"{metric}_std"]),
        axis=1
    )

pretty_summary = pretty_summary[
    [
        "question_id",
        "question_text",
        "n_subjects",
        "n_splits",
        "runs",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]
].rename(
    columns={
        "question_id": "Question ID",
        "question_text": "Question Text",
        "n_subjects": "Subjects",
        "n_splits": "Folds",
        "runs": "Runs",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "auc": "AUC",
    }
)

print("\n" + "=" * 80)
print("Final Summary Table (One Question at a Time)")
print("=" * 80)
print(pretty_summary.to_string(index=False))

# =========================
# Save results
# =========================
if SAVE_RESULTS:
    results_df.to_csv(DETAILS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Single-question repeated CV results\n")
        f.write("=" * 80 + "\n\n")
        f.write(pretty_summary.to_string(index=False))
        f.write("\n\n")

        if skipped_questions:
            f.write("Skipped questions\n")
            f.write("-" * 80 + "\n")
            for item in skipped_questions:
                f.write(
                    f"Question {item['question_id']} | "
                    f"{item['question_text']} | "
                    f"Reason: {item['reason']}\n"
                )

    print(f"\nSaved detailed results to {DETAILS_CSV}")
    print(f"Saved summary results to {SUMMARY_CSV}")
    print(f"Saved text report to {REPORT_TXT}")

if skipped_questions:
    print("\nSkipped questions:")
    for item in skipped_questions:
        print(
            f"Question {item['question_id']} | "
            f"{item['question_text']} | "
            f"Reason: {item['reason']}"
        )
