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
# tf.config.set_visible_devices([], 'GPU')
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'

# =========================
# Global config
# =========================
DATA_PATH = "./Dataset.xlsx"

SUBJECT_COL = "id"
QUESTION_COL = "question"
ANSWER_COL = "text"
LABEL_COL = "class"

MAX_LEN = 150
N_SPLITS = 5
SEEDS = [0, 1, 2, 3, 4]

METHODS = [
    "answer_meanpool_lr",
    "question_probavg_lr",
    "qa_meanpool_lr"
]

SAVE_RESULTS = True
DETAILS_CSV = "all_methods_repeated_cv_details.csv"
SUMMARY_CSV = "all_methods_repeated_cv_summary.csv"

# If QUESTION_COL stores only codes like Q1/Q2,
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

def extract_sentence_embedding(text, tokenizer, model, max_length=100):
    inputs = tokenizer(
        text,
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

def aggregate_probabilities_to_subject(subject_ids, y_true_text, y_prob_text, threshold=0.5):
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "y_true": y_true_text,
        "prob": y_prob_text
    })
    subject_df = (
        df.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            prob=("prob", "mean")
        )
    )
    subject_df["pred"] = (subject_df["prob"] > threshold).astype(int)
    return subject_df

def method_display_name(method):
    mapping = {
        "answer_meanpool_lr": "Answer-only encoding + mean pooling + subject-level LR",
        "question_probavg_lr": "Question-level prediction + subject-level probability averaging",
        "qa_meanpool_lr": "Question-answer joint encoding + mean pooling + subject-level LR",
    }
    return mapping.get(method, method)

# =========================
# Load data
# =========================
data = pd.read_excel(DATA_PATH)

required_cols = [SUBJECT_COL, ANSWER_COL, LABEL_COL, QUESTION_COL]
missing_cols = [c for c in required_cols if c not in data.columns]
if missing_cols:
    raise ValueError(f"Dataset is missing required columns: {missing_cols}")

data[SUBJECT_COL] = data[SUBJECT_COL].astype(str)
data[ANSWER_COL] = data[ANSWER_COL].astype(str)
data[QUESTION_COL] = data[QUESTION_COL].astype(str)

print("Dataset loaded.")
print(f"Total samples: {len(data)}")
print(f"Unique subjects: {data[SUBJECT_COL].nunique()}")
print("Sample-level label distribution:")
print(data[LABEL_COL].value_counts())

print("\nQuestions per subject:")
print(data.groupby(SUBJECT_COL)[QUESTION_COL].nunique().value_counts().sort_index())

# =========================
# Prepare text variants
# =========================
data["question_text"] = data[QUESTION_COL].apply(normalize_question_text)
data["answer_text"] = data[ANSWER_COL]
data["qa_text"] = (
    "Question: " + data["question_text"].astype(str) +
    " Answer: " + data["answer_text"].astype(str)
)

# =========================
# Load ERNIE once
# =========================
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
model = AutoModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
model.eval()

# =========================
# Extract embeddings once for each text type
# =========================
print("\nExtracting answer-only embeddings...")
data["answer_embedding"] = data["answer_text"].apply(
    lambda x: extract_sentence_embedding(x, tokenizer, model, max_length=MAX_LEN)
)

print("\nExtracting question-answer joint embeddings...")
data["qa_embedding"] = data["qa_text"].apply(
    lambda x: extract_sentence_embedding(x, tokenizer, model, max_length=MAX_LEN)
)

# =========================
# Shared subject-level tables
# =========================
subject_answer_df = (
    data.groupby(SUBJECT_COL, as_index=False)
    .agg(
        label=(LABEL_COL, "first"),
        embedding=("answer_embedding", lambda x: np.mean(np.stack(x.values), axis=0)),
        n_questions=(ANSWER_COL, "count")
    )
)

subject_qa_df = (
    data.groupby(SUBJECT_COL, as_index=False)
    .agg(
        label=(LABEL_COL, "first"),
        embedding=("qa_embedding", lambda x: np.mean(np.stack(x.values), axis=0)),
        n_questions=(ANSWER_COL, "count")
    )
)

print("\nSubject counts by number of questions:")
print(subject_answer_df["n_questions"].value_counts().sort_index())

# =========================
# Run all methods
# =========================
all_results = []

for method in METHODS:
    print("\n" + "=" * 80)
    print(f"Running method: {method_display_name(method)}")
    print("=" * 80)

    if method == "answer_meanpool_lr":
        X_subject = np.stack(subject_answer_df["embedding"].values)
        y_subject = subject_answer_df["label"].values
        groups_subject = subject_answer_df[SUBJECT_COL].values

        for repeat_id, seed in enumerate(SEEDS, start=1):
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, val_idx) in enumerate(
                sgkf.split(X_subject, y_subject, groups_subject), start=1
            ):
                X_train, X_val = X_subject[train_idx], X_subject[val_idx]
                y_train, y_val = y_subject[train_idx], y_subject[val_idx]

                train_subjects = set(groups_subject[train_idx])
                val_subjects = set(groups_subject[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                y_prob = clf.predict_proba(X_val)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)

                acc, pre, rec, f1, auc = compute_metrics(y_val, y_pred, y_prob)

                all_results.append({
                    "method": method,
                    "method_name": method_display_name(method),
                    "repeat": repeat_id,
                    "seed": seed,
                    "fold": fold,
                    "accuracy": acc,
                    "precision": pre,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc
                })

    elif method == "question_probavg_lr":
        X = np.stack(data["answer_embedding"].values)
        y = data[LABEL_COL].values
        groups = data[SUBJECT_COL].values

        for repeat_id, seed in enumerate(SEEDS, start=1):
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, val_idx) in enumerate(
                sgkf.split(X, y, groups), start=1
            ):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                val_groups = groups[val_idx]

                train_subjects = set(groups[train_idx])
                val_subjects = set(groups[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                y_prob_question = clf.predict_proba(X_val)[:, 1]

                subject_pred_df = aggregate_probabilities_to_subject(
                    subject_ids=val_groups,
                    y_true_text=y_val,
                    y_prob_text=y_prob_question,
                    threshold=0.5
                )

                acc, pre, rec, f1, auc = compute_metrics(
                    subject_pred_df["y_true"].values,
                    subject_pred_df["pred"].values,
                    subject_pred_df["prob"].values
                )

                all_results.append({
                    "method": method,
                    "method_name": method_display_name(method),
                    "repeat": repeat_id,
                    "seed": seed,
                    "fold": fold,
                    "accuracy": acc,
                    "precision": pre,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc
                })

    elif method == "qa_meanpool_lr":
        X_subject = np.stack(subject_qa_df["embedding"].values)
        y_subject = subject_qa_df["label"].values
        groups_subject = subject_qa_df[SUBJECT_COL].values

        for repeat_id, seed in enumerate(SEEDS, start=1):
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=seed
            )

            for fold, (train_idx, val_idx) in enumerate(
                sgkf.split(X_subject, y_subject, groups_subject), start=1
            ):
                X_train, X_val = X_subject[train_idx], X_subject[val_idx]
                y_train, y_val = y_subject[train_idx], y_subject[val_idx]

                train_subjects = set(groups_subject[train_idx])
                val_subjects = set(groups_subject[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                y_prob = clf.predict_proba(X_val)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)

                acc, pre, rec, f1, auc = compute_metrics(y_val, y_pred, y_prob)

                all_results.append({
                    "method": method,
                    "method_name": method_display_name(method),
                    "repeat": repeat_id,
                    "seed": seed,
                    "fold": fold,
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

summary_df = (
    results_df.groupby(["method", "method_name"], as_index=False)
    .agg(
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

# Pretty-print table for paper use
pretty_summary = summary_df.copy()
for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    pretty_summary[metric] = pretty_summary.apply(
        lambda row: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}",
        axis=1
    )

pretty_summary = pretty_summary[
    ["method_name", "accuracy", "precision", "recall", "f1", "auc"]
].rename(columns={"method_name": "Method"})

print("\n" + "=" * 80)
print("Final Summary Table")
print("=" * 80)
print(pretty_summary.to_string(index=False))

# =========================
# Save results
# =========================
if SAVE_RESULTS:
    results_df.to_csv(DETAILS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved detailed results to {DETAILS_CSV}")
    print(f"Saved summary results to {SUMMARY_CSV}")
