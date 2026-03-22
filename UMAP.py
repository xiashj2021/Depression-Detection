
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, pyplot
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
import umap

# =========================
# Global plot style
# =========================
TIMES_FONT_PATH = "/mnt/c/Windows/Fonts/times.ttf"
if os.path.exists(TIMES_FONT_PATH):
    font_manager.fontManager.addfont(TIMES_FONT_PATH)

pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["axes.spines.top"] = False
pyplot.rcParams["axes.spines.right"] = False
pyplot.rcParams["axes.titlesize"] = 20
pyplot.rcParams["axes.labelsize"] = 18
pyplot.rcParams["legend.fontsize"] = 18
pyplot.rcParams["font.size"] = 18

warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value .* overridden to 1 by setting random_state"
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

SAVE_VISUALIZATIONS = True
OUTPUT_DIR = "outputs_pdf"
COMBINED_VIS_PDF = "all_methods_before_after_plots.pdf"

# =========================
# Publication-style plot config
# =========================
POINT_SIZE = 70
POINT_ALPHA = 0.85
POINT_EDGE_WIDTH = 0.5
PROB_VMIN = 0.0
PROB_VMAX = 1.0
FIG_WIDTH = 18
FIG_HEIGHT = 5.4
SUPTITLE_SIZE = 17
PANEL_TITLE_SIZE = 15

# If QUESTION_COL stores only codes like Q1/Q2,
# fill this dictionary. Otherwise keep it empty.
QUESTION_TEXT_MAP = {
    # "Q1": "Question text 1",
    # "Q2": "Question text 2",
}

# =========================
# Reproducibility
# =========================
BASE_SEED = 42
tf.random.set_seed(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)

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


def method_display_name_short(method):
    mapping = {
        "answer_meanpool_lr": "Answer Embedding + LR",
        "question_probavg_lr": "Question Prob Avg",
        "qa_meanpool_lr": "QA Embedding + LR",
    }
    return mapping.get(method, method)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def get_method_plot_filename(method):
    return f"{method}_all_folds_overlay.pdf"


def compute_global_umap_df(subject_df, title_prefix="Model"):
    if len(subject_df) < 2:
        raise ValueError(f"Not enough subjects to build a global UMAP plot for {title_prefix}.")

    X_subject = np.stack(subject_df["embedding"].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subject)

    reducer = umap.UMAP(
        n_neighbors=min(15, max(2, len(X_scaled) - 1)),
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42
    )
    coords = reducer.fit_transform(X_scaled)

    plot_df = subject_df[[SUBJECT_COL, "label"]].copy()
    plot_df["umap_1"] = coords[:, 0]
    plot_df["umap_2"] = coords[:, 1]
    return plot_df


def aggregate_oof_predictions(oof_records):
    if not oof_records:
        raise ValueError("No OOF records were collected for plotting.")

    oof_df = pd.DataFrame(oof_records)
    subject_oof_df = (
        oof_df.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean")
        )
    )
    subject_oof_df["y_pred"] = (subject_oof_df["y_prob"] > 0.5).astype(int)
    subject_oof_df["correct"] = (subject_oof_df["y_true"] == subject_oof_df["y_pred"]).astype(int)
    return subject_oof_df


def plot_all_folds_overlay_pdf(plot_df, save_path, title_prefix="Model", pdf_pages=None):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT))

    true_name_map = {0: "Control", 1: "Depression"}
    pred_name_map = {0: "Pred 0", 1: "Pred 1"}

    for cls in sorted(np.unique(plot_df["y_true"].values)):
        idx = plot_df["y_true"].values == cls
        axes[0].scatter(
            plot_df.loc[idx, "umap_1"],
            plot_df.loc[idx, "umap_2"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=POINT_EDGE_WIDTH,
            edgecolors="black",
            label=true_name_map.get(int(cls), f"True {cls}")
        )
    axes[0].set_title("Before", fontsize=PANEL_TITLE_SIZE, pad=10)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(frameon=False, loc="best", handletextpad=0.4)

    for cls in sorted(np.unique(plot_df["y_pred"].values)):
        idx = plot_df["y_pred"].values == cls
        axes[1].scatter(
            plot_df.loc[idx, "umap_1"],
            plot_df.loc[idx, "umap_2"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=POINT_EDGE_WIDTH,
            edgecolors="black",
            label=pred_name_map.get(int(cls), f"Pred {cls}")
        )
    axes[1].set_title("After", fontsize=PANEL_TITLE_SIZE, pad=10)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].legend(frameon=False, loc="best", handletextpad=0.4)

    sc = axes[2].scatter(
        plot_df["umap_1"],
        plot_df["umap_2"],
        c=plot_df["y_prob"],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=POINT_EDGE_WIDTH,
        edgecolors="black",
        vmin=PROB_VMIN,
        vmax=PROB_VMAX
    )
    axes[2].set_title("Probability", fontsize=PANEL_TITLE_SIZE, pad=10)
    axes[2].set_xlabel("UMAP-1")
    axes[2].set_ylabel("UMAP-2")
    cbar = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("P(class=1)")
    cbar.set_ticks(np.linspace(PROB_VMIN, PROB_VMAX, 6))

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.margins(x=0.05, y=0.08)

    fig.suptitle(title_prefix, fontsize=SUPTITLE_SIZE, y=0.85)
    plt.tight_layout(rect=[0, 0, 1, 0.92], w_pad=2.0)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=600)
    if pdf_pages is not None:
        pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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

ensure_output_dir(OUTPUT_DIR)
combined_pdf_path = os.path.join(OUTPUT_DIR, COMBINED_VIS_PDF)
combined_pdf = PdfPages(combined_pdf_path) if SAVE_VISUALIZATIONS else None
saved_plot_paths = []

global_answer_plot_df = compute_global_umap_df(
    subject_answer_df[[SUBJECT_COL, "label", "embedding"]].copy(),
    title_prefix="Answer Embedding + LR"
)
global_qa_plot_df = compute_global_umap_df(
    subject_qa_df[[SUBJECT_COL, "label", "embedding"]].copy(),
    title_prefix="QA Embedding + LR"
)

# =========================
# Run all methods
# =========================
all_results = []

try:
    for method in METHODS:
        print("\n" + "=" * 80)
        print(f"Running method: {method_display_name(method)}")
        print("=" * 80)

        oof_records = []

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

                    for sid, yt, yp, ypr in zip(groups_subject[val_idx], y_val, y_pred, y_prob):
                        oof_records.append({
                            "subject_id": sid,
                            "y_true": yt,
                            "y_pred": yp,
                            "y_prob": ypr,
                            "repeat": repeat_id,
                            "fold": fold
                        })

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

            if SAVE_VISUALIZATIONS:
                subject_oof_df = aggregate_oof_predictions(oof_records)
                plot_df = subject_oof_df.merge(
                    global_answer_plot_df,
                    left_on="subject_id",
                    right_on=SUBJECT_COL,
                    how="left"
                )
                if plot_df["umap_1"].isna().any():
                    raise ValueError("Missing UMAP coordinates for answer_meanpool_lr visualization.")

                save_path = os.path.join(OUTPUT_DIR, get_method_plot_filename(method))
                plot_all_folds_overlay_pdf(
                    plot_df=plot_df,
                    save_path=save_path,
                    title_prefix=method_display_name_short(method),
                    pdf_pages=combined_pdf
                )
                saved_plot_paths.append(save_path)

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

                    for _, row in subject_pred_df.iterrows():
                        oof_records.append({
                            "subject_id": row["subject_id"],
                            "y_true": row["y_true"],
                            "y_pred": row["pred"],
                            "y_prob": row["prob"],
                            "repeat": repeat_id,
                            "fold": fold
                        })

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

            if SAVE_VISUALIZATIONS:
                subject_oof_df = aggregate_oof_predictions(oof_records)
                plot_df = subject_oof_df.merge(
                    global_answer_plot_df,
                    left_on="subject_id",
                    right_on=SUBJECT_COL,
                    how="left"
                )
                if plot_df["umap_1"].isna().any():
                    raise ValueError("Missing UMAP coordinates for question_probavg_lr visualization.")

                save_path = os.path.join(OUTPUT_DIR, get_method_plot_filename(method))
                plot_all_folds_overlay_pdf(
                    plot_df=plot_df,
                    save_path=save_path,
                    title_prefix=method_display_name_short(method),
                    pdf_pages=combined_pdf
                )
                saved_plot_paths.append(save_path)

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

                    for sid, yt, yp, ypr in zip(groups_subject[val_idx], y_val, y_pred, y_prob):
                        oof_records.append({
                            "subject_id": sid,
                            "y_true": yt,
                            "y_pred": yp,
                            "y_prob": ypr,
                            "repeat": repeat_id,
                            "fold": fold
                        })

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

            if SAVE_VISUALIZATIONS:
                subject_oof_df = aggregate_oof_predictions(oof_records)
                plot_df = subject_oof_df.merge(
                    global_qa_plot_df,
                    left_on="subject_id",
                    right_on=SUBJECT_COL,
                    how="left"
                )
                if plot_df["umap_1"].isna().any():
                    raise ValueError("Missing UMAP coordinates for qa_meanpool_lr visualization.")

                save_path = os.path.join(OUTPUT_DIR, get_method_plot_filename(method))
                plot_all_folds_overlay_pdf(
                    plot_df=plot_df,
                    save_path=save_path,
                    title_prefix=method_display_name_short(method),
                    pdf_pages=combined_pdf
                )
                saved_plot_paths.append(save_path)
finally:
    if combined_pdf is not None:
        combined_pdf.close()

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

if SAVE_VISUALIZATIONS:
    print("\nVisualization PDF files:")
    for path in saved_plot_paths:
        print(path)
    print(combined_pdf_path)

if SAVE_RESULTS:
    details_path = os.path.join(OUTPUT_DIR, DETAILS_CSV)
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    results_df.to_csv(details_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved detailed results to {details_path}")
    print(f"Saved summary results to {summary_path}")
