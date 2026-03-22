import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager, pyplot
import shap

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

# =========================
# SHAP config
# =========================
RUN_SHAP = True
SHAP_DIR = "shap_outputs"
SHAP_MAX_DISPLAY = 20
SAVE_SUBJECT_CONTRIB_PER_FOLD = 1  # how many validation subjects to export per fold
SAVE_QUESTION_LEVEL_DETAILS = True

# Plot export options
SAVE_PLOTS_AS_PNG = False
SAVE_PLOTS_AS_PDF = True

# Text-level SHAP config
RUN_TEXT_SHAP = True
TEXT_SHAP_MAX_DISPLAY = 15
TEXT_SHAP_MAX_EVALS = 128
TEXT_SHAP_BATCH_SIZE = 8
TEXT_SHAP_QUESTION_SAMPLES_PER_FOLD = 2
TEXT_SHAP_SUBJECTS_PER_FOLD = 1
TEXT_SHAP_REPEAT_IDS = [1, 2, 3, 4, 5]
TEXT_SHAP_FOLD_IDS = [1, 2, 3, 4, 5]

# If QUESTION_COL stores only codes like Q1/Q2,
# fill this dictionary. Otherwise keep it empty.
QUESTION_TEXT_MAP = {
    # "Q1": "Full question text 1",
    # "Q2": "Full question text 2",
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

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_feature_names(n_features):
    return [f"emb_{i}" for i in range(n_features)]

def run_linear_shap(clf, X_train, X_explain, method_name, repeat_id, fold, output_dir, max_display=20):
    """
    Run SHAP for a trained LogisticRegression model.
    Saves:
      - global importance csv
      - SHAP values as .npy
      - beeswarm summary plot
      - bar summary plot
    """
    feature_names = get_feature_names(X_explain.shape[1])
    explainer = shap.LinearExplainer(clf, X_train)
    shap_values = explainer.shap_values(X_explain)

    # Compatibility across SHAP versions
    if isinstance(shap_values, list):
        shap_matrix = shap_values[0]
    else:
        shap_matrix = shap_values

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coef": clf.coef_[0],
        "mean_abs_shap": np.abs(shap_matrix).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    base_name = f"{method_name}_repeat{repeat_id}_fold{fold}"

    importance_df.to_csv(
        os.path.join(output_dir, f"{base_name}_global_importance.csv"),
        index=False
    )

    np.save(
        os.path.join(output_dir, f"{base_name}_shap_values.npy"),
        shap_matrix
    )

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_matrix,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=max_display
    )
    plt.tight_layout()
    if SAVE_PLOTS_AS_PNG:
        plt.savefig(
            os.path.join(output_dir, f"{base_name}_summary_beeswarm.png"),
            dpi=300,
            bbox_inches="tight"
        )
    if SAVE_PLOTS_AS_PDF:
        plt.savefig(
            os.path.join(output_dir, f"{base_name}_summary_beeswarm.pdf"),
            bbox_inches="tight"
        )
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_matrix,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_display
    )
    plt.tight_layout()
    if SAVE_PLOTS_AS_PNG:
        plt.savefig(
            os.path.join(output_dir, f"{base_name}_summary_bar.png"),
            dpi=300,
            bbox_inches="tight"
        )
    if SAVE_PLOTS_AS_PDF:
        plt.savefig(
            os.path.join(output_dir, f"{base_name}_summary_bar.pdf"),
            bbox_inches="tight"
        )
    plt.close()

    return explainer, shap_matrix, importance_df

def save_question_level_shap_details(
    data_df,
    row_indices,
    clf,
    shap_matrix,
    y_prob_question,
    method_name,
    repeat_id,
    fold,
    output_dir
):
    """
    Save question-level SHAP details for question_probavg_lr.
    Each row is one question/answer sample.
    """
    subset = data_df.iloc[row_indices][[
        SUBJECT_COL, QUESTION_COL, "question_text", ANSWER_COL, LABEL_COL
    ]].copy().reset_index(drop=True)

    subset["question_prob"] = y_prob_question
    subset["question_logit"] = clf.decision_function(np.stack(data_df.iloc[row_indices]["answer_embedding"].values))
    subset["total_shap"] = shap_matrix.sum(axis=1)

    feature_names = get_feature_names(shap_matrix.shape[1])
    top_pos_feature = []
    top_pos_value = []
    top_neg_feature = []
    top_neg_value = []

    for row in shap_matrix:
        pos_idx = int(np.argmax(row))
        neg_idx = int(np.argmin(row))
        top_pos_feature.append(feature_names[pos_idx])
        top_pos_value.append(row[pos_idx])
        top_neg_feature.append(feature_names[neg_idx])
        top_neg_value.append(row[neg_idx])

    subset["top_positive_feature"] = top_pos_feature
    subset["top_positive_shap"] = top_pos_value
    subset["top_negative_feature"] = top_neg_feature
    subset["top_negative_shap"] = top_neg_value

    base_name = f"{method_name}_repeat{repeat_id}_fold{fold}"
    subset.to_csv(
        os.path.join(output_dir, f"{base_name}_question_level_explanations.csv"),
        index=False
    )
    return subset

def explain_subject_question_contributions_meanpool(
    subject_id,
    raw_df,
    clf,
    background_mean,
    emb_col,
    method_name,
    repeat_id,
    fold,
    output_dir
):
    """
    For mean-pooling methods:
    subject embedding = mean(question embeddings)
    logit = w^T x_subject + b

    This function decomposes the subject-level logit shift into question-level contributions:
        contribution(question_j) = sum_i [ w_i * (q_ji - background_i) / K ]
    where K is the number of questions of this subject.
    """
    subject_rows = raw_df[raw_df[SUBJECT_COL] == str(subject_id)].copy()
    if subject_rows.empty:
        return None

    q_embs = np.stack(subject_rows[emb_col].values)
    coef = clf.coef_[0]
    n_questions = q_embs.shape[0]

    question_feature_contrib = ((q_embs - background_mean) * coef) / n_questions
    question_total_contrib = question_feature_contrib.sum(axis=1)

    out = subject_rows[[
        SUBJECT_COL, QUESTION_COL, "question_text", ANSWER_COL, LABEL_COL
    ]].copy()

    out["question_logit_contribution"] = question_total_contrib
    out["direction"] = np.where(
        out["question_logit_contribution"] >= 0,
        "push_to_class_1",
        "push_to_class_0"
    )

    feature_names = get_feature_names(q_embs.shape[1])
    top_pos_feature = []
    top_pos_value = []
    top_neg_feature = []
    top_neg_value = []

    for row in question_feature_contrib:
        pos_idx = int(np.argmax(row))
        neg_idx = int(np.argmin(row))
        top_pos_feature.append(feature_names[pos_idx])
        top_pos_value.append(row[pos_idx])
        top_neg_feature.append(feature_names[neg_idx])
        top_neg_value.append(row[neg_idx])

    out["top_positive_feature"] = top_pos_feature
    out["top_positive_contribution"] = top_pos_value
    out["top_negative_feature"] = top_neg_feature
    out["top_negative_contribution"] = top_neg_value

    out = out.sort_values("question_logit_contribution", ascending=False)

    base_name = f"{method_name}_repeat{repeat_id}_fold{fold}_subject_{subject_id}"
    out.to_csv(
        os.path.join(output_dir, f"{base_name}_question_contributions.csv"),
        index=False
    )
    return out


def encode_texts_individually(text_list, tokenizer, model, max_length=MAX_LEN):
    """
    Encode raw texts one by one to preserve the same embedding path used in training.
    This avoids batch-padding differences when mean pooling over token states.
    """
    embeddings = []
    for text_item in text_list:
        emb = extract_sentence_embedding(text_item, tokenizer, model, max_length=max_length)
        embeddings.append(emb)
    return np.stack(embeddings)

def build_text_logit_predictor(clf, tokenizer, model, max_length=MAX_LEN):
    """
    Black-box predictor for question_probavg_lr:
    raw answer text -> embedding -> LR logit
    """
    def predict_fn(text_list):
        X = encode_texts_individually(text_list, tokenizer, model, max_length=max_length)
        return clf.decision_function(X)
    return predict_fn

def build_centered_contribution_predictor(
    clf,
    tokenizer,
    model,
    background_mean,
    divisor,
    max_length=MAX_LEN
):
    """
    Black-box predictor for mean-pooling methods:
    raw text -> embedding -> centered contribution to subject-level logit

    contribution(text_j) = w^T (emb_j - background_mean) / divisor
    """
    coef = clf.coef_[0].copy()
    bg = np.asarray(background_mean).copy()
    divisor = float(divisor)

    def predict_fn(text_list):
        X = encode_texts_individually(text_list, tokenizer, model, max_length=max_length)
        return ((X - bg) * coef).sum(axis=1) / divisor

    return predict_fn

def should_run_text_shap(repeat_id, fold):
    return (repeat_id in TEXT_SHAP_REPEAT_IDS) and (fold in TEXT_SHAP_FOLD_IDS)

def flatten_text_shap_values(sample_explanation):
    values = np.asarray(sample_explanation.values, dtype=float).squeeze()
    tokens = sample_explanation.data

    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    else:
        tokens = list(tokens)

    if len(tokens) != len(values):
        min_len = min(len(tokens), len(values))
        tokens = tokens[:min_len]
        values = values[:min_len]

    return tokens, values

def clean_token_text(token):
    token = str(token)
    token = token.replace("##", "")
    token = token.replace("Ġ", " ")
    token = token.replace("▁", " ")
    return token

def explanation_to_token_df(sample_explanation):
    tokens, values = flatten_text_shap_values(sample_explanation)
    rows = []
    for idx, (token, value) in enumerate(zip(tokens, values), start=1):
        clean_token = clean_token_text(token)
        if clean_token.strip() == "":
            continue
        rows.append({
            "position": idx,
            "token": clean_token,
            "shap_value": float(value),
            "abs_shap_value": float(abs(value)),
            "direction": "push_up" if value >= 0 else "push_down"
        })
    return pd.DataFrame(rows)

def merge_token_spans(token_df):
    """
    Merge adjacent tokens with the same direction into spans.
    This is a simple heuristic so that Chinese explanations are not too fragmented.
    """
    if token_df.empty:
        return pd.DataFrame(columns=[
            "span_rank", "span_text", "span_shap_value",
            "abs_span_shap_value", "direction", "start_position", "end_position"
        ])

    spans = []
    current_tokens = [token_df.iloc[0]["token"]]
    current_value = float(token_df.iloc[0]["shap_value"])
    current_dir = token_df.iloc[0]["direction"]
    start_pos = int(token_df.iloc[0]["position"])
    end_pos = start_pos

    for _, row in token_df.iloc[1:].iterrows():
        token = row["token"]
        value = float(row["shap_value"])
        direction = row["direction"]
        position = int(row["position"])

        if direction == current_dir and position == end_pos + 1:
            current_tokens.append(token)
            current_value += value
            end_pos = position
        else:
            spans.append({
                "span_text": "".join(current_tokens).strip(),
                "span_shap_value": current_value,
                "abs_span_shap_value": abs(current_value),
                "direction": current_dir,
                "start_position": start_pos,
                "end_position": end_pos
            })
            current_tokens = [token]
            current_value = value
            current_dir = direction
            start_pos = position
            end_pos = position

    spans.append({
        "span_text": "".join(current_tokens).strip(),
        "span_shap_value": current_value,
        "abs_span_shap_value": abs(current_value),
        "direction": current_dir,
        "start_position": start_pos,
        "end_position": end_pos
    })

    span_df = pd.DataFrame(spans)
    span_df = span_df[span_df["span_text"] != ""].sort_values(
        "abs_span_shap_value", ascending=False
    ).reset_index(drop=True)
    span_df.insert(0, "span_rank", np.arange(1, len(span_df) + 1))
    return span_df

def save_top_token_barplot(token_df, output_path, title, topk=15):
    if token_df.empty:
        return

    plot_df = token_df.sort_values("abs_shap_value", ascending=False).head(topk).copy()
    plot_df = plot_df.sort_values("shap_value", ascending=True)

    plt.figure(figsize=(10, max(4, 0.35 * len(plot_df) + 1.5)))
    plt.barh(plot_df["token"], plot_df["shap_value"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP value")
    plt.ylabel("Token")
    plt.title(title)
    plt.tight_layout()

    if SAVE_PLOTS_AS_PNG:
        png_path = output_path.replace(".pdf", ".png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
    if SAVE_PLOTS_AS_PDF:
        plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def save_text_shap_batch_outputs(
    shap_values,
    metadata_rows,
    output_dir,
    base_name,
    topk=15
):
    """
    Save per-sample token-level explanations and a fold-level summary.
    """
    ensure_dir(output_dir)
    summary_rows = []

    for i in range(len(metadata_rows)):
        sample_exp = shap_values[i]
        token_df = explanation_to_token_df(sample_exp)
        span_df = merge_token_spans(token_df)

        meta = metadata_rows[i].copy()
        sample_stub = f"{base_name}_sample{i+1:02d}"

        if token_df.empty:
            continue

        for k, v in meta.items():
            token_df[k] = v
            span_df[k] = v

        token_df.to_csv(
            os.path.join(output_dir, f"{sample_stub}_token_shap.csv"),
            index=False
        )
        span_df.to_csv(
            os.path.join(output_dir, f"{sample_stub}_span_shap.csv"),
            index=False
        )

        title_parts = []
        if "method" in meta:
            title_parts.append(str(meta["method"]))
        if "subject_id" in meta:
            title_parts.append(f"subject {meta['subject_id']}")
        if "question" in meta:
            title_parts.append(str(meta["question"]))
        title_text = " | ".join(title_parts) if title_parts else "Text SHAP"

        save_top_token_barplot(
            token_df=token_df,
            output_path=os.path.join(output_dir, f"{sample_stub}_top_tokens.pdf"),
            title=title_text,
            topk=topk
        )

        top_token = token_df.sort_values("abs_shap_value", ascending=False).iloc[0]
        top_span = span_df.iloc[0] if not span_df.empty else None

        summary_rows.append({
            **meta,
            "base_value": float(np.asarray(sample_exp.base_values).squeeze()),
            "model_output": float(np.asarray(sample_exp.values).sum() + np.asarray(sample_exp.base_values).squeeze()),
            "top_token": top_token["token"],
            "top_token_shap": float(top_token["shap_value"]),
            "top_span": "" if top_span is None else top_span["span_text"],
            "top_span_shap": np.nan if top_span is None else float(top_span["span_shap_value"])
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(output_dir, f"{base_name}_summary.csv"),
            index=False
        )

def explain_question_probavg_texts(
    data_df,
    row_indices,
    clf,
    tokenizer,
    model,
    method_name,
    repeat_id,
    fold,
    output_dir
):
    selected = list(row_indices)[:TEXT_SHAP_QUESTION_SAMPLES_PER_FOLD]
    if len(selected) == 0:
        return

    texts = data_df.iloc[selected]["answer_text"].tolist()
    metadata_rows = []
    for row_idx in selected:
        row = data_df.iloc[row_idx]
        metadata_rows.append({
            "method": method_name,
            "repeat": repeat_id,
            "fold": fold,
            "subject_id": row[SUBJECT_COL],
            "question": row["question_text"],
            "label": row[LABEL_COL],
            "text_type": "answer_text"
        })

    predictor = build_text_logit_predictor(
        clf=clf,
        tokenizer=tokenizer,
        model=model,
        max_length=MAX_LEN
    )
    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
    explainer = shap.Explainer(predictor, masker)
    shap_values = explainer(
        texts,
        max_evals=TEXT_SHAP_MAX_EVALS,
        batch_size=TEXT_SHAP_BATCH_SIZE
    )

    save_text_shap_batch_outputs(
        shap_values=shap_values,
        metadata_rows=metadata_rows,
        output_dir=output_dir,
        base_name=f"{method_name}_repeat{repeat_id}_fold{fold}_text_shap",
        topk=TEXT_SHAP_MAX_DISPLAY
    )

def explain_meanpool_subject_texts(
    data_df,
    subject_id,
    clf,
    tokenizer,
    model,
    background_mean,
    emb_text_col,
    method_name,
    repeat_id,
    fold,
    output_dir
):
    subject_rows = data_df[data_df[SUBJECT_COL] == str(subject_id)].copy().reset_index(drop=True)
    if subject_rows.empty:
        return

    texts = subject_rows[emb_text_col].tolist()
    divisor = len(texts)

    predictor = build_centered_contribution_predictor(
        clf=clf,
        tokenizer=tokenizer,
        model=model,
        background_mean=background_mean,
        divisor=divisor,
        max_length=MAX_LEN
    )
    masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
    explainer = shap.Explainer(predictor, masker)
    shap_values = explainer(
        texts,
        max_evals=TEXT_SHAP_MAX_EVALS,
        batch_size=TEXT_SHAP_BATCH_SIZE
    )

    metadata_rows = []
    for _, row in subject_rows.iterrows():
        metadata_rows.append({
            "method": method_name,
            "repeat": repeat_id,
            "fold": fold,
            "subject_id": row[SUBJECT_COL],
            "question": row["question_text"],
            "label": row[LABEL_COL],
            "text_type": emb_text_col
        })

    save_text_shap_batch_outputs(
        shap_values=shap_values,
        metadata_rows=metadata_rows,
        output_dir=output_dir,
        base_name=f"{method_name}_repeat{repeat_id}_fold{fold}_subject_{subject_id}_text_shap",
        topk=TEXT_SHAP_MAX_DISPLAY
    )
# =========================
# Plot style
# =========================
TIMES_FONT_PATHS = [
    "/mnt/c/Windows/Fonts/times.ttf",
    "/mnt/c/Windows/Fonts/timesbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/timesnewroman.ttf",
]

CHINESE_FONT_PATHS = [
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/mnt/c/Windows/Fonts/simsun.ttf",
]

for _font_path in TIMES_FONT_PATHS + CHINESE_FONT_PATHS:
    if os.path.exists(_font_path):
        try:
            font_manager.fontManager.addfont(_font_path)
        except Exception:
            pass

pyplot.rcParams["font.family"] = [
    "Times New Roman",
    "SimSun",
]
pyplot.rcParams["axes.unicode_minus"] = False
pyplot.rcParams["axes.spines.top"] = False
pyplot.rcParams["axes.spines.right"] = False
pyplot.rcParams["axes.titlesize"] = 20
pyplot.rcParams["axes.labelsize"] = 18
pyplot.rcParams["legend.fontsize"] = 18
pyplot.rcParams["font.size"] = 18

# =========================
# Prepare folders
# =========================
if RUN_SHAP:
    ensure_dir(SHAP_DIR)

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

    method_shap_dir = os.path.join(SHAP_DIR, method) if RUN_SHAP else None
    method_sentence_level_dir = os.path.join(method_shap_dir, "sentence_level") if RUN_SHAP else None
    method_text_shap_dir = os.path.join(method_shap_dir, "text_level") if RUN_SHAP else None
    if RUN_SHAP:
        ensure_dir(method_shap_dir)
        ensure_dir(method_sentence_level_dir)
        if RUN_TEXT_SHAP:
            ensure_dir(method_text_shap_dir)

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
                print(f"[{method}] repeat={repeat_id}, fold={fold}")

                X_train, X_val = X_subject[train_idx], X_subject[val_idx]
                y_train, y_val = y_subject[train_idx], y_subject[val_idx]

                train_subjects = set(groups_subject[train_idx])
                val_subjects = set(groups_subject[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                if RUN_SHAP:
                    _, shap_matrix, importance_df = run_linear_shap(
                        clf=clf,
                        X_train=X_train,
                        X_explain=X_val,
                        method_name=method,
                        repeat_id=repeat_id,
                        fold=fold,
                        output_dir=method_sentence_level_dir,
                        max_display=SHAP_MAX_DISPLAY
                    )
                    print("  SHAP global top features:")
                    print(importance_df.head(10).to_string(index=False))

                    background_mean = X_train.mean(axis=0)
                    example_subject_ids = groups_subject[val_idx][:SAVE_SUBJECT_CONTRIB_PER_FOLD]
                    for example_subject_id in example_subject_ids:
                        explain_subject_question_contributions_meanpool(
                            subject_id=example_subject_id,
                            raw_df=data,
                            clf=clf,
                            background_mean=background_mean,
                            emb_col="answer_embedding",
                            method_name=method,
                            repeat_id=repeat_id,
                            fold=fold,
                            output_dir=method_sentence_level_dir
                        )

                    if RUN_TEXT_SHAP and should_run_text_shap(repeat_id, fold):
                        text_subject_ids = groups_subject[val_idx][:TEXT_SHAP_SUBJECTS_PER_FOLD]
                        for subject_id_to_explain in text_subject_ids:
                            explain_meanpool_subject_texts(
                                data_df=data,
                                subject_id=subject_id_to_explain,
                                clf=clf,
                                tokenizer=tokenizer,
                                model=model,
                                background_mean=background_mean,
                                emb_text_col="answer_text",
                                method_name=method,
                                repeat_id=repeat_id,
                                fold=fold,
                                output_dir=method_text_shap_dir
                            )

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
                print(f"[{method}] repeat={repeat_id}, fold={fold}")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                val_groups = groups[val_idx]

                train_subjects = set(groups[train_idx])
                val_subjects = set(groups[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                y_prob_question = clf.predict_proba(X_val)[:, 1]

                if RUN_SHAP:
                    _, shap_matrix, importance_df = run_linear_shap(
                        clf=clf,
                        X_train=X_train,
                        X_explain=X_val,
                        method_name=method,
                        repeat_id=repeat_id,
                        fold=fold,
                        output_dir=method_sentence_level_dir,
                        max_display=SHAP_MAX_DISPLAY
                    )
                    print("  SHAP global top features:")
                    print(importance_df.head(10).to_string(index=False))

                    if SAVE_QUESTION_LEVEL_DETAILS:
                        save_question_level_shap_details(
                            data_df=data,
                            row_indices=val_idx,
                            clf=clf,
                            shap_matrix=shap_matrix,
                            y_prob_question=y_prob_question,
                            method_name=method,
                            repeat_id=repeat_id,
                            fold=fold,
                            output_dir=method_sentence_level_dir
                        )

                    if RUN_TEXT_SHAP and should_run_text_shap(repeat_id, fold):
                        explain_question_probavg_texts(
                            data_df=data,
                            row_indices=val_idx,
                            clf=clf,
                            tokenizer=tokenizer,
                            model=model,
                            method_name=method,
                            repeat_id=repeat_id,
                            fold=fold,
                            output_dir=method_text_shap_dir
                        )

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
                print(f"[{method}] repeat={repeat_id}, fold={fold}")

                X_train, X_val = X_subject[train_idx], X_subject[val_idx]
                y_train, y_val = y_subject[train_idx], y_subject[val_idx]

                train_subjects = set(groups_subject[train_idx])
                val_subjects = set(groups_subject[val_idx])
                assert train_subjects.isdisjoint(val_subjects), "Subject leakage detected!"

                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)

                if RUN_SHAP:
                    _, shap_matrix, importance_df = run_linear_shap(
                        clf=clf,
                        X_train=X_train,
                        X_explain=X_val,
                        method_name=method,
                        repeat_id=repeat_id,
                        fold=fold,
                        output_dir=method_sentence_level_dir,
                        max_display=SHAP_MAX_DISPLAY
                    )
                    print("  SHAP global top features:")
                    print(importance_df.head(10).to_string(index=False))

                    background_mean = X_train.mean(axis=0)
                    example_subject_ids = groups_subject[val_idx][:SAVE_SUBJECT_CONTRIB_PER_FOLD]
                    for example_subject_id in example_subject_ids:
                        explain_subject_question_contributions_meanpool(
                            subject_id=example_subject_id,
                            raw_df=data,
                            clf=clf,
                            background_mean=background_mean,
                            emb_col="qa_embedding",
                            method_name=method,
                            repeat_id=repeat_id,
                            fold=fold,
                            output_dir=method_sentence_level_dir
                        )

                    if RUN_TEXT_SHAP and should_run_text_shap(repeat_id, fold):
                        text_subject_ids = groups_subject[val_idx][:TEXT_SHAP_SUBJECTS_PER_FOLD]
                        for subject_id_to_explain in text_subject_ids:
                            explain_meanpool_subject_texts(
                                data_df=data,
                                subject_id=subject_id_to_explain,
                                clf=clf,
                                tokenizer=tokenizer,
                                model=model,
                                background_mean=background_mean,
                                emb_text_col="qa_text",
                                method_name=method,
                                repeat_id=repeat_id,
                                fold=fold,
                                output_dir=method_text_shap_dir
                            )

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

if RUN_SHAP:
    print(f"Saved SHAP outputs under: {SHAP_DIR}")
    if RUN_TEXT_SHAP:
        print("Text-level SHAP uses the original-text -> embedding -> LR path and does not change metric computation.")
