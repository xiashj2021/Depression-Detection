import os
import re
import glob
import json
import math
import argparse
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, pyplot

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Optional semantic embedding backend
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


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
pyplot.rcParams["axes.titlesize"] = 18
pyplot.rcParams["axes.labelsize"] = 14
pyplot.rcParams["legend.fontsize"] = 12
pyplot.rcParams["font.size"] = 12


# =========================
# Defaults
# =========================
DEFAULT_SHAP_ROOT = "shap_outputs"
DEFAULT_OUTPUT_DIR = "shap_pattern_summary"

TOKEN_MIN_ABS_SHAP = 0.0
SPAN_MIN_ABS_SHAP = 0.0
TOPK_PLOT = 20

RUN_SEMANTIC_CLUSTERING = True
SEMANTIC_MODEL_NAME = "nghuyong/ernie-3.0-base-zh"
SEMANTIC_MAX_SPANS = 200
SEMANTIC_MIN_CLUSTER_SIZE = 3
SEMANTIC_MAX_CLUSTERS = 8
SEMANTIC_RANDOM_STATE = 42

STOP_UNITS = {
    "", " ", "...", "Question", "Answer", "Question:", "Answer:",
    "question", "answer", "Question :", "Answer :"
}


# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_slug(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"

def discover_shap_csvs(root_dir: str) -> Dict[str, List[str]]:
    return {
        "token": sorted(glob.glob(os.path.join(root_dir, "**", "*_token_shap.csv"), recursive=True)),
        "span": sorted(glob.glob(os.path.join(root_dir, "**", "*_span_shap.csv"), recursive=True)),
        "summary": sorted(glob.glob(os.path.join(root_dir, "**", "*_summary.csv"), recursive=True)),
    }

def infer_method_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for method in ["answer_meanpool_lr", "question_probavg_lr", "qa_meanpool_lr"]:
        if method in parts:
            return method
    return "unknown_method"

def normalize_unit(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.replace("##", "")
    text = text.replace("Ġ", " ")
    text = text.replace("▁", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # remove fixed qa prefixes if they survive span/token export
    text = re.sub(r"^(Question\s*:?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(Answer\s*:?\s*)", "", text, flags=re.IGNORECASE)

    # trim surrounding punctuation/spaces
    text = re.sub(r"^[\s\-\–\—\|,:;，。！？、（）()\[\]{}<>\"'`]+", "", text)
    text = re.sub(r"[\s\-\–\—\|,:;，。！？、（）()\[\]{}<>\"'`]+$", "", text)

    return text.strip()

def is_meaningful_unit(text: str) -> bool:
    if text in STOP_UNITS:
        return False
    if text.strip() == "":
        return False
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return False
    return True

def load_shap_tables(file_paths: List[str], unit_type: str) -> pd.DataFrame:
    tables = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            continue

        if df.empty:
            continue

        unit_col = "token" if unit_type == "token" else "span_text"
        if unit_col not in df.columns:
            continue

        df = df.copy()
        df["source_file"] = os.path.basename(path)
        df["source_path"] = path
        df["method"] = df["method"] if "method" in df.columns else infer_method_from_path(path)
        if "method" in df.columns:
            df["method"] = df["method"].fillna(infer_method_from_path(path))
        else:
            df["method"] = infer_method_from_path(path)

        for col, default in [
            ("repeat", np.nan),
            ("fold", np.nan),
            ("subject_id", ""),
            ("question", ""),
            ("label", np.nan),
            ("text_type", "")
        ]:
            if col not in df.columns:
                df[col] = default

        df["unit_text"] = df[unit_col].astype(str)
        df["unit_norm"] = df["unit_text"].apply(normalize_unit)
        df = df[df["unit_norm"].apply(is_meaningful_unit)].copy()
        if df.empty:
            continue

        if "direction" not in df.columns:
            value_col = "shap_value" if "shap_value" in df.columns else "span_shap_value"
            df["direction"] = np.where(df[value_col] >= 0, "push_up", "push_down")

        df["sample_key"] = (
            df["method"].astype(str) + "|" +
            df["repeat"].astype(str) + "|" +
            df["fold"].astype(str) + "|" +
            df["subject_id"].astype(str) + "|" +
            df["source_file"].astype(str)
        )
        df["question_key"] = (
            df["method"].astype(str) + "|" +
            df["subject_id"].astype(str) + "|" +
            df["question"].astype(str)
        )

        tables.append(df)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)

def aggregate_units(
    df: pd.DataFrame,
    value_col: str,
    abs_col: str,
    group_cols: List[str]
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    aggregated = (
        df.groupby(group_cols, dropna=False)
        .agg(
            occurrence_count=(value_col, "count"),
            sample_count=("sample_key", pd.Series.nunique),
            subject_count=("subject_id", pd.Series.nunique),
            question_count=("question_key", pd.Series.nunique),
            mean_shap=(value_col, "mean"),
            median_shap=(value_col, "median"),
            mean_abs_shap=(abs_col, "mean"),
            median_abs_shap=(abs_col, "median"),
            sum_shap=(value_col, "sum"),
            sum_abs_shap=(abs_col, "sum"),
            max_abs_shap=(abs_col, "max"),
        )
        .reset_index()
    )

    aggregated["impact_score"] = (
        aggregated["sum_abs_shap"] * np.log1p(aggregated["sample_count"].clip(lower=1))
    )
    aggregated = aggregated.sort_values(
        ["impact_score", "mean_abs_shap", "occurrence_count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    aggregated.insert(0, "rank", np.arange(1, len(aggregated) + 1))
    return aggregated

def save_barplot(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output_pdf: str,
    topk: int = 20
):
    if df.empty:
        return

    plot_df = df.head(topk).copy()
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, max(4, 0.38 * len(plot_df) + 1.5)))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel(value_col)
    plt.ylabel(label_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def top_feature_strings(texts: List[str], max_features: int = 8) -> str:
    texts = [t for t in texts if isinstance(t, str) and t.strip() != ""]
    if len(texts) == 0:
        return ""

    try:
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, max_features=100)
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        top_idx = np.argsort(-scores)[:max_features]
        top_terms = [terms[i] for i in top_idx if str(terms[i]).strip() != ""]
        return " | ".join(top_terms)
    except Exception:
        return ""

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()

def encode_texts_for_clustering(
    texts: List[str],
    model_name: str = SEMANTIC_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 64
) -> np.ndarray:
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch are not available for semantic clustering.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        with torch.no_grad():
            outputs = model(**encoded)
        emb = masked_mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings.append(emb)

    X = np.vstack(embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return X / norms

def choose_cluster_count(n_items: int, max_clusters: int = SEMANTIC_MAX_CLUSTERS) -> int:
    if n_items < 2:
        return 1
    heuristic = int(round(math.sqrt(n_items / 2)))
    heuristic = max(2, heuristic)
    heuristic = min(max_clusters, heuristic, n_items)
    return heuristic

def run_semantic_clustering(
    span_stats: pd.DataFrame,
    output_dir: str,
    scope_name: str,
    model_name: str = SEMANTIC_MODEL_NAME
):
    if span_stats.empty:
        return

    work_df = span_stats.copy()
    work_df = work_df[work_df["direction"] == "push_up"].copy()
    if work_df.empty:
        return

    work_df = work_df.sort_values(
        ["impact_score", "mean_abs_shap", "sample_count"],
        ascending=[False, False, False]
    ).head(SEMANTIC_MAX_SPANS).copy()

    if len(work_df) < SEMANTIC_MIN_CLUSTER_SIZE:
        return

    cluster_count = choose_cluster_count(len(work_df), SEMANTIC_MAX_CLUSTERS)
    if cluster_count < 2:
        return

    try:
        embeddings = encode_texts_for_clustering(
            texts=work_df["unit_norm"].tolist(),
            model_name=model_name
        )
    except Exception as e:
        warnings.warn(f"Semantic clustering failed for {scope_name}: {e}")
        return

    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=SEMANTIC_RANDOM_STATE,
        n_init=10
    )
    work_df["cluster_id"] = kmeans.fit_predict(embeddings)

    cluster_summary_rows = []
    member_rows = []

    for cluster_id, cluster_df in work_df.groupby("cluster_id"):
        cluster_df = cluster_df.sort_values(
            ["impact_score", "mean_abs_shap", "sample_count"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        representative_spans = " | ".join(cluster_df["unit_norm"].head(5).tolist())
        top_features = top_feature_strings(cluster_df["unit_norm"].tolist(), max_features=10)

        cluster_summary_rows.append({
            "cluster_id": int(cluster_id),
            "scope_name": scope_name,
            "cluster_size": int(len(cluster_df)),
            "total_occurrences": int(cluster_df["occurrence_count"].sum()),
            "mean_impact_score": float(cluster_df["impact_score"].mean()),
            "mean_abs_shap": float(cluster_df["mean_abs_shap"].mean()),
            "representative_spans": representative_spans,
            "top_character_features": top_features
        })

        cluster_members = cluster_df.copy()
        cluster_members["scope_name"] = scope_name
        member_rows.append(cluster_members)

    cluster_summary_df = pd.DataFrame(cluster_summary_rows).sort_values(
        ["mean_impact_score", "cluster_size"],
        ascending=[False, False]
    ).reset_index(drop=True)

    cluster_members_df = pd.concat(member_rows, ignore_index=True)

    cluster_summary_df.to_csv(
        os.path.join(output_dir, f"{scope_name}_semantic_clusters.csv"),
        index=False
    )
    cluster_members_df.to_csv(
        os.path.join(output_dir, f"{scope_name}_semantic_cluster_members.csv"),
        index=False
    )

    plt.figure(figsize=(10, max(4, 0.45 * len(cluster_summary_df) + 1.5)))
    plot_df = cluster_summary_df.sort_values("mean_impact_score", ascending=True)
    y_labels = [f"Cluster {cid}" for cid in plot_df["cluster_id"]]
    plt.barh(y_labels, plot_df["mean_impact_score"])
    plt.xlabel("Mean impact score")
    plt.ylabel("Cluster")
    plt.title(f"Semantic pattern clusters: {scope_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{scope_name}_semantic_clusters.pdf"), bbox_inches="tight")
    plt.close()

def build_scope_outputs(
    token_df: pd.DataFrame,
    span_df: pd.DataFrame,
    output_dir: str,
    scope_name: str
):
    ensure_dir(output_dir)

    scope_token = token_df.copy()
    scope_span = span_df.copy()

    # token-level aggregates
    token_overall = aggregate_units(
        scope_token,
        value_col="shap_value",
        abs_col="abs_shap_value",
        group_cols=["direction", "unit_norm"]
    )
    token_by_method = aggregate_units(
        scope_token,
        value_col="shap_value",
        abs_col="abs_shap_value",
        group_cols=["method", "direction", "unit_norm"]
    )

    # span-level aggregates
    span_overall = aggregate_units(
        scope_span,
        value_col="span_shap_value",
        abs_col="abs_span_shap_value",
        group_cols=["direction", "unit_norm"]
    )
    span_by_method = aggregate_units(
        scope_span,
        value_col="span_shap_value",
        abs_col="abs_span_shap_value",
        group_cols=["method", "direction", "unit_norm"]
    )

    token_overall.to_csv(os.path.join(output_dir, f"{scope_name}_token_overall.csv"), index=False)
    token_by_method.to_csv(os.path.join(output_dir, f"{scope_name}_token_by_method.csv"), index=False)
    span_overall.to_csv(os.path.join(output_dir, f"{scope_name}_span_overall.csv"), index=False)
    span_by_method.to_csv(os.path.join(output_dir, f"{scope_name}_span_by_method.csv"), index=False)

    # positive/negative subsets
    token_pos = token_overall[token_overall["direction"] == "push_up"].copy()
    token_neg = token_overall[token_overall["direction"] == "push_down"].copy()
    span_pos = span_overall[span_overall["direction"] == "push_up"].copy()
    span_neg = span_overall[span_overall["direction"] == "push_down"].copy()

    token_pos.to_csv(os.path.join(output_dir, f"{scope_name}_token_positive.csv"), index=False)
    token_neg.to_csv(os.path.join(output_dir, f"{scope_name}_token_negative.csv"), index=False)
    span_pos.to_csv(os.path.join(output_dir, f"{scope_name}_span_positive.csv"), index=False)
    span_neg.to_csv(os.path.join(output_dir, f"{scope_name}_span_negative.csv"), index=False)

    save_barplot(
        token_pos, "unit_norm", "impact_score",
        f"Top positive tokens: {scope_name}",
        os.path.join(output_dir, f"{scope_name}_top_positive_tokens.pdf"),
        topk=TOPK_PLOT
    )
    save_barplot(
        token_neg, "unit_norm", "impact_score",
        f"Top negative tokens: {scope_name}",
        os.path.join(output_dir, f"{scope_name}_top_negative_tokens.pdf"),
        topk=TOPK_PLOT
    )
    save_barplot(
        span_pos, "unit_norm", "impact_score",
        f"Top positive spans: {scope_name}",
        os.path.join(output_dir, f"{scope_name}_top_positive_spans.pdf"),
        topk=TOPK_PLOT
    )
    save_barplot(
        span_neg, "unit_norm", "impact_score",
        f"Top negative spans: {scope_name}",
        os.path.join(output_dir, f"{scope_name}_top_negative_spans.pdf"),
        topk=TOPK_PLOT
    )

    metadata = {
        "scope_name": scope_name,
        "token_rows": int(len(scope_token)),
        "span_rows": int(len(scope_span)),
        "unique_tokens": int(token_overall["unit_norm"].nunique()) if not token_overall.empty else 0,
        "unique_spans": int(span_overall["unit_norm"].nunique()) if not span_overall.empty else 0,
        "methods": sorted(scope_token["method"].dropna().astype(str).unique().tolist()) if not scope_token.empty else [],
    }
    save_json(metadata, os.path.join(output_dir, f"{scope_name}_metadata.json"))

    if RUN_SEMANTIC_CLUSTERING:
        run_semantic_clustering(
            span_stats=span_pos,
            output_dir=output_dir,
            scope_name=scope_name,
            model_name=SEMANTIC_MODEL_NAME
        )

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate token/span-level SHAP outputs into cross-sample statistics."
    )
    parser.add_argument("--shap_root", type=str, default=DEFAULT_SHAP_ROOT, help="Root directory containing shap_outputs.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save aggregated reports.")
    parser.add_argument("--topk_plot", type=int, default=TOPK_PLOT, help="Top-K items to show in bar plots.")
    parser.add_argument("--token_min_abs_shap", type=float, default=TOKEN_MIN_ABS_SHAP, help="Minimum absolute SHAP for tokens.")
    parser.add_argument("--span_min_abs_shap", type=float, default=SPAN_MIN_ABS_SHAP, help="Minimum absolute SHAP for spans.")
    parser.add_argument("--disable_semantic_clustering", action="store_true", help="Disable semantic pattern clustering.")
    parser.add_argument("--semantic_model_name", type=str, default=SEMANTIC_MODEL_NAME, help="Transformer model for semantic clustering.")
    parser.add_argument("--semantic_max_spans", type=int, default=SEMANTIC_MAX_SPANS, help="Max number of spans for clustering.")
    parser.add_argument("--semantic_max_clusters", type=int, default=SEMANTIC_MAX_CLUSTERS, help="Max number of clusters.")
    args = parser.parse_args()

    globals()["TOPK_PLOT"] = args.topk_plot
    globals()["TOKEN_MIN_ABS_SHAP"] = args.token_min_abs_shap
    globals()["SPAN_MIN_ABS_SHAP"] = args.span_min_abs_shap
    globals()["RUN_SEMANTIC_CLUSTERING"] = not args.disable_semantic_clustering
    globals()["SEMANTIC_MODEL_NAME"] = args.semantic_model_name
    globals()["SEMANTIC_MAX_SPANS"] = args.semantic_max_spans
    globals()["SEMANTIC_MAX_CLUSTERS"] = args.semantic_max_clusters

    ensure_dir(args.output_dir)

    discovered = discover_shap_csvs(args.shap_root)
    print("Discovered files:")
    for key, files in discovered.items():
        print(f"  {key}: {len(files)}")

    token_df = load_shap_tables(discovered["token"], unit_type="token")
    span_df = load_shap_tables(discovered["span"], unit_type="span")

    if token_df.empty or span_df.empty:
        raise RuntimeError(
            "No token/span SHAP CSV files were found. "
            "Please run the text-level SHAP script first so that *_token_shap.csv and *_span_shap.csv are generated."
        )

    token_df = token_df[token_df["abs_shap_value"] >= TOKEN_MIN_ABS_SHAP].copy()
    span_df = span_df[span_df["abs_span_shap_value"] >= SPAN_MIN_ABS_SHAP].copy()

    build_scope_outputs(
        token_df=token_df,
        span_df=span_df,
        output_dir=args.output_dir,
        scope_name="overall"
    )

    for method_name in sorted(token_df["method"].dropna().astype(str).unique()):
        method_token = token_df[token_df["method"] == method_name].copy()
        method_span = span_df[span_df["method"] == method_name].copy()
        if method_token.empty or method_span.empty:
            continue

        method_dir = os.path.join(args.output_dir, safe_slug(method_name))
        build_scope_outputs(
            token_df=method_token,
            span_df=method_span,
            output_dir=method_dir,
            scope_name=safe_slug(method_name)
        )

    print(f"Saved aggregated pattern reports to: {args.output_dir}")
    print("This script is post-hoc only: it reads text-level SHAP outputs and does not change model metrics.")


if __name__ == "__main__":
    main()
