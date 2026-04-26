import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#constants
PROJECTS   = ["TensorFlow", "PyTorch", "Keras", "Mxnet", "Caffe"]
N_REPEATS  = 30
TRAIN_FRAC = 0.70
RANDOM_SEED = 42

TEXT_COLS  = ["Title", "title", "text", "body", "report",
              "description", "summary", "content"]
LABEL_COLS = ["label", "Label", "class", "is_performance_bug",
              "performance", "bug_type"]

#loadfiles

def load_csv(path):
    """Load a project CSV and return (texts, labels) arrays."""
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    # detect text column
    text_col = next((c for c in TEXT_COLS if c in df.columns), None)
    if text_col is None:
        # fallback first object column
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if not obj_cols:
            raise ValueError(f"No text column found in {path}. Columns: {df.columns.tolist()}")
        text_col = obj_cols[0]

    # detect label column
    label_col = next((c for c in LABEL_COLS if c in df.columns), None)
    if label_col is None:
        # fallback first integer/boolean column
        int_cols = df.select_dtypes(include=["int64", "int32", "bool"]).columns.tolist()
        if not int_cols:
            raise ValueError(f"No label column found in {path}. Columns: {df.columns.tolist()}")
        label_col = int_cols[0]

    texts  = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def make_baseline():
    """Naive Bayes + TF-IDF pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB())
    ])


def make_proposed():
    """LinearSVC + TF-IDF pipeline with class balancing."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(
            class_weight="balanced",   # handles ~16% positive imbalance
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_SEED
        ))
    ])


def single_run(texts, labels, seed):
    """One train and test split, return metrics for both approaches."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(texts))
    rng.shuffle(indices)

    #positive samples in both splits
    pos_idx = [i for i in indices if labels[i] == 1]
    neg_idx = [i for i in indices if labels[i] == 0]

    n_pos_train = max(1, int(len(pos_idx) * TRAIN_FRAC))
    n_neg_train = max(1, int(len(neg_idx) * TRAIN_FRAC))

    train_idx = pos_idx[:n_pos_train] + neg_idx[:n_neg_train]
    test_idx  = pos_idx[n_pos_train:] + neg_idx[n_neg_train:]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test  = [texts[i] for i in test_idx]
    y_test  = [labels[i] for i in test_idx]

    results = {}
    for name, model in [("baseline", make_baseline()),
                         ("proposed", make_proposed())]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score   (y_test, y_pred, zero_division=0),
            "f1":        f1_score       (y_test, y_pred, zero_division=0),
        }
    return results


def run_project(project, data_dir):
    path = os.path.join(data_dir, f"{project}.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None

    texts, labels = load_csv(path)
    print(f"  Loaded {len(texts)} reports "
          f"({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

    agg = {"baseline": {"precision":[], "recall":[], "f1":[]},
           "proposed":  {"precision":[], "recall":[], "f1":[]}}

    for rep in range(N_REPEATS):
        res = single_run(texts, labels, seed=rep)
        for approach in ["baseline", "proposed"]:
            for metric in ["precision", "recall", "f1"]:
                agg[approach][metric].append(res[approach][metric])

    # convert to numpy arrays
    for approach in agg:
        for metric in agg[approach]:
            agg[approach][metric] = np.array(agg[approach][metric])

    return agg


def wilcoxon_test(a, b):
    if np.allclose(a, b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b)
    except Exception:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def vargha_delaney(a, b):
    n = len(a) * len(b)
    rank_sum = sum(1 if ai > bi else 0.5 if ai == bi else 0
                   for ai in a for bi in b)
    return rank_sum / n


#plotting

def plot_boxplots(all_results, out_dir):
    metrics = ["precision", "recall", "f1"]
    projects = [p for p in PROJECTS if all_results.get(p) is not None]

    for metric in metrics:
        fig, axes = plt.subplots(1, len(projects),
                                 figsize=(3 * len(projects), 4),
                                 sharey=True)
        if len(projects) == 1:
            axes = [axes]

        for ax, project in zip(axes, projects):
            data_b = all_results[project]["baseline"][metric]
            data_p = all_results[project]["proposed"][metric]
            bp = ax.boxplot([data_b, data_p],
                            patch_artist=True,
                            widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            bp["boxes"][0].set_facecolor("#AEC6CF")
            bp["boxes"][1].set_facecolor("#FFD1A9")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Baseline\n(NB)", "Proposed\n(SVM)"],
                               fontsize=8)
            ax.set_title(project, fontsize=9)
            ax.set_ylim(0, 1.05)

        axes[0].set_ylabel(metric.capitalize(), fontsize=10)
        fig.suptitle(f"{metric.capitalize()} across projects "
                     f"({N_REPEATS} runs each)", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"boxplot_{metric}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")


def plot_mean_bar(all_results, out_dir):
    projects = [p for p in PROJECTS if all_results.get(p) is not None]
    x = np.arange(len(projects))
    width = 0.35

    means_b = [all_results[p]["baseline"]["f1"].mean() for p in projects]
    means_p = [all_results[p]["proposed"]["f1"].mean()  for p in projects]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar(x - width/2, means_b, width,
                    label="Baseline (NB+TF-IDF)", color="#AEC6CF", edgecolor="black")
    bars_p = ax.bar(x + width/2, means_p, width,
                    label="Proposed (SVM+TF-IDF)", color="#FFD1A9", edgecolor="black")

    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Mean F1 Score: Baseline vs. Proposed")
    ax.set_xticks(x)
    ax.set_xticklabels(projects)
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "mean_f1_bar.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


#Summary Table

def print_summary_table(all_results):
    rows = []
    for project in PROJECTS:
        if all_results.get(project) is None:
            continue
        for metric in ["precision", "recall", "f1"]:
            data_b = all_results[project]["baseline"][metric]
            data_p = all_results[project]["proposed"][metric]
            p_val  = wilcoxon_test(data_b, data_p)
            vd     = vargha_delaney(data_p, data_b)
            rows.append({
                "Project":   project,
                "Metric":    metric.capitalize(),
                "Baseline (mean±std)": f"{data_b.mean():.3f}±{data_b.std():.3f}",
                "Proposed (mean±std)": f"{data_p.mean():.3f}±{data_p.std():.3f}",
                "p-value":   f"{p_val:.4f}",
                "VD-A":      f"{vd:.3f}",
                "Sig.":      "✓" if p_val < 0.05 else "✗",
            })

    df = pd.DataFrame(rows)
    print("\n" + "="*90)
    print(df.to_string(index=False))
    print("="*90)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing project CSVs")
    parser.add_argument("--out_dir", default="./results",
                        help="Output directory for figures and CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for project in PROJECTS:
        print(f"\n[{project}]")
        result = run_project(project, args.data_dir)
        all_results[project] = result

    print("\nSummary")
    df_summary = print_summary_table(all_results)
    csv_path = os.path.join(args.out_dir, "results_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    print("\nGenerating figures")
    plot_boxplots(all_results, args.out_dir)
    plot_mean_bar(all_results, args.out_dir)

    #Save arrays for replication
    raw = {}
    for project in PROJECTS:
        if all_results.get(project) is None:
            continue
        raw[project] = {}
        for approach in ["baseline", "proposed"]:
            raw[project][approach] = {
                m: all_results[project][approach][m].tolist()
                for m in ["precision", "recall", "f1"]
            }
    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw results saved to {args.out_dir}/raw_results.json")
if __name__ == "__main__":
    main()
