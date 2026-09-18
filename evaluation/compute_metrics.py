import os
import glob
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    matthews_corrcoef,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# Suppress division by zero warnings on edge-case thresholding
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

DATA_DIRS = [
    os.environ.get("DATA_DIR", "./datasets"),
    "./datasets",
    "../datasets"
]

PRIMARY_TESTS = ["test_ecoli", "test_enhancer", "test_anellovirus", "test_VM"]

PRIMARY_TASK_NAMES = {
    "test_ecoli": "E. coli",
    "test_enhancer": "Enhancer",
    "test_anellovirus": "Anellovirus",
    "test_VM": "DS-VM (Viral)"
}

ARCH_LABELS = {
    "genexplorer": "GenExplorer",
    "fulle": "FullE",
    "nucleictransformer": "Nucleic Transformer",
    "dnabert2camp": "DNABERT2-CAMP"
}

LABEL_CACHE = {}

def get_valid_data_root():
    for d in DATA_DIRS:
        if os.path.exists(d):
            return d
    return DATA_DIRS[0]

DATA_ROOT = get_valid_data_root()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

def find_dataset_file(test_set_name):
    """Locate test set CSV files using structured and recursive lookup."""
    direct_candidates = [
        # E. coli
        os.path.join(DATA_ROOT, "ecoli", "test_ecoli.csv"),
        os.path.join(DATA_ROOT, "ecoli", "ecoli_independent_test_NT.csv"),
        # Enhancer
        os.path.join(DATA_ROOT, "enhancer", "test_enhancer.csv"),
        os.path.join(DATA_ROOT, "enhancer", "bert_enhancer_ind_test_set.csv"),
        # Anellovirus
        os.path.join(DATA_ROOT, "anellovirus", "test_anellovirus.csv"),
        os.path.join(DATA_ROOT, "unseen_anell", "test_data_anell_300_binary.csv"),
        os.path.join(DATA_ROOT, "anellovirus", "test_data_anell_300_binary.csv"),
        # Viral VM
        os.path.join(DATA_ROOT, "viral", "test_VM.csv"),
        os.path.join(DATA_ROOT, "DNA_data", "fullset_test.csv"),
        os.path.join(DATA_ROOT, "viral", "fullset_test.csv"),
    ]

    if test_set_name == "test_ecoli":
        for c in direct_candidates[:2]:
            if os.path.exists(c): return c
    elif test_set_name == "test_enhancer":
        for c in direct_candidates[2:4]:
            if os.path.exists(c): return c
    elif test_set_name == "test_anellovirus":
        for c in direct_candidates[4:7]:
            if os.path.exists(c): return c
    elif test_set_name == "test_VM":
        for c in direct_candidates[7:]:
            if os.path.exists(c): return c

    # Noise pattern matching (SIM benchmarks)
    m = re.match(r"test_sim_noise_(\d+)_seed_(\d+)_fullset_test", test_set_name)
    if m:
        noise_lvl, seed = m.groups()
        noise_paths = [
            os.path.join(DATA_ROOT, "viral", "noise_tests", f"{test_set_name}.csv"),
            os.path.join(DATA_ROOT, "SIM", f"noise_{noise_lvl}", f"seed_{seed}", "fullset_test.csv"),
            os.path.join(DATA_ROOT, f"noise_{noise_lvl}", f"seed_{seed}", "fullset_test.csv")
        ]
        for np_path in noise_paths:
            if os.path.exists(np_path):
                return np_path

    # Recursive glob fallback
    target_clean = test_set_name.replace("test_", "")
    matches = glob.glob(f"{DATA_ROOT}/**/{test_set_name}.csv", recursive=True)
    if matches:
        return matches[0]
    matches_clean = glob.glob(f"{DATA_ROOT}/**/{target_clean}.csv", recursive=True)
    if matches_clean:
        return matches_clean[0]

    return None

def load_ground_truth(test_set_name):
    """Load ground-truth binary labels (0/1) preserving sample indices."""
    if test_set_name in LABEL_CACHE:
        return LABEL_CACHE[test_set_name]

    file_path = find_dataset_file(test_set_name)
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        df_head = pd.read_csv(file_path)
        first_col_name = str(df_head.columns[-1]).strip()

        if first_col_name.lower() in ['label', 'labels', 'target', 'class', 'y']:
            labels = df_head[first_col_name].values.astype(int)
        else:
            df_nohead = pd.read_csv(file_path, header=None)
            labels = df_nohead.iloc[:, -1].values.astype(int)

        LABEL_CACHE[test_set_name] = labels
        return labels
    except Exception:
        return None

def safe_mcc(y_true, y_pred):
    """Compute Matthews Correlation Coefficient with zero-division safety."""
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return 0.0
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            val = matthews_corrcoef(y_true, y_pred)
            return 0.0 if np.isnan(val) else float(val)
    except Exception:
        return 0.0

def compute_all_metrics(y_true, y_score):
    """Compute classification metrics with standard (0.5) and Youden's J optimal threshold."""
    if len(np.unique(y_true)) < 2:
        return {
            "AUROC": np.nan, "AUPRC": np.nan, "MCC": np.nan, "MCC_05": np.nan,
            "BACC": np.nan, "ACC": np.nan, "F1": np.nan, "Best_Thresh": 0.5
        }

    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)
    except Exception:
        auprc = np.nan

    # Youden's J threshold optimization (J = Sensitivity + Specificity - 1)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        if np.isnan(best_thresh) or np.isinf(best_thresh):
            best_thresh = 0.5
    except Exception:
        best_thresh = 0.5

    y_pred_opt = (y_score >= best_thresh).astype(int)
    y_pred_05 = (y_score >= 0.5).astype(int)

    mcc = safe_mcc(y_true, y_pred_opt)
    mcc_05 = safe_mcc(y_true, y_pred_05)

    try:
        bacc = balanced_accuracy_score(y_true, y_pred_opt)
    except Exception:
        bacc = np.nan

    try:
        acc = accuracy_score(y_true, y_pred_opt)
    except Exception:
        acc = np.nan

    try:
        f1 = f1_score(y_true, y_pred_opt, zero_division=0)
    except TypeError:
        try:
            f1 = f1_score(y_true, y_pred_opt)
        except Exception:
            f1 = np.nan
    except Exception:
        f1 = np.nan

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": mcc,
        "MCC_05": mcc_05,
        "BACC": bacc,
        "ACC": acc,
        "F1": f1,
        "Best_Thresh": best_thresh
    }

def analyze_directory(logits_dir, output_suffix=""):
    files = sorted(glob.glob(os.path.join(logits_dir, "*.txt")))
    if not files:
        print(f"[!] No logit files found in '{logits_dir}'.")
        return None, None

    records = []
    ensemble_data = {}
    pattern = re.compile(r"^([a-zA-Z0-9]+)_(cfg[A-Z])_([a-zA-Z0-9_]+)_(test_.+)\.txt$")

    for f in files:
        fname = os.path.basename(f)
        match = pattern.match(fname)
        if not match:
            continue

        model_name, cfg, train_task, test_set = match.groups()
        raw_logits = np.loadtxt(f)
        if raw_logits.ndim == 0:
            continue
        if raw_logits.ndim > 1:
            raw_logits = raw_logits[:, 0]

        scores = sigmoid(raw_logits)
        labels = load_ground_truth(test_set)

        if labels is None or len(scores) != len(labels):
            continue

        metrics = compute_all_metrics(labels, scores)
        model_display = ARCH_LABELS.get(model_name.lower(), model_name)

        records.append({
            "Architecture": model_display,
            "Config": cfg.replace("cfg", "Config "),
            "TrainTask": train_task,
            "TestSet": test_set,
            "AUROC": metrics["AUROC"],
            "AUPRC": metrics["AUPRC"],
            "MCC": metrics["MCC"],
            "MCC_05": metrics["MCC_05"],
            "BACC": metrics["BACC"],
            "ACC": metrics["ACC"],
            "F1": metrics["F1"],
            "Thresh": metrics["Best_Thresh"],
            "N_Samples": len(labels),
            "N_Pos": int(np.sum(labels))
        })

        key = (train_task, cfg, test_set)
        if key not in ensemble_data:
            ensemble_data[key] = {"labels": labels, "scores": {}, "raw_logits": {}}
        ensemble_data[key]["scores"][model_display] = scores
        ensemble_data[key]["raw_logits"][model_display] = raw_logits

    df_models = pd.DataFrame(records)

    # Inter-Architectural Soft Voting Ensemble
    ens_records = []
    for (train_task, cfg, test_set), payload in ensemble_data.items():
        labels = payload["labels"]
        model_scores = payload["scores"]
        all_s = list(model_scores.values())

        if not all_s or len(all_s) < 2:
            continue

        avg_score = np.mean(all_s, axis=0)
        ens_metrics = compute_all_metrics(labels, avg_score)

        ens_records.append({
            "Architecture": f"Ensemble_{len(all_s)}M (Soft Voting)",
            "Config": cfg.replace("cfg", "Config "),
            "TrainTask": train_task,
            "TestSet": test_set,
            "AUROC": ens_metrics["AUROC"],
            "AUPRC": ens_metrics["AUPRC"],
            "MCC": ens_metrics["MCC"],
            "MCC_05": ens_metrics["MCC_05"],
            "BACC": ens_metrics["BACC"],
            "ACC": ens_metrics["ACC"],
            "F1": ens_metrics["F1"],
            "Thresh": ens_metrics["Best_Thresh"],
            "N_Samples": len(labels),
            "N_Pos": int(np.sum(labels)),
            "Models_Included": "+".join(sorted(model_scores.keys()))
        })

    df_ens = pd.DataFrame(ens_records)
    combined_df = pd.concat([df_models, df_ens], ignore_index=True)

    out_all = f"paper_metrics_all{output_suffix}.csv"
    out_primary = f"paper_metrics_primary{output_suffix}.csv"
    out_ens = f"paper_metrics_ensemble{output_suffix}.csv"

    combined_df.to_csv(out_all, index=False)
    df_ens.to_csv(out_ens, index=False)

    primary_df = combined_df[combined_df["TestSet"].isin(PRIMARY_TESTS)].copy()
    primary_df.to_csv(out_primary, index=False)

    return combined_df, primary_df

def main():
    parser = argparse.ArgumentParser(description="Performance Evaluation and Modular vs Monolithic Analysis")
    parser.add_argument("--dir_mod", type=str, default="global_logits", help="Modular logits directory")
    parser.add_argument("--dir_mono", type=str, default="global_logits_monolithic", help="Monolithic logits directory")
    args = parser.parse_args()

    print(f"[*] Identified Dataset Root: {DATA_ROOT}")

    # =========================================================================
    # SECTION 1: MODULAR LOGITS EVALUATION
    # =========================================================================
    print("\n" + "=" * 115)
    print(f"{'1. MODULAR BENCHMARK EVALUATION (global_logits)':^115}")
    print("=" * 115)
    combined_mod, primary_mod = analyze_directory(args.dir_mod, output_suffix="_modular")

    if primary_mod is not None:
        p_disp = primary_mod.sort_values(by=["TrainTask", "TestSet", "Config", "Architecture"]).copy()
        for col in ["AUROC", "AUPRC", "MCC", "BACC", "F1"]:
            p_disp[col] = p_disp[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        print(p_disp[["TrainTask", "TestSet", "Config", "Architecture", "AUROC", "AUPRC", "MCC", "BACC", "F1"]].to_string(index=False))

    # =========================================================================
    # SECTION 2: MONOLITHIC LOGITS EVALUATION
    # =========================================================================
    print("\n" + "=" * 115)
    print(f"{'2. MONOLITHIC BENCHMARK EVALUATION (global_logits_monolithic)':^115}")
    print("=" * 115)
    combined_mono, primary_mono = analyze_directory(args.dir_mono, output_suffix="_monolithic")

    if primary_mono is not None:
        p_disp_mono = primary_mono.sort_values(by=["TrainTask", "TestSet", "Config", "Architecture"]).copy()
        for col in ["AUROC", "AUPRC", "MCC", "BACC", "F1"]:
            p_disp_mono[col] = p_disp_mono[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        print(p_disp_mono[["TrainTask", "TestSet", "Config", "Architecture", "AUROC", "AUPRC", "MCC", "BACC", "F1"]].to_string(index=False))

    # =========================================================================
    # SECTION 3: MODULAR VS MONOLITHIC COMPARISON (DELTA ANALYSIS)
    # =========================================================================
    if primary_mod is not None and primary_mono is not None:
        print("\n" + "=" * 115)
        print(f"{'3. MODULAR VS MONOLITHIC COMPARISON (AUROC DELTA = MODULAR - MONOLITHIC)':^115}")
        print("=" * 115)
        merged_comp = pd.merge(
            primary_mod[["TrainTask", "TestSet", "Config", "Architecture", "AUROC"]],
            primary_mono[["TrainTask", "TestSet", "Config", "Architecture", "AUROC"]],
            on=["TrainTask", "TestSet", "Config", "Architecture"],
            suffixes=("_Modular", "_Monolithic")
        )
        merged_comp["Delta_AUROC"] = merged_comp["AUROC_Modular"] - merged_comp["AUROC_Monolithic"]
        merged_comp.to_csv("paper_metrics_comparison_delta.csv", index=False)
        
        comp_disp = merged_comp.sort_values(by=["TrainTask", "TestSet", "Config", "Architecture"]).copy()
        for col in ["AUROC_Modular", "AUROC_Monolithic", "Delta_AUROC"]:
            comp_disp[col] = comp_disp[col].apply(lambda x: f"{x:+.4f}" if pd.notnull(x) else "N/A")
        print(comp_disp[["TrainTask", "TestSet", "Config", "Architecture", "AUROC_Modular", "AUROC_Monolithic", "Delta_AUROC"]].to_string(index=False))

    # =========================================================================
    # SECTION 4: SEQUENCING NOISE ROBUSTNESS
    # =========================================================================
    if combined_mod is not None:
        noise_df = combined_mod[combined_mod["TestSet"].str.contains("noise", case=False)].copy()
        if not noise_df.empty:
            noise_df["Noise_Level"] = noise_df["TestSet"].apply(
                lambda x: re.search(r"noise_(\d+)", x).group(0) if re.search(r"noise_(\d+)", x) else "unknown"
            )
            noise_summary = noise_df.groupby(["Noise_Level", "Config", "Architecture"])[
                ["AUROC", "AUPRC", "MCC", "BACC", "F1"]
            ].agg(["mean", "std"]).reset_index()

            noise_summary.columns = ['_'.join(c).strip('_') for c in noise_summary.columns]
            noise_summary.to_csv("paper_metrics_noise_summary.csv", index=False)

            print("\n" + "=" * 115)
            print(f"{'4. SEQUENCING NOISE ROBUSTNESS (MEAN ± STD OVER 5 SEEDS)':^115}")
            print("=" * 115)
            for m in ["AUROC", "AUPRC", "MCC", "BACC", "F1"]:
                noise_summary[m] = noise_summary.apply(
                    lambda r: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}", axis=1
                )
            print(noise_summary[["Noise_Level", "Config", "Architecture", "AUROC", "AUPRC", "MCC", "BACC", "F1"]].sort_values(by=["Noise_Level", "Config", "Architecture"]).to_string(index=False))

    print("\n[+] Metric extraction completed successfully.")
    print("    - paper_metrics_all_modular.csv / paper_metrics_all_monolithic.csv")
    print("    - paper_metrics_primary_modular.csv / paper_metrics_primary_monolithic.csv")
    print("    - paper_metrics_comparison_delta.csv")
    print("    - paper_metrics_noise_summary.csv")

if __name__ == "__main__":
    main()