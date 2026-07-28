import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.csv_dataset import CSVDataset
from src.model.network import Branch, Merger
from src.utils.create_model import create_model
from src.utils.metrics import auroc as score_fn
from src.utils.train_model import train_model

BRANCHES = [
    "branch/frequency",
    "branch/frequency-paper",
    "branch/lp",
    "branch/pattern",
    "branch/pattern-paper",
]

MERGERS = [
    "merger/frequency+pattern+lp",
    "merger/lp+frequency",
    "merger/lp+pattern",
    "merger/viraminer",
    "merger/viraminer-paper",
]

BRANCH_KEY_MAP = {
    "Frequency Branch": "branch/frequency",
    "Frequency-paper Branch": "branch/frequency-paper",
    "LP Branch": "branch/lp",
    "Pattern Branch": "branch/pattern",
    "Pattern-paper Branch": "branch/pattern-paper",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FullE Complete Ensemble Pipeline (10 Models x Config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["A", "B", "C"],
        help="Training configuration strategy (A, B, C)",
    )
    parser.add_argument(
        "--viraminer_folder",
        default="dataset",
        help="Path to native fullset_train/validation/test.csv files",
    )
    parser.add_argument(
        "--data_folder",
        "-d",
        default="/home/simionricc/datasets",
        help="Root directory for external benchmark datasets",
    )
    parser.add_argument(
        "--template_dir",
        "-t",
        default="ready_to_train_files/final/onehot",
        help="Path to JSON model configuration templates",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="results_fulle_ensemble",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--logits_dir",
        default=None,
        help="Target directory for output logit text files",
    )
    return parser.parse_args()


# ========================================== #
# SEED FOR REPRODUCIBILITY                   #
# ========================================== #
def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========================================== #
# TENSOR-LEVEL DATA AUGMENTATION             #
# ========================================== #
def apply_tensor_augmentation(X, augment_type):
    """Applies augmentation directly on One-Hot encoded PyTorch tensors."""
    if augment_type == "none":
        return X

    X = X.clone()
    B = X.shape[0]

    n_channels = X.shape[1] if X.shape[1] in (4, 5) else X.shape[2]
    channel_first = X.shape[1] in (4, 5)
    rc_perm = [3, 2, 1, 0, 4] if n_channels == 5 else [3, 2, 1, 0]

    if channel_first:
        L = X.shape[2]
        if augment_type == "rc_shift":
            for i in range(B):
                if np.random.rand() > 0.5:
                    X[i] = torch.flip(X[i], dims=[1])
                    X[i] = X[i][rc_perm, :]
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    X[i] = torch.roll(X[i], shifts=shift, dims=1)
        elif augment_type == "mutation":
            for i in range(B):
                pos = np.random.choice(L, min(40, L), replace=False)
                for p in pos:
                    current_channel = torch.argmax(X[i, :, p]).item()
                    if current_channel >= 4:
                        continue
                    new_channel = np.random.choice(
                        [c for c in range(4) if c != current_channel]
                    )
                    X[i, :, p] = 0.0
                    X[i, new_channel, p] = 1.0
    else:
        L = X.shape[1]
        if augment_type == "rc_shift":
            for i in range(B):
                if np.random.rand() > 0.5:
                    X[i] = torch.flip(X[i], dims=[0])
                    X[i] = X[i][:, rc_perm]
                shift = np.random.randint(-5, 6)
                if shift != 0:
                    X[i] = torch.roll(X[i], shifts=shift, dims=0)
        elif augment_type == "mutation":
            for i in range(B):
                pos = np.random.choice(L, min(40, L), replace=False)
                for p in pos:
                    current_channel = torch.argmax(X[i, p, :]).item()
                    if current_channel >= 4:
                        continue
                    new_channel = np.random.choice(
                        [c for c in range(4) if c != current_channel]
                    )
                    X[i, p, :] = 0.0
                    X[i, p, new_channel] = 1.0
    return X


class AugmentedDataLoader:
    """DataLoader wrapper for online tensor-level data augmentation."""
    def __init__(self, dataloader, augment_type):
        self.dataloader = dataloader
        self.augment_type = augment_type

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)):
                X_aug = apply_tensor_augmentation(batch[0], self.augment_type)
                yield (X_aug, batch[1])
            else:
                yield batch


# ========================================== #
# LOSS FUNCTIONS                             #
# ========================================== #
class ProbabilityFocalLoss(nn.Module):
    """Focal Loss operating on predicted probabilities."""
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, probs, targets):
        targets = targets.view_as(probs)
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        loss = (
            -self.alpha
            * ((1 - probs) ** self.gamma)
            * targets_smooth
            * torch.log(probs)
            - (1 - self.alpha)
            * (probs ** self.gamma)
            * (1 - targets_smooth)
            * torch.log(1 - probs)
        )
        return loss.mean()


class SafeBCELoss(nn.Module):
    """Safe Binary Cross-Entropy loss wrapper."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        return self.bce(probs, targets.view_as(probs).float())


# ========================================== #
# DATASET LOADING HELPERS                    #
# ========================================== #
def _peek_columns(path):
    """Detects header and column structure of CSV files."""
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(",")

    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not _is_number(fields[-1])
    return len(fields), has_header


def _normalize_to_temp_csv(path):
    """Normalizes dataset layout to standard temporary CSV format."""
    ncols, has_header = _peek_columns(path)
    df = pd.read_csv(path) if has_header else pd.read_csv(path, header=None)
    if df.shape[1] == 2:
        seqs = df.iloc[:, 0].astype(str).values
        labels = df.iloc[:, 1].astype(float).values
    else:
        seqs = df.iloc[:, 1].astype(str).values
        labels = df.iloc[:, 2].astype(float).values

    seqs = np.array([s.upper() for s in seqs])
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"norm_{os.path.basename(path)}"
    )
    pd.DataFrame({0: range(len(seqs)), 1: seqs, 2: labels}).to_csv(
        tmp_path, header=False, index=False
    )
    return tmp_path


def make_eval_loader(csv_path, normalize=False):
    """Creates evaluation DataLoader with One-Hot encoding."""
    if normalize:
        csv_path = _normalize_to_temp_csv(csv_path)
    ds = CSVDataset(csv_path, transform="onehot_encoding")
    return ds.get_dataloader(batch_size=128, shuffle=False)


def _find_single_csv(folder, label):
    """Finds test CSV file in directory."""
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(all_csvs) == 0:
        print(f"[SKIP] '{label}': No CSV found in {folder}.")
        return None
    test_csvs = [f for f in all_csvs if "test" in os.path.basename(f).lower()]
    if test_csvs:
        return test_csvs[0]
    return all_csvs[0]


def _find_json_config(folder_path):
    """Finds JSON configuration file for model definitions."""
    json_files = sorted(list(Path(folder_path).glob("*.json")))
    if not json_files:
        raise FileNotFoundError(
            f"No .json configuration file found in: {folder_path}"
        )
    return json_files[0]


def _sanitize_submodel_name(rel_path):
    """Sanitizes model relative paths for output filenames."""
    return rel_path.replace("/", "_")


# ========================================== #
# MAIN PIPELINE                              #
# ========================================== #
def main():
    args = parse_args()
    seed_everything(2020)

    CONFIG_MAP = {
        "A": {
            "epochs": 120,
            "dropout": 0.25,
            "loss": "focal",
            "optimizer": "adamw",
            "lr_sched": "cosine",
            "augment": "rc_shift",
        },
        "B": {
            "epochs": 30,
            "dropout": 0.2,
            "loss": "bce",
            "optimizer": "adam",
            "lr_sched": "step",
            "augment": "none",
        },
        "C": {
            "epochs": 60,
            "dropout": 0.1,
            "loss": "bce",
            "optimizer": "adam",
            "lr_sched": "aiayn",
            "augment": "mutation",
        },
    }
    cfg = CONFIG_MAP[args.config]

    cfg_ckpt_dir = Path(args.output_dir) / f"config_{args.config}"
    os.makedirs(cfg_ckpt_dir, exist_ok=True)

    if args.logits_dir is not None:
        LOCAL_LOGITS_DIR = args.logits_dir
    else:
        LOCAL_LOGITS_DIR = f"./global_logits_temp_{args.config}"

    os.makedirs(LOCAL_LOGITS_DIR, exist_ok=True)

    train_file = os.path.join(args.viraminer_folder, "fullset_train.csv")
    valid_file = os.path.join(args.viraminer_folder, "fullset_validation.csv")
    test_file = os.path.join(args.viraminer_folder, "fullset_test.csv")

    train_df = CSVDataset(f"{train_file}", transform="onehot_encoding")
    valid_df = CSVDataset(f"{valid_file}", transform="onehot_encoding")
    test_df = CSVDataset(f"{test_file}", transform="onehot_encoding")

    raw_train_dl = train_df.get_dataloader(batch_size=128, shuffle=True)
    aug_train_dl = AugmentedDataLoader(raw_train_dl, cfg["augment"])

    dl_valid = valid_df.get_dataloader(batch_size=128, shuffle=False)
    dl_test = test_df.get_dataloader(batch_size=128, shuffle=False)

    trained_model_paths = {}

    def train_one_network(model_infos, save_path, is_merger=False):
        """Helper to train a single branch or merger network."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if "parameters" in model_infos:
            model_infos["parameters"]["dropout_prob"] = cfg["dropout"]

        model = create_model(model_infos)

        # Freeze branch parameters during merger training
        if is_merger and isinstance(model, Merger):
            model.set_branches_trainability(False)
            trainable_names = [
                name for name, p in model.named_parameters() if p.requires_grad
            ]
            print(
                f"    [CHECK Merger Freezing] Trainable parameters: {trainable_names}"
            )

        loss_fn = (
            ProbabilityFocalLoss() if cfg["loss"] == "focal" else SafeBCELoss()
        )

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        if cfg["lr_sched"] == "aiayn":
            lr_base = 1.0
        elif cfg["optimizer"] == "adamw":
            lr_base = 1e-4
        else:
            lr_base = 0.001

        if cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params, lr=lr_base, weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=lr_base)

        if cfg["lr_sched"] == "cosine":

            def lr_lambda(epoch):
                if epoch < 5:
                    return (epoch + 1) / 5
                return 0.5 * (
                    1 + math.cos(math.pi * (epoch - 5) / (cfg["epochs"] - 5))
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        elif cfg["lr_sched"] == "aiayn":
            steps_per_epoch = len(raw_train_dl)

            def aiayn_epoch_lr(epoch):
                step = max(1, (epoch + 1) * steps_per_epoch)
                d_model = 512
                warmup_steps = 4000
                return (d_model**-0.5) * min(step**-0.5, step * (warmup_steps**-1.5))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, aiayn_epoch_lr)

        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )

        current_train_dl = raw_train_dl if is_merger else aug_train_dl

        epsilon = model_infos.get("epsilon", 0)
        train_model(
            model,
            train_dl=current_train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg["epochs"],
            epsilon=epsilon,
            min_score=0,
            validation_fn=score_fn,
            val_dl=dl_valid,
            verbose=0,
        )

        model.save_model(save_path)
        return save_path

    # -------------------------------------------------------------------------
    # STEP 1: Train or retrieve 5 Branch networks
    # -------------------------------------------------------------------------
    print(f"\n[>>>] STEP 1: Training / Loading 5 Branch Submodels for Config {args.config}")
    for branch_rel in BRANCHES:
        folder_path = Path(args.template_dir) / branch_rel
        json_path = _find_json_config(folder_path)
        model_infos = json.loads(json_path.read_text())
        if isinstance(model_infos, list):
            model_infos = model_infos[0]

        save_ptm = cfg_ckpt_dir / branch_rel / "model.ptm"
        if save_ptm.exists():
            print(f"  [=] Branch '{branch_rel}' already trained. Skipping training.")
        else:
            print(f"  [+] Training Branch: {branch_rel}...")
            train_one_network(model_infos, str(save_ptm), is_merger=False)

        trained_model_paths[branch_rel] = str(save_ptm)

    # -------------------------------------------------------------------------
    # STEP 2: Train or retrieve 5 Merger networks
    # -------------------------------------------------------------------------
    print(f"\n[>>>] STEP 2: Training / Loading 5 Merger Networks for Config {args.config}")
    for merger_rel in MERGERS:
        folder_path = Path(args.template_dir) / merger_rel
        json_path = _find_json_config(folder_path)
        model_infos = json.loads(json_path.read_text())
        if isinstance(model_infos, list):
            model_infos = model_infos[0]

        branches_dict = model_infos["parameters"]["branches_dict"]
        updated_dict = {}
        for b_key in branches_dict.keys():
            target_rel = BRANCH_KEY_MAP[b_key]
            updated_dict[b_key] = trained_model_paths[target_rel]
        model_infos["parameters"]["branches_dict"] = updated_dict

        save_ptm = cfg_ckpt_dir / merger_rel / "model.ptm"
        if save_ptm.exists():
            print(f"  [=] Merger '{merger_rel}' already trained. Skipping training.")
        else:
            print(f"  [+] Training Merger: {merger_rel}...")
            train_one_network(model_infos, str(save_ptm), is_merger=True)

        trained_model_paths[merger_rel] = str(save_ptm)

    # -------------------------------------------------------------------------
    # STEP 3: Logit Extraction & Sum-Rule Ensemble
    # -------------------------------------------------------------------------
    print(f"\n[>>>] STEP 3: Logit Extraction and Ensemble for Config {args.config}")

    datasets_eval = {"val": dl_valid, "test_VM": dl_test}

    _ecoli_csv = _find_single_csv(os.path.join(args.data_folder, "ecoli"), "test_ecoli")
    if _ecoli_csv:
        datasets_eval["test_ecoli"] = make_eval_loader(_ecoli_csv, normalize=True)

    _enhancer_csv = _find_single_csv(os.path.join(args.data_folder, "enhancer"), "test_enhancer")
    if _enhancer_csv:
        datasets_eval["test_enhancer"] = make_eval_loader(_enhancer_csv, normalize=True)

    _anell_test = os.path.join(args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv")
    if os.path.exists(_anell_test):
        datasets_eval["test_anellovirus"] = make_eval_loader(_anell_test, normalize=True)

    _sim_base_dir = os.path.join(args.data_folder, "SIM")
    _sim_files = sorted(glob.glob(os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True))
    for _f in _sim_files:
        _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
        _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
        datasets_eval[_key] = make_eval_loader(_f, normalize=True)

    num_models = len(trained_model_paths)

    for dataset_name, dl in datasets_eval.items():
        print(f"  [>] Processing Dataset: {dataset_name}...")

        ensemble_probs_sum = None
        y_true_list = []
        first_model_key = list(trained_model_paths.keys())[0]

        for rel_path, ptm_path in trained_model_paths.items():
            if "merger" in rel_path:
                model = Merger(ptm_path)
            else:
                model = Branch(ptm_path)

            model.eval()
            # Device configuration to prevent CPU/CUDA tensors mismatch
            model.set_best_available_device()

            model_logits = []
            model_probs = []
            with torch.no_grad():
                for batch in dl:
                    X = batch[0].to(model.device)
                    y = batch[1]

                    pred, logit = model(X)
                    model_probs.extend(pred.cpu().numpy().flatten())
                    model_logits.extend(logit.cpu().numpy().flatten())

                    if rel_path == first_model_key:
                        y_true_list.extend(y.numpy().flatten())

            model_probs = np.array(model_probs)
            model_logits = np.array(model_logits)

            # Save individual submodel logits
            submodel_tag = _sanitize_submodel_name(rel_path)
            submodel_filename = f"fulle_cfg{args.config}_{submodel_tag}_{dataset_name}.txt"
            np.savetxt(
                os.path.join(LOCAL_LOGITS_DIR, submodel_filename),
                model_logits,
                fmt="%.6f",
            )

            if ensemble_probs_sum is None:
                ensemble_probs_sum = model_probs.copy()
            else:
                ensemble_probs_sum += model_probs

            model.set_to_cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save Sum-Rule ensemble predictions
        ensemble_filename = f"fulle_cfg{args.config}_sumrule_{dataset_name}.txt"
        save_file = os.path.join(LOCAL_LOGITS_DIR, ensemble_filename)
        np.savetxt(save_file, ensemble_probs_sum, fmt="%.6f")
        print(f"  [✓] Logits for {num_models} submodels + sum-rule ensemble saved for '{dataset_name}'")

        # Metric computation
        if len(y_true_list) == len(ensemble_probs_sum):
            y_true = np.array(y_true_list)
            probs_normalized = ensemble_probs_sum / num_models

            try:
                auc_score = score_fn(y_true, probs_normalized)[0]
                precision, recall, _ = precision_recall_curve(y_true, probs_normalized)
                auprc_score = auc(recall, precision)
                f1 = f1_score(y_true, (probs_normalized >= 0.5).astype(int))
                mcc = matthews_corrcoef(y_true, (probs_normalized >= 0.5).astype(int))

                print(
                    f"      Metrics ({dataset_name}): AUROC = {auc_score:.4f} | AUPRC = {auprc_score:.4f} | F1 = {f1:.4f} | MCC = {mcc:.4f}"
                )
            except Exception:
                pass


if __name__ == "__main__":
    main()