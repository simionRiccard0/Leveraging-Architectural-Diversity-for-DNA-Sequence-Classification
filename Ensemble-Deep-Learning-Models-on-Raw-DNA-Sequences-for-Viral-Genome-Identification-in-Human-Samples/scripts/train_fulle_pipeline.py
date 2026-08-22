import argparse
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
from sklearn.metrics import (
    auc,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)
import torch
import torch.nn as nn

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
        help="Configurazione di training (A, B, C)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="viral",
        choices=["viral", "ecoli", "enhancer", "anellovirus"],
        help="Task biologico (default: viral)",
    )
    parser.add_argument(
        "--viraminer_folder",
        default="dataset",
        help="Path ai file nativi fullset_train/validation/test.csv",
    )
    parser.add_argument(
        "--data_folder",
        "-d",
        default="/home/simionricc/datasets",
        help="Path alla cartella principale dei dataset",
    )
    parser.add_argument(
        "--template_dir",
        "-t",
        default="ready_to_train_files/final/onehot",
        help="Path ai JSON template originali",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="results_fulle_ensemble",
        help="Directory di output per i checkpoint .ptm",
    )
    return parser.parse_args()


def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_tensor_augmentation(X, augment_type):
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
                pos = np.random.choice(
                    L, min(40, max(1, int(L * 0.13))), replace=False
                )
                for p in pos:
                    curr = torch.argmax(X[i, :, p]).item()
                    if curr >= 4:
                        continue
                    new_c = np.random.choice([c for c in range(4) if c != curr])
                    X[i, :, p] = 0.0
                    X[i, new_c, p] = 1.0
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
                pos = np.random.choice(
                    L, min(40, max(1, int(L * 0.13))), replace=False
                )
                for p in pos:
                    curr = torch.argmax(X[i, p, :]).item()
                    if curr >= 4:
                        continue
                    new_c = np.random.choice([c for c in range(4) if c != curr])
                    X[i, p, :] = 0.0
                    X[i, p, new_c] = 1.0
    return X


class AugmentedDataLoader:
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
                yield (
                    apply_tensor_augmentation(batch[0], self.augment_type),
                    batch[1],
                )
            else:
                yield batch


class ProbabilityFocalLoss(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        return self.bce(probs, targets.view_as(probs).float())


def _peek_columns(path):
    with open(path, "r") as fh:
        first_line = fh.readline().strip()
    fields = first_line.split(",")
    has_header = not any(c.isdigit() for c in fields[-1])
    return len(fields), has_header


def _normalize_to_temp_csv(path):
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
    if normalize:
        csv_path = _normalize_to_temp_csv(csv_path)
    ds = CSVDataset(csv_path, transform="onehot_encoding")
    return ds.get_dataloader(batch_size=128, shuffle=False)


def _resolve_test_file(folder, default_name):
    target = os.path.join(folder, default_name)
    if os.path.exists(target):
        return target
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    test_csvs = [f for f in all_csvs if "test" in os.path.basename(f).lower()]
    return test_csvs[0] if test_csvs else (all_csvs[0] if all_csvs else None)


def _find_json_config(folder_path):
    json_files = sorted(list(Path(folder_path).glob("*.json")))
    if not json_files:
        raise FileNotFoundError(f"Nessun .json in: {folder_path}")
    return json_files[0]


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

    cfg_ckpt_dir = (
        Path(args.output_dir) / f"{args.task}" / f"config_{args.config}"
    )
    os.makedirs(cfg_ckpt_dir, exist_ok=True)
    GLOBAL_LOGITS_DIR = "/home/simionricc/global_logits"
    os.makedirs(GLOBAL_LOGITS_DIR, exist_ok=True)

    if args.task == "viral":
        train_file = os.path.join(args.viraminer_folder, "fullset_train.csv")
        valid_file = os.path.join(
            args.viraminer_folder, "fullset_validation.csv"
        )
        if not os.path.exists(train_file):
            train_file = os.path.join(
                args.data_folder, "DNA_data", "fullset_train.csv"
            )
            valid_file = os.path.join(
                args.data_folder, "DNA_data", "fullset_validation.csv"
            )
        if not os.path.exists(train_file):
            train_file = os.path.join(args.data_folder, "fullset_train.csv")
            valid_file = os.path.join(
                args.data_folder, "fullset_validation.csv"
            )
        norm_train = _normalize_to_temp_csv(train_file)
        norm_val = _normalize_to_temp_csv(valid_file)

    else:
        # Gestione unificata e robusta di E. coli, Enhancer e Anellovirus
        if args.task == "anellovirus":
            raw_path = os.path.join(
                args.data_folder, "unseen_anell", "training_data_anell_300_binary.csv"
            )
        elif args.task == "ecoli":
            raw_path = os.path.join(
                args.data_folder, "ecoli", "ecoli_train_NT.csv"
            )
        elif args.task == "enhancer":
            raw_path = os.path.join(
                args.data_folder, "enhancer", "bert_enhancer_dataset.csv"
            )

        ncols, has_header = _peek_columns(raw_path)
        df_raw = pd.read_csv(raw_path) if has_header else pd.read_csv(raw_path, header=None)
        if df_raw.shape[1] == 2:
            seqs = df_raw.iloc[:, 0].astype(str).values
            labels = df_raw.iloc[:, 1].astype(float).values
        else:
            seqs = df_raw.iloc[:, 1].astype(str).values
            labels = df_raw.iloc[:, 2].astype(float).values

        seqs = np.array([s.upper() for s in seqs])
        full_df = pd.DataFrame({0: range(len(seqs)), 1: seqs, 2: labels})
        full_df = full_df.sample(frac=1, random_state=2020).reset_index(drop=True)
        full_df[0] = range(len(full_df))  # Re-indicizzazione esatta 3 colonne

        n_val = int(len(full_df) * 0.2)
        norm_val = os.path.join(
            tempfile.gettempdir(), f"val_{args.task}_{args.config}.csv"
        )
        norm_train = os.path.join(
            tempfile.gettempdir(), f"train_{args.task}_{args.config}.csv"
        )
        full_df.iloc[:n_val].to_csv(norm_val, header=False, index=False)
        full_df.iloc[n_val:].to_csv(norm_train, header=False, index=False)

    train_df = CSVDataset(norm_train, transform="onehot_encoding")
    valid_df = CSVDataset(norm_val, transform="onehot_encoding")

    raw_train_dl = train_df.get_dataloader(batch_size=128, shuffle=True)
    aug_train_dl = AugmentedDataLoader(raw_train_dl, cfg["augment"])
    dl_valid = valid_df.get_dataloader(batch_size=128, shuffle=False)

    print(
        f"[i] FullE | Task: '{args.task}' | Config: '{args.config}' | Train: {len(train_df)} | Val: {len(valid_df)}",
        flush=True,
    )

    trained_model_paths = {}

    def train_one_network(model_infos, save_path, is_merger=False):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if "parameters" in model_infos:
            model_infos["parameters"]["dropout_prob"] = cfg["dropout"]

        model = create_model(model_infos)
        if is_merger and isinstance(model, Merger):
            model.set_branches_trainability(False)

        loss_fn = (
            ProbabilityFocalLoss() if cfg["loss"] == "focal" else SafeBCELoss()
        )
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        lr_base = (
            1.0
            if cfg["lr_sched"] == "aiayn"
            else (1e-4 if cfg["optimizer"] == "adamw" else 0.001)
        )
        optimizer = (
            torch.optim.AdamW(trainable_params, lr=lr_base, weight_decay=1e-4)
            if cfg["optimizer"] == "adamw"
            else torch.optim.Adam(trainable_params, lr=lr_base)
        )

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
                return (d_model**-0.5) * min(
                    step**-0.5, step * (warmup_steps**-1.5)
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, aiayn_epoch_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )

        train_model(
            model,
            train_dl=(raw_train_dl if is_merger else aug_train_dl),
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=cfg["epochs"],
            epsilon=model_infos.get("epsilon", 0),
            min_score=0,
            validation_fn=score_fn,
            val_dl=dl_valid,
            verbose=0,
        )

        model.save_model(save_path)
        return save_path

    # Step 1: Branches
    for branch_rel in BRANCHES:
        folder_path = Path(args.template_dir) / branch_rel
        json_path = _find_json_config(folder_path)
        model_infos = json.loads(json_path.read_text())
        if isinstance(model_infos, list):
            model_infos = model_infos[0]
        save_ptm = cfg_ckpt_dir / branch_rel / "model.ptm"
        if not save_ptm.exists():
            train_one_network(model_infos, str(save_ptm), is_merger=False)
        trained_model_paths[branch_rel] = str(save_ptm)

    # Step 2: Mergers
    for merger_rel in MERGERS:
        folder_path = Path(args.template_dir) / merger_rel
        json_path = _find_json_config(folder_path)
        model_infos = json.loads(json_path.read_text())
        if isinstance(model_infos, list):
            model_infos = model_infos[0]
        for b_key in model_infos["parameters"]["branches_dict"].keys():
            target_rel = BRANCH_KEY_MAP[b_key]
            model_infos["parameters"]["branches_dict"][b_key] = (
                trained_model_paths[target_rel]
            )
        save_ptm = cfg_ckpt_dir / merger_rel / "model.ptm"
        if not save_ptm.exists():
            train_one_network(model_infos, str(save_ptm), is_merger=True)
        trained_model_paths[merger_rel] = str(save_ptm)

    # Step 3: Targeted Cascade Evaluation (Solo i test set rilevanti)
    print("\n--- Targeted Test Evaluation ---", flush=True)
    datasets_eval = {}

    if args.task == "viral":
        vm_test = os.path.join(args.data_folder, "fullset_test.csv")
        if not os.path.exists(vm_test):
            vm_test = os.path.join(args.data_folder, "DNA_data", "fullset_test.csv")
        if not os.path.exists(vm_test):
            vm_test = os.path.join(args.viraminer_folder, "fullset_test.csv")
        if os.path.exists(vm_test):
            datasets_eval["test_VM"] = make_eval_loader(vm_test, normalize=True)

        _sim_base_dir = os.path.join(args.data_folder, "SIM")
        if os.path.exists(_sim_base_dir):
            for _f in sorted(
                glob.glob(
                    os.path.join(_sim_base_dir, "**", "*.csv"), recursive=True
                )
            ):
                _rel_noext = os.path.splitext(os.path.relpath(_f, _sim_base_dir))[0]
                _key = "test_sim_" + _rel_noext.replace(os.sep, "_")
                datasets_eval[_key] = make_eval_loader(_f, normalize=True)

    elif args.task == "anellovirus":
        anell_test = os.path.join(
            args.data_folder, "unseen_anell", "test_data_anell_300_binary.csv"
        )
        if os.path.exists(anell_test):
            datasets_eval["test_anellovirus"] = make_eval_loader(
                anell_test, normalize=True
            )

    elif args.task == "ecoli":
        ecoli_test = _resolve_test_file(
            os.path.join(args.data_folder, "ecoli"), "ecoli_independent_test_NT.csv"
        )
        if ecoli_test:
            datasets_eval["test_ecoli"] = make_eval_loader(
                ecoli_test, normalize=True
            )

    elif args.task == "enhancer":
        enh_test = _resolve_test_file(
            os.path.join(args.data_folder, "enhancer"),
            "bert_enhancer_ind_test_set.csv",
        )
        if enh_test:
            datasets_eval["test_enhancer"] = make_eval_loader(
                enh_test, normalize=True
            )

    for dataset_name, dl in datasets_eval.items():
        ensemble_probs_sum = None
        for rel_path, ptm_path in trained_model_paths.items():
            model = Merger(ptm_path) if "merger" in rel_path else Branch(ptm_path)
            model.eval()
            model.set_best_available_device()

            model_probs = []
            with torch.no_grad():
                for batch in dl:
                    pred, _ = model(batch[0].to(model.device))
                    model_probs.extend(pred.cpu().numpy().flatten())

            model_probs = np.array(model_probs)

            ensemble_probs_sum = (
                model_probs.copy()
                if ensemble_probs_sum is None
                else ensemble_probs_sum + model_probs
            )
            model.set_to_cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ens_name = f"fulle_cfg{args.config}_{args.task}_{dataset_name}.txt"
        np.savetxt(
            os.path.join(GLOBAL_LOGITS_DIR, ens_name),
            ensemble_probs_sum,
            fmt="%.6f",
        )
        print(
            f" Logits saved ({len(ensemble_probs_sum)} samples): {ens_name}",
            flush=True,
        )


if __name__ == "__main__":
    main()