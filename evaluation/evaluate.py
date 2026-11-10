#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.genexplorer import GenExplorer
from models.nucleic_transformer import NucleicTransformer
from models.fulle import build_fulle_10ensemble
from models.helixpr import HelixPR

from utils.data_utils import (
    seed_everything,
    resolve_task_splits,
    SequenceClassificationDataset,
    load_seq_label_dataframe
)


def load_fulle_weights(model, ckpt):
    """
    Loads FullE weights from either a model_state_dict or the flat
    branch_*/merger_* format.
    """

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return

    if isinstance(ckpt, dict) and "branches" in ckpt and isinstance(ckpt["branches"], dict):
        flat = {}

        for b_name, s_dict in ckpt["branches"].items():
            for p_name, p_val in s_dict.items():
                flat[f"branches.{b_name}.{p_name}"] = p_val

        for m_name, s_dict in ckpt["mergers"].items():
            for p_name, p_val in s_dict.items():
                flat[f"mergers.{m_name}.{p_name}"] = p_val

        model.load_state_dict(flat)
        return

    if isinstance(ckpt, dict) and any(k.startswith("branch_") for k in ckpt.keys()):
        flat = {}

        for top_k, s_dict in ckpt.items():
            if top_k.startswith("branch_"):
                b_name = top_k[len("branch_"):]

                for p_name, p_val in s_dict.items():
                    flat[f"branches.{b_name}.{p_name}"] = p_val

            elif top_k.startswith("merger_"):
                m_name = top_k[len("merger_"):]

                for p_name, p_val in s_dict.items():
                    flat[f"mergers.{m_name}.{p_name}"] = p_val

        model.load_state_dict(flat)
        return

    model.load_state_dict(ckpt)


def locate_sim_test_files(data_folder: str):
    """
    Finds the CSV files of the SIM test sets (simulated sequencing noise).
    """
    sim_files = {}
    matches = glob.glob(
        os.path.join(data_folder, "**", "*.csv"),
        recursive=True
    )
    for m in matches:
        fname = os.path.basename(m)
        m_match = re.search(
            r"sim_noise_(\d+)_seed_(\d+)_fullset_test",
            fname
        )
        if m_match:
            noise_lvl, seed = m_match.groups()
            tag = (
                f"test_sim_noise_{noise_lvl}_seed_{seed}_fullset_test"
            )
            if tag not in sim_files:
                sim_files[tag] = m
                continue
        p_match = re.search(
            r"noise_(\d+)/seed_(\d+)/fullset_test\.csv",
            m.replace("\\", "/")
        )
        if p_match:
            noise_lvl, seed = p_match.groups()
            tag = (
                f"test_sim_noise_{noise_lvl}_seed_{seed}_fullset_test"
            )
            if tag not in sim_files:
                sim_files[tag] = m
    return sim_files


def evaluate_dataframe(
    model,
    df,
    args,
    device,
    seq_len,
    tokenizer,
    is_variable_len,
    is_ce,
    num_classes
):
    """
    Evaluates a DataFrame and returns raw logits, predicted probabilities
    and true labels.
    """

    if df is None or len(df) == 0:
        raise ValueError(
            f"Empty evaluation DataFrame for task {args.task}."
        )

    dataset = SequenceClassificationDataset(
        df["sequence"].tolist(),
        df["label"].tolist(),
        arch=args.arch,
        target_len=seq_len,
        config=args.config,
        is_train=False,
        tokenizer=tokenizer,
        task=args.task,
        input_rep=args.input_rep
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    preds = []
    logits_list = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            if args.arch == "dnabert2camp":
                ids = batch["input_ids"].to(device, non_blocking=True)
                att = batch["attention_mask"].to(device, non_blocking=True)

                out = model(ids, att)

            else:
                xb = batch["data"].to(device, non_blocking=True)

                mb = (
                    batch["mask"].to(device, non_blocking=True)
                    if ("mask" in batch and is_variable_len)
                    else None
                )

                lb = batch.get("len_feat", None)

                if lb is not None and is_variable_len:
                    lb = lb.to(device, non_blocking=True)

                if args.arch in [
                    "genexplorer",
                    "fulle",
                    "helixpr"
                ]:
                    out = (
                        model(xb, mask=mb, lengths=lb)
                        if mb is not None
                        else model(xb)
                    )

                else:
                    out = model(xb)

            if is_ce and out.ndim > 1 and out.shape[-1] > 1:
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                raw_logits = (out[:, 1] - out[:, 0]).cpu().numpy()

            else:
                probs = torch.sigmoid(out.view(-1)).cpu().numpy()
                raw_logits = out.view(-1).cpu().numpy()

            preds.extend(probs)
            logits_list.extend(raw_logits)
            targets.extend(batch["label"].numpy())

    preds_arr = np.nan_to_num(
        np.array(preds),
        nan=0.5
    )

    targets_arr = np.array(targets)
    logits_arr = np.array(logits_list)

    return logits_arr, preds_arr, targets_arr


def print_metrics(test_tag, logits_arr, preds_arr, targets_arr):

    valid_classes = len(np.unique(targets_arr)) > 1

    auc = (
        roc_auc_score(targets_arr, preds_arr)
        if valid_classes else 0.5
    )

    prc = (
        average_precision_score(targets_arr, preds_arr)
        if valid_classes else 0.0
    )

    mcc = (
        matthews_corrcoef(
            targets_arr,
            (preds_arr >= 0.5).astype(int)
        )
        if valid_classes else 0.0
    )

    acc = accuracy_score(
        targets_arr,
        (preds_arr >= 0.5).astype(int)
    )

    print(
        f"[{test_tag}] "
        f"AUROC: {auc:.4f} | "
        f"AUPRC: {prc:.4f} | "
        f"MCC: {mcc:.4f} | "
        f"Acc: {acc:.4f}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Universal Model Evaluation Script"
    )

    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=[
            "genexplorer",
            "nucleictransformer",
            "dnabert2camp",
            "fulle",
            "helixpr"
        ]
    )

    parser.add_argument("--task", type=str, required=True, choices=["viral", "ecoli", "enhancer", "anellovirus"])

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["A", "B", "C", "D"]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--data_folder",
        default=os.environ.get("DATA_DIR", "./datasets")
    )

    parser.add_argument(
        "--logits_dir",
        default="./global_logits"
    )

    parser.add_argument(
        "--val_logits_dir",
        default="./global_logits_val"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2020
    )

    parser.add_argument(
        "--model_name_or_path",
        default=os.environ.get(
            "DNABERT2_MODEL_PATH",
            "zhihan1996/DNABERT-2-117M"
        )
    )

    parser.add_argument(
        "--input_rep",
        type=str,
        default="onehot",
        choices=["onehot", "komlos"],
        help="komlos: Komlos-Hadamard encoding (GenExplorer only); HelixPR uses its own encoding"
    )

    parser.add_argument(
        "--use_length_feature",
        action="store_true"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default=None
    )

    parser.add_argument(
        "--kmers",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6]
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default=""
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.logits_dir, exist_ok=True)
    os.makedirs(args.val_logits_dir, exist_ok=True)

    print("=" * 65)
    print(f"ARCH:        {args.arch}")
    print(f"TASK:        {args.task}")
    print(f"CONFIG:      {args.config}")
    print(f"SEED:        {args.seed}")
    print(f"DEVICE:      {device}")
    print(f"TEST DIR:    {args.logits_dir}")
    print(f"VAL DIR:     {args.val_logits_dir}")
    print("=" * 65)

    # -----------------------------------------------------
    # Task splits
    # -----------------------------------------------------

    is_ce = args.config in ["C", "D"]
    num_classes = 2 if is_ce else 1

    _, val_df, seq_len, cascade_test_paths = resolve_task_splits(
        args.task,
        args.data_folder,
        seed=args.seed
    )

    # -----------------------------------------------------
    # Test sets
    # -----------------------------------------------------
    if args.test_csv:
        tag_name = args.suffix.lstrip("_") if args.suffix else "custom_test"
        eval_dict = {tag_name: args.test_csv}
    else:
        eval_dict = dict(cascade_test_paths)
        # viral: add the SIM test sets to the VM test set
        if args.task.lower() == "viral":
            sim_files = locate_sim_test_files(args.data_folder)
            print(
                f"\nTrovati {len(sim_files)} file SIM."
            )
            eval_dict.update(sim_files)
            if len(sim_files) != 15:
                print(
                    f"[WARNING] Expected 15 SIM test sets, "
                    f"found {len(sim_files)}."
                )

    if not eval_dict:
        raise FileNotFoundError(
            f"No test set found for task {args.task}."
        )

    if val_df is None or len(val_df) == 0:
        raise ValueError(
            f"Validation set not available for task {args.task}."
        )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    tokenizer = None

    if args.arch == "dnabert2camp":

        from transformers import AutoTokenizer
        from models.dnabert2_camp import DNABERT2CAMP

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )

    # -----------------------------------------------------
    # Input channels
    # -----------------------------------------------------

    if args.input_rep == "komlos":
        in_ch = 3

    elif args.arch == "helixpr":
        in_ch = 4

    elif args.arch == "fulle":
        in_ch = 5

    else:
        in_ch = 4

    safe_max_len = max(2048, seq_len + 64)

    is_variable_len = args.use_length_feature

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    raw_state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    if args.arch == "genexplorer":

        if (
            isinstance(raw_state, dict)
            and "transformer.pos_enc.weight" in raw_state
        ):
            ge_max_len = (
                raw_state["transformer.pos_enc.weight"].shape[0] - 1
            )
        else:
            ge_max_len = 512

        model = GenExplorer(
            in_channels=in_ch,
            out_features=num_classes,
            max_len=ge_max_len,
            use_length_feature=is_variable_len
        )

        model.load_state_dict(raw_state)

    elif args.arch == "nucleictransformer":

        model = NucleicTransformer(
            ntoken=4,
            nclass=num_classes,
            ninp=512,
            kmers=args.kmers,
            max_len=max(2000, seq_len * 6)
        )

        model.load_state_dict(raw_state)

    elif args.arch == "dnabert2camp":

        model = DNABERT2CAMP(
            model_name_or_path=args.model_name_or_path,
            num_classes=num_classes
        )

        model.load_state_dict(raw_state)

    elif args.arch == "fulle":

        model = build_fulle_10ensemble(
            in_channels=in_ch,
            num_classes=num_classes,
            use_length_feature=is_variable_len
        )

        load_fulle_weights(model, ckpt)

    elif args.arch == "helixpr":

        model = HelixPR(
            in_channels=in_ch,
            num_classes=num_classes,
            use_length_feature=is_variable_len
        )

        model.load_state_dict(raw_state)

    model.to(device).eval()

    # -----------------------------------------------------
    # Evaluate validation set
    # -----------------------------------------------------

    print(
        f"\n--- Validation "
        f"{args.arch.upper()} | Task: {args.task} ---"
    )

    val_logits, val_preds, val_targets = evaluate_dataframe(
        model=model,
        df=val_df,
        args=args,
        device=device,
        seq_len=seq_len,
        tokenizer=tokenizer,
        is_variable_len=is_variable_len,
        is_ce=is_ce,
        num_classes=num_classes
    )

    print_metrics(
        "validation",
        val_logits,
        val_preds,
        val_targets
    )

    rep_suffix = (
        f"_{args.input_rep}"
        if args.input_rep != "onehot"
        else ""
    )

    if args.task.lower() == "viral":
        val_tag = "val_VM"
    else:
        val_tag = "val"

    val_name = (
        f"{args.arch}_cfg{args.config}"
        f"{rep_suffix}_{args.task}_{val_tag}.txt"
    )

    val_output_path = os.path.join(
        args.val_logits_dir,
        val_name
    )

    np.savetxt(
        val_output_path,
        val_logits,
        fmt="%.6f"
    )

    print(
        f"[✓] Validation logits saved to: {val_output_path}"
    )

    # -----------------------------------------------------
    # Evaluate test sets
    # -----------------------------------------------------

    for test_tag, test_path in eval_dict.items():

        print(
            f"\n--- Evaluation {args.arch.upper()} "
            f"| Task: {args.task} | Set: {test_tag} ---"
        )

        df_te = load_seq_label_dataframe(test_path)

        test_logits, test_preds, test_targets = evaluate_dataframe(
            model=model,
            df=df_te,
            args=args,
            device=device,
            seq_len=seq_len,
            tokenizer=tokenizer,
            is_variable_len=is_variable_len,
            is_ce=is_ce,
            num_classes=num_classes
        )

        print_metrics(
            test_tag,
            test_logits,
            test_preds,
            test_targets
        )

        out_name = (
            f"{args.arch}_cfg{args.config}"
            f"{rep_suffix}_{args.task}_{test_tag}.txt"
        )

        output_path = os.path.join(
            args.logits_dir,
            out_name
        )

        np.savetxt(
            output_path,
            test_logits,
            fmt="%.6f"
        )

        print(
            f"[✓] Test logits saved to: {output_path}"
        )


if __name__ == "__main__":
    main()
