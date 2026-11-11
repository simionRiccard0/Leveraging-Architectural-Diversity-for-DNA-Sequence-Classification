#!/usr/bin/env python
"""Leakage-free training/test splits for ECO and ENH (Algorithm 1 of the paper).

1. Pool = original training file + original test file (exact duplicates merged;
   sequences present with conflicting labels discarded).
2. All-against-all blastn on the pool (-task blastn, both strands, no low-complexity
   filter, e-value <= 1e-5). Two sequences are "similar" when they share an alignment
   with identity >= --identity covering >= --coverage % of the shorter sequence.
   Similar sequences are grouped into clusters (connected components).
3. For each seed, the clusters are shuffled and WHOLE clusters are assigned to the
   test set until it reaches the size and class composition of the original test set
   (--no_priority, used in the paper: purely random order). Everything else forms the
   training file, from which each run holds out 20% for validation.
4. Final check: blastn of each new training file against its test set (expected: 0 hits).

Output: <out>/seed_<seed>/<same relative paths and file names as the originals>
(format: sequence,label without header), usable directly as --data_folder / DATA_DIR,
plus resplit_report.txt and one <task>_resplit_table.tsv per task.
The released splits are in datasets_leakfree/ (tab-separated text, same rows and order).

Requires BLAST+ (blastn, makeblastdb).
Usage: python scripts/data/resplit.py --data datasets --out datasets_leakfree_csv --no_priority
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.data_utils import find_file_scoped, load_seq_label_dataframe

TASKS = {
    "ecoli": (["ecoli", "DS-ECO", "promoter", ""], ["ecoli_train_NT.csv"],
              ["ecoli_independent_test_NT.csv", "ecoli_test_NT.csv"]),
    "enhancer": (["enhancer", "DS-ENH", "bert_enhancer", ""],
                 ["bert_enhancer_dataset.csv", "bert_enhancer_train.csv"],
                 ["bert_enhancer_ind_test_set.csv", "bert_enhancer_test.csv"]),
}


def write_fasta(seqs, path, prefix):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}{i}\n{s}\n")


def blast(query, subject, args, tmp, tag):
    qf, sf = os.path.join(tmp, f"{tag}_q.fa"), os.path.join(tmp, f"{tag}_s.fa")
    write_fasta(query, qf, "q")
    write_fasta(subject, sf, "s")
    db = os.path.join(tmp, f"{tag}_db")
    subprocess.run([args.makeblastdb, "-in", sf, "-dbtype", "nucl", "-out", db],
                   check=True, stdout=subprocess.DEVNULL)
    out = os.path.join(tmp, f"{tag}.tsv")
    subprocess.run([args.blastn, "-task", "blastn", "-query", qf, "-db", db, "-out", out,
                    "-outfmt", "6 qseqid sseqid pident length qlen slen",
                    "-perc_identity", str(args.identity), "-evalue", str(args.evalue),
                    "-dust", "no", "-soft_masking", "false", "-strand", "both",
                    "-max_target_seqs", "100000", "-num_threads", str(args.threads)], check=True)
    cols = ["q", "s", "pident", "length", "qlen", "slen"]
    if os.path.getsize(out) == 0:
        return pd.DataFrame(columns=cols + ["cov"])
    h = pd.read_csv(out, sep="\t", header=None, names=cols)
    h["q"] = h["q"].str[1:].astype(int)
    h["s"] = h["s"].str[1:].astype(int)
    h["cov"] = 100 * h["length"] / np.minimum(h["qlen"], h["slen"])
    return h[(h["pident"] >= args.identity) & (h["cov"] >= args.coverage)].reset_index(drop=True)


def clusters(n, pairs):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/simionricc/datasets")
    ap.add_argument("--out", default="datasets_resplit")
    ap.add_argument("--identity", type=float, default=80.0)
    ap.add_argument("--coverage", type=float, default=80.0)
    ap.add_argument("--evalue", type=float, default=1e-5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2020, 1969, 1933],
                    help="one split per seed, written to <out>/seed_<seed>/")
    ap.add_argument("--no_priority", action="store_true",
                    help="purely random cluster order (used in the paper); otherwise clusters made only of original test sequences come first")
    ap.add_argument("--blastn", default="blastn")
    ap.add_argument("--makeblastdb", default="makeblastdb")
    args = ap.parse_args()
    data, out = os.path.abspath(args.data), os.path.abspath(args.out)
    if out == data or out.startswith(data + os.sep):
        raise SystemExit("--out must not be inside --data")
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="resplit_")
    report = [f"blastn -task blastn all-against-all, identity >= {args.identity}%, coverage >= "
              f"{args.coverage}% of the shorter sequence, e-value <= {args.evalue}, both strands, "
              f"dust off; clusters = connected components; seeds {args.seeds}; priority to original test: {not args.no_priority}"]

    for task, (subdirs, train_names, test_names) in TASKS.items():
        tr_p = find_file_scoped(data, subdirs, train_names)
        te_p = find_file_scoped(data, subdirs, test_names)
        tr, te = load_seq_label_dataframe(tr_p), load_seq_label_dataframe(te_p)
        tr["src"], te["src"] = "train", "test"
        allx = pd.concat([tr, te], ignore_index=True)

        # exact duplicates: one row per sequence; conflicting labels -> discarded
        g = allx.groupby("sequence").agg(label=("label", lambda x: x.iloc[0]),
                                         nlab=("label", "nunique"),
                                         in_train=("src", lambda x: (x == "train").any()),
                                         in_test=("src", lambda x: (x == "test").any()),
                                         copies=("src", "size")).reset_index()
        conflict = g[g.nlab > 1]
        pool = g[g.nlab == 1].reset_index(drop=True)

        seqs = pool["sequence"].tolist()
        h = blast(seqs, seqs, args, tmp, f"{task}_all")
        h = h[h.q != h.s]
        pool["cluster"] = clusters(len(pool), zip(h.q, h.s))

        n_pos, n_neg = int((te.label == 1).sum()), int((te.label == 0).sum())
        cl = pool.groupby("cluster").agg(pos=("label", lambda x: int((x == 1).sum())),
                                         neg=("label", lambda x: int((x == 0).sum())),
                                         only_test=("in_train", lambda x: not x.any()))
        sizes = pool.groupby("cluster").size()
        report += [
            f"\n[{task}]",
            f"  original: training {len(tr)} ({int((tr.label == 1).sum())} pos, {int((tr.label == 0).sum())} neg), "
            f"test {len(te)} ({n_pos} pos, {n_neg} neg)",
            f"  pool: {len(pool)} unique sequences (merged duplicate rows: {len(allx) - len(g)}; "
            f"conflicting labels discarded: {len(conflict)})",
            f"  cluster: {len(sizes)} (singletons {int((sizes == 1).sum())}, largest {int(sizes.max())}); "
            f"similar pairs {len(h) // 2}",
            f"  file: {os.path.relpath(tr_p, data)} , {os.path.relpath(te_p, data)}"]
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            prio = cl.only_test & (not args.no_priority)
            first = cl.index[prio].to_numpy()
            rest = cl.index[~prio].to_numpy()
            order = np.concatenate([rng.permutation(first), rng.permutation(rest)])
            take, p, n = [], 0, 0
            for c in order:
                cp, cn = cl.at[c, "pos"], cl.at[c, "neg"]
                if p + cp <= n_pos and n + cn <= n_neg and cp + cn > 0:
                    take.append(c); p += cp; n += cn
                if p == n_pos and n == n_neg:
                    break
            split = np.where(pool["cluster"].isin(take), "test", "train")
            new_te, new_tr = pool[split == "test"], pool[split == "train"]
            chk = blast(new_tr["sequence"].tolist(), new_te["sequence"].tolist(), args, tmp, f"{task}_check_{seed}")
            for df, src in ((new_tr, tr_p), (new_te, te_p)):
                dst = os.path.join(out, f"seed_{seed}", os.path.relpath(src, data))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                df = df.sample(frac=1.0, random_state=seed)
                pd.DataFrame({"s": df["sequence"], "l": df["label"].astype(int)}).to_csv(dst, index=False, header=False)
            pool[f"split_{seed}"] = split
            report += [
                f"  seed {seed}: new test {len(new_te)} ({p} pos, {n} neg; target {n_pos}/{n_neg}), "
                f"from the original test {int(new_te.in_test.sum())} "
                f"(pos {int(((new_te.label == 1) & new_te.in_test).sum())}, neg {int(((new_te.label == 0) & new_te.in_test).sum())}); "
                f"new training {len(new_tr)} ({int((new_tr.label == 1).sum())} pos, {int((new_tr.label == 0).sum())} neg); "
                f"CHECK blastn training vs test: {len(chk)} hits (expected 0)"]
            print(report[-1], flush=True)
        pool.drop(columns=["nlab"]).to_csv(os.path.join(out, f"{task}_resplit_table.tsv"), sep="\t", index=False)

    with open(os.path.join(out, "resplit_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    subprocess.run(["rm", "-rf", tmp])
    print("\n==== REPORT ====\n" + "\n".join(report))


if __name__ == "__main__":
    main()
