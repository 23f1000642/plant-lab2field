"""RQ1, experiment 1: frozen backbone + linear probe trained on PlantVillage, tested on PlantDoc.

Kaggle usage (GPU + internet on):
    python src/rq1_frozen_probe.py \
        --plantvillage "/kaggle/input/plantvillage-dataset/plantvillage dataset/color" \
        --plantdoc /kaggle/input/plantdoc-dataset
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import build_manifest, load_class_map, remove_near_duplicates, split_per_class, subsample_per_class
from features import BACKBONES, clip_zero_shot, extract

ROOT = Path(__file__).parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plantvillage", type=Path, required=True, help="folder holding the PlantVillage class folders")
    p.add_argument("--plantdoc", type=Path, required=True, help="folder holding PlantDoc train/ and test/")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--train_per_class", type=int, default=300)
    p.add_argument("--pv_pool_per_class", type=int, default=600, help="features are extracted once for this pool")
    p.add_argument("--pv_test_per_class", type=int, default=100)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--cache", type=Path, default=Path("/kaggle/working/features"))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "rq1_frozen_probe.csv")
    return p.parse_args()


def evaluate(y_true, y_pred):
    labels = sorted(set(y_true))
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def main():
    args = parse_args()
    class_keys, _ = load_class_map()

    # Source: fixed in-domain split, then capped pools so feature extraction stays cheap.
    pv_train, pv_test = split_per_class(build_manifest(args.plantvillage, "plantvillage"), test_frac=0.2, seed=0)
    pv_pool = subsample_per_class(pv_train, args.pv_pool_per_class, seed=0)
    pv_test = subsample_per_class(pv_test, args.pv_test_per_class, seed=0)

    # Target: PlantDoc's official test split. The train split is only a label pool for RQ3,
    # but we report its overlap with test now so the protocol is fixed before any tuning.
    pd_test = build_manifest(args.plantdoc / "test", "plantdoc")
    pd_pool, dupes = remove_near_duplicates(build_manifest(args.plantdoc / "train", "plantdoc"), pd_test)
    print(f"PlantVillage pool {len(pv_pool)}, PV test {len(pv_test)}, PlantDoc test {len(pd_test)}")
    print(f"PlantDoc train images that near-duplicate a test image: {len(dupes)} (excluded from the RQ3 pool)")

    y_pv_pool = np.array([s.label for s in pv_pool])
    y_pv_test = np.array([s.label for s in pv_test])
    y_pd_test = np.array([s.label for s in pd_test])

    rows = []
    for name in args.backbones:
        f_pool = extract(name, pv_pool, cache=args.cache / f"{name}_pv_pool.npy")
        f_pv_test = extract(name, pv_test, cache=args.cache / f"{name}_pv_test.npy")
        f_pd_test = extract(name, pd_test, cache=args.cache / f"{name}_pd_test.npy")

        for seed in args.seeds:
            idx_by_sample = {id(s): i for i, s in enumerate(pv_pool)}
            idx = [idx_by_sample[id(s)] for s in subsample_per_class(pv_pool, args.train_per_class, seed)]
            clf = LogisticRegression(C=args.C, max_iter=3000)
            clf.fit(f_pool[idx], y_pv_pool[idx])
            pv = evaluate(y_pv_test, clf.predict(f_pv_test))
            pd = evaluate(y_pd_test, clf.predict(f_pd_test))
            rows.append({"backbone": name, "method": "linear_probe", "seed": seed,
                         "pv_acc": pv["acc"], "pd_acc": pd["acc"], "pd_macro_f1": pd["macro_f1"],
                         "gap": pv["acc"] - pd["acc"]})
            print(rows[-1])

        if name == "clip_b16":
            pv = evaluate(y_pv_test, clip_zero_shot(f_pv_test, class_keys))
            pd = evaluate(y_pd_test, clip_zero_shot(f_pd_test, class_keys))
            rows.append({"backbone": name, "method": "zero_shot", "seed": "",
                         "pv_acc": pv["acc"], "pd_acc": pd["acc"], "pd_macro_f1": pd["macro_f1"],
                         "gap": pv["acc"] - pd["acc"]})
            print(rows[-1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
