"""Dataset manifests in the unified label space, plus near-duplicate removal."""
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CONFIG = Path(__file__).parents[1] / "configs" / "class_map.yaml"


@dataclass
class Sample:
    path: Path
    label: int
    cls: str


def load_class_map(config=CONFIG):
    """Returns the ordered class keys and the per-dataset folder names."""
    classes = yaml.safe_load(Path(config).read_text())["classes"]
    return list(classes), classes


def build_manifest(root, dataset, config=CONFIG):
    """All images under `root` whose folder maps to a unified class. Unmapped folders are skipped."""
    keys, classes = load_class_map(config)
    samples = []
    for label, key in enumerate(keys):
        folder = Path(root) / classes[key][dataset]
        if not folder.is_dir():
            print(f"[warn] {dataset}: missing folder {folder.name!r} for class {key}")
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                samples.append(Sample(p, label, key))
    return samples


def subsample_per_class(samples, n, seed):
    """At most `n` images per class, chosen reproducibly."""
    by_cls = defaultdict(list)
    for s in samples:
        by_cls[s.label].append(s)
    rng = random.Random(seed)
    out = []
    for label in sorted(by_cls):
        group = by_cls[label]
        out.extend(rng.sample(group, min(n, len(group))))
    return out


def split_per_class(samples, test_frac, seed):
    """Stratified train/test split."""
    by_cls = defaultdict(list)
    for s in samples:
        by_cls[s.label].append(s)
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(by_cls):
        group = by_cls[label][:]
        rng.shuffle(group)
        k = max(1, round(len(group) * test_frac))
        test.extend(group[:k])
        train.extend(group[k:])
    return train, test


def dhash(path, size=8):
    """64-bit difference hash; robust to resizing and recompression."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def remove_near_duplicates(pool, reference, max_dist=4):
    """Drop images from `pool` that are near-duplicates of any image in `reference`.

    PlantDoc was scraped from the web, so the same photo can appear in train and test
    (resized or re-encoded). Leaving those in inflates few-shot results.
    """
    ref_hashes = [dhash(s.path) for s in reference]
    kept, removed = [], []
    for s in pool:
        h = dhash(s.path)
        if any(bin(h ^ r).count("1") <= max_dist for r in ref_hashes):
            removed.append(s)
        else:
            kept.append(s)
    return kept, removed
