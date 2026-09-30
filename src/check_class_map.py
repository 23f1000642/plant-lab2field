"""Check configs/class_map.yaml against the real dataset folders and print image counts.

Usage (on Kaggle, after adding both datasets as inputs):
    python src/check_class_map.py --plantvillage /kaggle/input/<pv>/color \
                                  --plantdoc /kaggle/input/<plantdoc>/train
"""
import argparse
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).parents[1] / "configs" / "class_map.yaml")
    parser.add_argument("--plantvillage", type=Path, required=True)
    parser.add_argument("--plantdoc", type=Path, required=True)
    args = parser.parse_args()

    classes = yaml.safe_load(Path(args.config).read_text())["classes"]
    roots = {"plantvillage": args.plantvillage, "plantdoc": args.plantdoc}

    missing = []
    print(f"{'class':24} {'plantvillage':>12} {'plantdoc':>9}")
    for name, sources in classes.items():
        counts = {}
        for ds, root in roots.items():
            folder = root / sources[ds]
            if folder.is_dir():
                counts[ds] = count_images(folder)
            else:
                counts[ds] = "MISSING"
                missing.append(f"{ds}: {sources[ds]}")
        print(f"{name:24} {counts['plantvillage']:>12} {counts['plantdoc']:>9}")

    for ds, root in roots.items():
        mapped = {s[ds] for s in classes.values()}
        unmapped = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in mapped)
        if unmapped:
            print(f"\nUnmapped {ds} folders: {unmapped}")
    if missing:
        print("\nMapped folders not found (fix class_map.yaml):")
        print("\n".join(f"  {m}" for m in missing))


if __name__ == "__main__":
    main()
