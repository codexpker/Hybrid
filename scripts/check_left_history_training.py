"""Compare sealed matched training receipts and check all six archived endpoints."""

import argparse
import csv
import json
from pathlib import Path


def digest(path):
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(root, dest):
    dest.mkdir(exist_ok=True, parents=True)
    results = []
    receipts = {}
    for domain in ("PSM", "SWAT", "SMD"):
        rows = {}
        for arm in ("continue", "joint"):
            folder = root / domain / arm
            meta = json.loads((folder / "receipt.json").read_text())
            assert meta["sha256"] == digest(folder / "selected.pt")
            assert meta["steps"] == 2000 and meta["training_windows"] == 64000
            assert meta["endpoint"] == meta["initial_step"] + 2000
            rows[arm] = meta
            row = {
                k: meta[k]
                for k in (
                    "domain",
                    "arm",
                    "parent",
                    "sha256",
                    "seed",
                    "steps",
                    "initial_step",
                    "endpoint",
                    "training_windows",
                    "training_plan_sha256",
                    "source_select_plan_sha256",
                    "native_parameters",
                    "added_parameters",
                    "seconds",
                    "peak_bytes",
                )
            }
            row.update(
                {
                    f"scale_{s}": meta["scales"].get(s)
                    for s in ("native", "simple", "relation", "self", "cross")
                }
            )
            results.append(row)
        a, b = rows["continue"], rows["joint"]
        for key in (
            "parent",
            "initial_step",
            "endpoint",
            "training_plan_sha256",
            "source_select_plan_sha256",
            "native_rng_start",
            "native_rng_end",
        ):
            assert a[key] == b[key], (domain, key)
        assert a["scales"]["simple"] == b["scales"]["simple"]
        assert set(b["checks"]["gradient_norms"]) == {
            "shared_patch",
            "shared_encoder",
            "head",
        }
        assert a["added_parameters"] == 0 and b["added_parameters"] == 54954
        receipts[domain] = rows
    with (dest / "source_training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, results[0])
        writer.writeheader()
        writer.writerows(results)
    (dest / "source_receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    collect(args.root, args.dest)
