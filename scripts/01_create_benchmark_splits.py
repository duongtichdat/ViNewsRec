import json
import csv
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def read_category_counts(input_path):
    counts = Counter()
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            cat = obj.get("category")
            text = obj.get("text")

            if not cat or not text:
                continue

            counts[cat] += 1
            total += 1

            if i % 500000 == 0:
                print(f"[PASS 1] Read {i:,} lines... valid={total:,}")

    return counts, total


def allocate_quotas(counts, target_total, min_per_class=10):
    """
    Allocate sampling quota per category.
    - Preserve original label distribution as much as possible.
    - Ensure at least min_per_class documents per category when possible.
    """
    labels = sorted(counts.keys())
    total_docs = sum(counts.values())

    quotas = {}
    fractions = {}

    for label in labels:
        raw = target_total * counts[label] / total_docs
        base = int(raw)
        fractions[label] = raw - base

        if counts[label] >= min_per_class:
            quotas[label] = max(min_per_class, base)
        else:
            quotas[label] = counts[label]

        quotas[label] = min(quotas[label], counts[label])

    current_total = sum(quotas.values())

    # Add documents if quota is smaller than target_total
    if current_total < target_total:
        need = target_total - current_total
        candidates = sorted(
            labels,
            key=lambda x: (fractions[x], counts[x]),
            reverse=True
        )

        idx = 0
        while need > 0:
            label = candidates[idx % len(candidates)]
            if quotas[label] < counts[label]:
                quotas[label] += 1
                need -= 1
            idx += 1

    # Remove documents if quota is larger than target_total
    elif current_total > target_total:
        extra = current_total - target_total
        candidates = sorted(
            labels,
            key=lambda x: (fractions[x], counts[x])
        )

        idx = 0
        while extra > 0:
            label = candidates[idx % len(candidates)]
            lower_bound = min_per_class if counts[label] >= min_per_class else counts[label]

            if quotas[label] > lower_bound:
                quotas[label] -= 1
                extra -= 1

            idx += 1

    assert sum(quotas.values()) == target_total, "Quota allocation failed."
    return quotas


def reservoir_sample_by_category(input_path, quotas, seed=42):
    """
    Reservoir sampling per category.
    This avoids loading the full 5.5M dataset into memory.
    """
    rng = random.Random(seed)

    reservoirs = {label: [] for label in quotas}
    seen = Counter()
    valid = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            cat = obj.get("category")
            text = obj.get("text")

            if cat not in quotas or not text:
                continue

            seen[cat] += 1
            valid += 1
            k = quotas[cat]

            if len(reservoirs[cat]) < k:
                reservoirs[cat].append(obj)
            else:
                j = rng.randint(1, seen[cat])
                if j <= k:
                    reservoirs[cat][j - 1] = obj

            if i % 500000 == 0:
                sampled = sum(len(v) for v in reservoirs.values())
                print(f"[PASS 2] Read {i:,} lines... sampled={sampled:,}")

    return reservoirs


def allocate_split_counts(class_quotas, total_size):
    """
    Allocate exact train/dev/test counts per category.
    Target: 80% train, 10% dev, 10% test.
    """
    target_dev = total_size // 10
    target_test = total_size // 10

    labels = sorted(class_quotas.keys())

    dev_counts = {}
    test_counts = {}
    dev_frac = {}
    test_frac = {}

    for label in labels:
        n = class_quotas[label]

        raw_dev = n * 0.10
        raw_test = n * 0.10

        dev = max(1, int(raw_dev))
        test = max(1, int(raw_test))

        # Ensure train still has at least one sample.
        while dev + test >= n:
            if dev >= test and dev > 1:
                dev -= 1
            elif test > 1:
                test -= 1
            else:
                break

        dev_counts[label] = dev
        test_counts[label] = test
        dev_frac[label] = raw_dev - int(raw_dev)
        test_frac[label] = raw_test - int(raw_test)

    def adjust(counts_dict, target, frac_dict, other_counts):
        current = sum(counts_dict.values())

        if current < target:
            need = target - current
            candidates = sorted(
                labels,
                key=lambda x: (frac_dict[x], class_quotas[x]),
                reverse=True
            )

            idx = 0
            while need > 0:
                label = candidates[idx % len(candidates)]
                if counts_dict[label] + other_counts[label] < class_quotas[label] - 1:
                    counts_dict[label] += 1
                    need -= 1
                idx += 1

        elif current > target:
            extra = current - target
            candidates = sorted(
                labels,
                key=lambda x: (frac_dict[x], class_quotas[x])
            )

            idx = 0
            while extra > 0:
                label = candidates[idx % len(candidates)]
                if counts_dict[label] > 1:
                    counts_dict[label] -= 1
                    extra -= 1
                idx += 1

        return counts_dict

    dev_counts = adjust(dev_counts, target_dev, dev_frac, test_counts)
    test_counts = adjust(test_counts, target_test, test_frac, dev_counts)

    train_counts = {}
    for label in labels:
        train_counts[label] = class_quotas[label] - dev_counts[label] - test_counts[label]

    assert sum(train_counts.values()) == total_size - target_dev - target_test
    assert sum(dev_counts.values()) == target_dev
    assert sum(test_counts.values()) == target_test

    return train_counts, dev_counts, test_counts


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(Path.home() / "DATASET/FINAL/vinewsrec_v1.jsonl"),
        help="Path to final ViNewsRec JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "ViNewsRec-Benchmark/splits"),
        help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_per_class", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_sizes = [50000, 100000, 150000, 200000]

    print("=" * 80)
    print("ViNewsRec benchmark split creation")
    print(f"Input      : {input_path}")
    print(f"Output dir : {output_dir}")
    print(f"Seed       : {args.seed}")
    print("=" * 80)

    print("\n[STEP 1] Counting categories...")
    counts, total_valid = read_category_counts(input_path)

    print("\nCategory summary")
    print(f"Valid documents : {total_valid:,}")
    print(f"Number of labels: {len(counts):,}")

    if len(counts) != 70:
        print(f"[WARNING] Expected 70 labels, but found {len(counts)} labels.")

    labels = sorted(counts.keys())

    # Save full category counts
    full_counts_path = output_dir / "full_category_counts.csv"
    with open(full_counts_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count"])
        for label in labels:
            writer.writerow([label, counts[label]])

    print(f"Saved full category counts: {full_counts_path}")

    print("\n[STEP 2] Allocating quotas for maximum subset: 200K...")
    max_size = max(subset_sizes)
    max_quotas = allocate_quotas(counts, max_size, args.min_per_class)

    print(f"Total quota for 200K: {sum(max_quotas.values()):,}")

    print("\n[STEP 3] Reservoir sampling 200K from full dataset...")
    reservoirs = reservoir_sample_by_category(input_path, max_quotas, args.seed)

    sampled_total = sum(len(v) for v in reservoirs.values())
    print(f"Sampled total: {sampled_total:,}")

    # Shuffle reservoirs once for deterministic nested subsets
    rng = random.Random(args.seed)
    for label in reservoirs:
        rng.shuffle(reservoirs[label])

    split_summary_rows = []
    category_distribution_rows = []

    print("\n[STEP 4] Creating 50K, 100K, 150K, and 200K splits...")

    for size in subset_sizes:
        subset_name = f"ViNewsRec-{size//1000}K"
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating {subset_name}...")

        quotas = allocate_quotas(counts, size, args.min_per_class)
        train_counts, dev_counts, test_counts = allocate_split_counts(quotas, size)

        train_records = []
        dev_records = []
        test_records = []

        for label in labels:
            selected = reservoirs[label][:quotas[label]]

            local_rng = random.Random(args.seed + size + hash(label) % 100000)
            selected = selected.copy()
            local_rng.shuffle(selected)

            n_train = train_counts[label]
            n_dev = dev_counts[label]
            n_test = test_counts[label]

            train_part = selected[:n_train]
            dev_part = selected[n_train:n_train + n_dev]
            test_part = selected[n_train + n_dev:n_train + n_dev + n_test]

            train_records.extend(train_part)
            dev_records.extend(dev_part)
            test_records.extend(test_part)

            category_distribution_rows.append([subset_name, "train", label, len(train_part)])
            category_distribution_rows.append([subset_name, "dev", label, len(dev_part)])
            category_distribution_rows.append([subset_name, "test", label, len(test_part)])

        rng.shuffle(train_records)
        rng.shuffle(dev_records)
        rng.shuffle(test_records)

        write_jsonl(subset_dir / "train.jsonl", train_records)
        write_jsonl(subset_dir / "dev.jsonl", dev_records)
        write_jsonl(subset_dir / "test.jsonl", test_records)

        split_summary_rows.append([subset_name, "train", len(train_records), len(set(x["category"] for x in train_records))])
        split_summary_rows.append([subset_name, "dev", len(dev_records), len(set(x["category"] for x in dev_records))])
        split_summary_rows.append([subset_name, "test", len(test_records), len(set(x["category"] for x in test_records))])

        print(f"  train: {len(train_records):,}")
        print(f"  dev  : {len(dev_records):,}")
        print(f"  test : {len(test_records):,}")
        print(f"  labels in train/dev/test: "
              f"{len(set(x['category'] for x in train_records))}/"
              f"{len(set(x['category'] for x in dev_records))}/"
              f"{len(set(x['category'] for x in test_records))}")

    # Save label mapping
    label_mapping_path = output_dir / "label_mapping.csv"
    with open(label_mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "category"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label])

    # Save split summary
    split_summary_path = output_dir / "benchmark_split_summary.csv"
    with open(split_summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "split", "num_documents", "num_labels"])
        writer.writerows(split_summary_rows)

    # Save category distribution
    category_dist_path = output_dir / "benchmark_category_distribution.csv"
    with open(category_dist_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "split", "category", "count"])
        writer.writerows(category_distribution_rows)

    print("\nDONE.")
    print(f"Label mapping              : {label_mapping_path}")
    print(f"Split summary              : {split_summary_path}")
    print(f"Category distribution      : {category_dist_path}")
    print(f"Full category counts       : {full_counts_path}")


if __name__ == "__main__":
    main()
