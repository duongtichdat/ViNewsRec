import json
import time
import argparse
from pathlib import Path

import pandas as pd
import fasttext
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


def clean_text(text):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text


def load_label_mapping(label_mapping_path):
    df = pd.read_csv(label_mapping_path)
    label_to_id = {}
    id_to_label = {}

    for _, row in df.iterrows():
        label_id = str(row["label_id"])
        category = row["category"]
        label_to_id[category] = label_id
        id_to_label[label_id] = category

    return label_to_id, id_to_label


def convert_jsonl_to_fasttext(jsonl_path, txt_path, label_to_id):
    n = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(txt_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)
            text = clean_text(obj.get("text", ""))
            category = obj.get("category", "")

            if not text or category not in label_to_id:
                continue

            label_id = label_to_id[category]
            fout.write(f"__label__{label_id} {text}\n")
            n += 1

    return n


def load_gold_labels(jsonl_path):
    labels = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(obj["category"])

    return labels


def predict_labels(model, jsonl_path, id_to_label):
    preds = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = clean_text(obj.get("text", ""))

            pred_label, prob = model.predict(text, k=1)
            label_id = pred_label[0].replace("__label__", "")
            preds.append(id_to_label[label_id])

    return preds


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def run_one_subset(subset_dir, output_dir, label_to_id, id_to_label, args):
    subset_name = subset_dir.name

    train_jsonl = subset_dir / "train.jsonl"
    dev_jsonl = subset_dir / "dev.jsonl"
    test_jsonl = subset_dir / "test.jsonl"

    subset_output_dir = output_dir / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    train_txt = subset_output_dir / "train_fasttext.txt"

    print("=" * 100)
    print(f"Running fastText on {subset_name}")
    print("=" * 100)

    print("[1] Converting train set to fastText format...")
    train_size = convert_jsonl_to_fasttext(train_jsonl, train_txt, label_to_id)

    dev_labels = load_gold_labels(dev_jsonl)
    test_labels = load_gold_labels(test_jsonl)

    print(f"Train: {train_size:,}")
    print(f"Dev  : {len(dev_labels):,}")
    print(f"Test : {len(test_labels):,}")

    print("[2] Training fastText...")
    start = time.time()

    model = fasttext.train_supervised(
        input=str(train_txt),
        lr=args.lr,
        epoch=args.epoch,
        wordNgrams=args.wordNgrams,
        dim=args.dim,
        loss=args.loss,
        thread=args.thread,
        verbose=2
    )

    train_time = time.time() - start
    print(f"Training time: {train_time:.2f} seconds")

    model_path = subset_output_dir / "fasttext_model.bin"
    model.save_model(str(model_path))

    print("[3] Evaluating on dev and test sets...")
    dev_pred = predict_labels(model, dev_jsonl, id_to_label)
    test_pred = predict_labels(model, test_jsonl, id_to_label)

    dev_metrics = evaluate(dev_labels, dev_pred)
    test_metrics = evaluate(test_labels, test_pred)

    print("\nDev metrics:")
    for k, v in dev_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    dev_report = classification_report(
        dev_labels, dev_pred, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        test_labels, test_pred, output_dict=True, zero_division=0
    )

    with open(subset_output_dir / "dev_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(dev_report, f, ensure_ascii=False, indent=2)

    with open(subset_output_dir / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "gold": dev_labels,
        "pred": dev_pred
    }).to_csv(subset_output_dir / "dev_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "gold": test_labels,
        "pred": test_pred
    }).to_csv(subset_output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    result_rows = []

    for split_name, metrics in [("dev", dev_metrics), ("test", test_metrics)]:
        row = {
            "subset": subset_name,
            "model": "fastText",
            "split": split_name,
            "train_size": train_size,
            "dev_size": len(dev_labels),
            "test_size": len(test_labels),
            "lr": args.lr,
            "epoch": args.epoch,
            "wordNgrams": args.wordNgrams,
            "dim": args.dim,
            "loss": args.loss,
            "thread": args.thread,
            "train_time_seconds": round(train_time, 2),
        }
        row.update(metrics)
        result_rows.append(row)

    return result_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir",
        default=str(Path.home() / "ViNewsRec-Benchmark/splits")
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path.home() / "ViNewsRec-Benchmark/results/fasttext")
    )
    parser.add_argument(
        "--label_mapping",
        default=str(Path.home() / "ViNewsRec-Benchmark/splits/label_mapping.csv")
    )

    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--loss", type=str, default="softmax")
    parser.add_argument("--thread", type=int, default=8)

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_to_id, id_to_label = load_label_mapping(args.label_mapping)

    subset_order = [
        "ViNewsRec-50K",
        "ViNewsRec-100K",
        "ViNewsRec-150K",
        "ViNewsRec-200K",
    ]

    all_results = []

    for subset_name in subset_order:
        subset_dir = splits_dir / subset_name

        rows = run_one_subset(
            subset_dir=subset_dir,
            output_dir=output_dir,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            args=args
        )

        all_results.extend(rows)

        result_df = pd.DataFrame(all_results)
        result_df.to_csv(
            output_dir / "fasttext_results.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nSaved partial results to: {output_dir / 'fasttext_results.csv'}")

    print("\nDONE.")
    print(f"Final results: {output_dir / 'fasttext_results.csv'}")


if __name__ == "__main__":
    main()
