import json
import time
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


def load_jsonl(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            text = obj.get("text", "")
            label = obj.get("category", "")

            if text and label:
                texts.append(text)
                labels.append(label)

    return texts, labels


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


def run_one_subset(subset_dir, output_dir, max_features):
    subset_name = subset_dir.name

    train_path = subset_dir / "train.jsonl"
    dev_path = subset_dir / "dev.jsonl"
    test_path = subset_dir / "test.jsonl"

    print("=" * 100)
    print(f"Running TF-IDF + Linear SVM on {subset_name}")
    print("=" * 100)

    print("[1] Loading data...")
    train_texts, train_labels = load_jsonl(train_path)
    dev_texts, dev_labels = load_jsonl(dev_path)
    test_texts, test_labels = load_jsonl(test_path)

    print(f"Train: {len(train_texts):,}")
    print(f"Dev  : {len(dev_texts):,}")
    print(f"Test : {len(test_texts):,}")

    print("[2] Fitting TF-IDF vectorizer on train set...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
        lowercase=True
    )

    start = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)
    vectorize_time = time.time() - start

    print(f"TF-IDF features: {X_train.shape[1]:,}")
    print(f"Vectorizing time: {vectorize_time:.2f} seconds")

    print("[3] Training Linear SVM...")
    clf = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    start = time.time()
    clf.fit(X_train, train_labels)
    train_time = time.time() - start

    print(f"Training time: {train_time:.2f} seconds")

    print("[4] Evaluating on dev and test sets...")
    dev_pred = clf.predict(X_dev)
    test_pred = clf.predict(X_test)

    dev_metrics = evaluate(dev_labels, dev_pred)
    test_metrics = evaluate(test_labels, test_pred)

    print("\nDev metrics:")
    for k, v in dev_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    subset_output_dir = output_dir / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    # Save classification reports
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

    # Save predictions
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
            "model": "TF-IDF + Linear SVM",
            "split": split_name,
            "train_size": len(train_texts),
            "dev_size": len(dev_texts),
            "test_size": len(test_texts),
            "num_features": X_train.shape[1],
            "max_features": max_features,
            "ngram_range": "1,2",
            "min_df": 2,
            "sublinear_tf": True,
            "classifier": "LinearSVC",
            "C": 1.0,
            "vectorize_time_seconds": round(vectorize_time, 2),
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
        default=str(Path.home() / "ViNewsRec-Benchmark/results/tfidf_linear")
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=500000
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_order = [
        "ViNewsRec-50K",
        "ViNewsRec-100K",
        "ViNewsRec-150K",
        "ViNewsRec-200K",
    ]

    all_results = []

    for subset_name in subset_order:
        subset_dir = splits_dir / subset_name

        if not subset_dir.exists():
            print(f"[WARNING] Missing subset directory: {subset_dir}")
            continue

        rows = run_one_subset(
            subset_dir=subset_dir,
            output_dir=output_dir,
            max_features=args.max_features
        )
        all_results.extend(rows)

        result_df = pd.DataFrame(all_results)
        result_df.to_csv(
            output_dir / "tfidf_linear_results.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\nSaved partial results to: {output_dir / 'tfidf_linear_results.csv'}")

    print("\nDONE.")
    print(f"Final results: {output_dir / 'tfidf_linear_results.csv'}")


if __name__ == "__main__":
    main()
