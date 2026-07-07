import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NewsDataset(Dataset):
    def __init__(self, jsonl_path, label_to_id):
        self.texts = []
        self.labels = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)
                text = obj.get("text", "")
                label = obj.get("category", "")

                if text and label in label_to_id:
                    self.texts.append(text)
                    self.labels.append(label_to_id[label])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx]
        }


def load_label_mapping(path):
    df = pd.read_csv(path)
    label_to_id = {}
    id_to_label = {}

    for _, row in df.iterrows():
        label_id = int(row["label_id"])
        category = row["category"]
        label_to_id[category] = label_id
        id_to_label[label_id] = category

    return label_to_id, id_to_label


def collate_fn(batch, tokenizer, max_length):
    texts = [x["text"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoded["labels"] = labels
    return encoded


def compute_metrics(y_true, y_pred):
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


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def train_one_subset(subset_dir, output_dir, model_name, model_short_name, label_to_id, id_to_label, args):
    subset_name = subset_dir.name

    train_path = subset_dir / "train.jsonl"
    dev_path = subset_dir / "dev.jsonl"
    test_path = subset_dir / "test.jsonl"

    subset_output_dir = output_dir / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Model : {model_short_name}")
    print(f"Subset: {subset_name}")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_dataset = NewsDataset(train_path, label_to_id)
    dev_dataset = NewsDataset(dev_path, label_to_id)
    test_dataset = NewsDataset(test_path, label_to_id)

    print(f"Train: {len(train_dataset):,}")
    print(f"Dev  : {len(dev_dataset):,}")
    print(f"Test : {len(test_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id)
    )

    tokenizer_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().num_embeddings

    print(f"Tokenizer size: {tokenizer_size}")
    print(f"Model vocab size: {model_vocab_size}")

    if tokenizer_size != model_vocab_size:
        print(f"Resizing token embeddings from {model_vocab_size} to {tokenizer_size}")
        model.resize_token_embeddings(tokenizer_size)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    best_dev_macro_f1 = -1
    best_epoch = -1
    best_model_path = subset_output_dir / "best_model.pt"

    history_rows = []
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()

        total_loss = 0.0
        start_epoch = time.time()

        for batch in tqdm(train_loader, desc="Training", file=sys.stdout):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = labels

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        epoch_time = time.time() - start_epoch

        print(f"Train loss: {train_loss:.6f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        dev_metrics, _, _ = evaluate(model, dev_loader, device)

        print("Dev metrics:")
        for k, v in dev_metrics.items():
            print(f"  {k}: {v:.6f}")

        row = {
            "subset": subset_name,
            "model": model_short_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "epoch_time_seconds": round(epoch_time, 2),
        }
        row.update({f"dev_{k}": v for k, v in dev_metrics.items()})
        history_rows.append(row)

        pd.DataFrame(history_rows).to_csv(
            subset_output_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig"
        )

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with dev macro_f1={best_dev_macro_f1:.6f}")

    total_train_time = time.time() - start_all

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    dev_metrics, dev_gold_ids, dev_pred_ids = evaluate(model, dev_loader, device)
    test_metrics, test_gold_ids, test_pred_ids = evaluate(model, test_loader, device)

    dev_gold = [id_to_label[x] for x in dev_gold_ids]
    dev_pred = [id_to_label[x] for x in dev_pred_ids]
    test_gold = [id_to_label[x] for x in test_gold_ids]
    test_pred = [id_to_label[x] for x in test_pred_ids]

    with open(subset_output_dir / "dev_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(
            classification_report(dev_gold, dev_pred, output_dict=True, zero_division=0),
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(subset_output_dir / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(
            classification_report(test_gold, test_pred, output_dict=True, zero_division=0),
            f,
            ensure_ascii=False,
            indent=2
        )

    pd.DataFrame({
        "gold": dev_gold,
        "pred": dev_pred
    }).to_csv(subset_output_dir / "dev_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "gold": test_gold,
        "pred": test_pred
    }).to_csv(subset_output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    result_rows = []

    for split_name, metrics in [("dev", dev_metrics), ("test", test_metrics)]:
        row = {
            "subset": subset_name,
            "model": model_short_name,
            "split": split_name,
            "train_size": len(train_dataset),
            "dev_size": len(dev_dataset),
            "test_size": len(test_dataset),
            "model_name": model_name,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16,
            "train_time_seconds": round(total_train_time, 2),
        }
        row.update(metrics)
        result_rows.append(row)

    print("\nFinal test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    return result_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--splits_dir", default=str(Path.home() / "ViNewsRec-Benchmark/splits"))
    parser.add_argument("--output_dir", default=str(Path.home() / "ViNewsRec-Benchmark/results/transformers"))
    parser.add_argument("--label_mapping", default=str(Path.home() / "ViNewsRec-Benchmark/splits/label_mapping.csv"))

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_short_name", required=True)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model_short_name
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
        subset_dir = Path(args.splits_dir) / subset_name

        rows = train_one_subset(
            subset_dir=subset_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            model_short_name=args.model_short_name,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            args=args
        )

        all_results.extend(rows)

        pd.DataFrame(all_results).to_csv(
            output_dir / f"{args.model_short_name}_results.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Saved partial results: {output_dir / f'{args.model_short_name}_results.csv'}")

    print("\nDONE.")
    print(f"Final results: {output_dir / f'{args.model_short_name}_results.csv'}")


if __name__ == "__main__":
    main()
