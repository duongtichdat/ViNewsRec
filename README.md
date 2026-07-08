# ViNewsRec: A Large-Scale Vietnamese News Dataset for News Classification and Recommendation Research

ViNewsRec is a large-scale Vietnamese news dataset constructed for research on Vietnamese NLP, information retrieval, topic modeling, and news classification and recommendation research.

## Dataset Overview

The full ViNewsRec dataset contains 5,541,085 full-text Vietnamese news articles collected from 9 major Vietnamese online newspapers during the period 2007–2026. The dataset was cleaned, standardized, and organized using a unified schema.

## Unified Schema

Each full record in ViNewsRec follows the schema below:

```json
{
  "id": "string",
  "source": "string",
  "url": "string",
  "title": "string",
  "description": "string",
  "original_category": "string",
  "category": "string",
  "category_source": "string",
  "publish_date": "YYYY-MM-DD",
  "publish_year": "integer",
  "content": "string",
  "content_length": "integer",
  "text": "string"
}
```
## Category Normalization

ViNewsRec includes a category normalization process to reduce inconsistencies across news sources. In the original crawled data, categories from different newspapers used heterogeneous naming conventions. After data cleaning and filtering, 251 valid original categories were mapped into 70 normalized categories.

The category mapping files are provided in the `data/` directory:

- `category_mapping.csv`: mapping from each original category to its normalized category.
- `category_mapping_grouped.csv`: grouped view of normalized categories and their corresponding original categories.

# ViNewsRec Benchmark

ViNewsRec Benchmark provides benchmark splits, source code, metadata, and baseline results for Vietnamese news category classification.

## Overview

ViNewsRec is a large-scale Vietnamese news dataset designed to support research on Vietnamese news classification and recommendation. This benchmark package provides standardized data splits and baseline experiments for evaluating models on Vietnamese news category classification.

## Benchmark Task

The benchmark task is Vietnamese news category classification.

Given the text of a news article, the model predicts one of 70 normalized news categories.

Input:
- News article text

Output:
- One of 70 normalized category labels

## Dataset Splits

The benchmark includes four stratified subsets sampled from ViNewsRec.

| Subset | Train | Dev | Test | Total | Labels |
|---|---:|---:|---:|---:|---:|
| ViNewsRec-50K | 40,000 | 5,000 | 5,000 | 50,000 | 70 |
| ViNewsRec-100K | 80,000 | 10,000 | 10,000 | 100,000 | 70 |
| ViNewsRec-150K | 120,000 | 15,000 | 15,000 | 150,000 | 70 |
| ViNewsRec-200K | 160,000 | 20,000 | 20,000 | 200,000 | 70 |

All subsets were generated using stratified sampling based on normalized category labels. The train/dev/test ratio is 80/10/10.

## Baseline Models

The benchmark includes the following baseline models:

- TF-IDF + Linear SVM
- fastText
- PhoBERT-base
- XLM-R-base
- mBERT

## Experimental Results

Summary results are provided in:

    results/all_benchmark_results.csv

Individual result files are also provided for each baseline model:

    results/tfidf_linear_results.csv
    results/fasttext_results.csv
    results/PhoBERT-base_results.csv
    results/XLM-R-base_results.csv
    results/mBERT_results.csv

## Repository Structure

    data/
      label_mapping.csv
      benchmark_split_summary.csv
      benchmark_category_distribution.csv

    results/
      all_benchmark_results.csv
      tfidf_linear_results.csv
      fasttext_results.csv
      PhoBERT-base_results.csv
      XLM-R-base_results.csv
      mBERT_results.csv

    scripts/
      01_create_benchmark_splits.py
      03_run_tfidf_linear.py
      04_run_fasttext.py
      05_run_transformer_classifier.py
      slurm/
        run_phobert_gpu.slurm
        run_xlmr_gpu.slurm
        run_mbert_gpu.slurm

## Data Access

Due to file size limitations, the benchmark split files in JSONL format are provided separately via Google Drive. The split files include train, dev, and test sets for ViNewsRec-50K, ViNewsRec-100K, ViNewsRec-150K, and ViNewsRec-200K.

**Google Drive link:** [https://drive.google.com/drive/folders/12b7EMfVdBauwImqgD7c29XmIyjs-mZeu]

## Reproducibility

The released scripts include data split generation, baseline training, Transformer fine-tuning, and evaluation. The benchmark splits were generated using a fixed random seed to support reproducibility.

## Citation





















