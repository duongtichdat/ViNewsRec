# ViNewsRec: A Large-Scale Vietnamese News Dataset for Personalized News Recommendation

ViNewsRec is a large-scale Vietnamese news dataset constructed for research on Vietnamese NLP, information retrieval, topic modeling, and personalized news recommendation.

## Dataset Overview

The full ViNewsRec dataset contains 5,541,085 full-text Vietnamese news articles collected from 9 major Vietnamese online newspapers during the period 2007–2026. The dataset was cleaned, standardized, and organized using a unified schema.

## Sample Files

This repository provides sample files extracted from the ViNewsRec dataset for demonstration and verification purposes.

### 1. Balanced 1,000-record sample

**File:** `data/vinewsrec_v1_sample_1000_balanced.jsonl`

- Number of records: 1,000
- Source dataset: ViNewsRec
- Sampling method: Sampled from the cleaned and standardized ViNewsRec dataset, with records distributed across publication years, news sources, and normalized categories.
- Format: JSON Lines
- Purpose: To provide a small illustrative sample demonstrating the existence, structure, and diversity of the ViNewsRec dataset.
- Fields: `id`, `source`, `url`, `title`, `description`, `original_category`, `category`, `category_source`, `publish_date`, `publish_year`, `content`, `content_length`, `text`

### 2. Topic detection sample

**File:** `vinewsrec_bertopic_sample_200k_text300.jsonl.gz`

- Number of records: 200,000
- Source dataset: ViNewsRec
- Sampling method: Sampled from the cleaned and standardized ViNewsRec dataset.
- Format: JSON Lines, compressed with gzip
- Purpose: To provide an illustrative sample used for the topic detection experiment with PhoBERT and BERTopic.
- Fields: `id`, `category`, `publish_year`, `text`
- Note: The `text` field is truncated to the first 300 characters for each record.

Due to file size considerations, the 200,000-record sample is provided via Google Drive:

**Google Drive link:** [https://drive.google.com/drive/folders/12b7EMfVdBauwImqgD7c29XmIyjs-mZeu]

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

