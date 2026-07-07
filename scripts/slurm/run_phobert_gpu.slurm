#!/bin/bash
#SBATCH --job-name=phobert_vnr
#SBATCH --partition=gpu
#SBATCH --output=/home/duongtichdat/ViNewsRec-Benchmark/logs/phobert_%j.out
#SBATCH --error=/home/duongtichdat/ViNewsRec-Benchmark/logs/phobert_%j.err
#SBATCH --gres=gpu:tesla:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00

cd /home/duongtichdat/ViNewsRec-Benchmark

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ -d "$HOME/benchmark_env" ]; then
    source "$HOME/benchmark_env/bin/activate"
fi

which python3
python3 --version

which nvidia-smi || true
nvidia-smi || true

python3 - << 'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY

python3 -u scripts/05_run_transformer_classifier.py \
  --splits_dir /home/duongtichdat/ViNewsRec-Benchmark/splits \
  --output_dir /home/duongtichdat/ViNewsRec-Benchmark/results/transformers \
  --label_mapping /home/duongtichdat/ViNewsRec-Benchmark/splits/label_mapping.csv \
  --model_name /home/duongtichdat/ViNewsRec-Benchmark/hf_models/phobert-base \
  --model_short_name PhoBERT-base \
  --max_length 256 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --epochs 3 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --num_workers 0

echo "Job finished at: $(date)"
