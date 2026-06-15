# Project Guide

This guide summarizes how the ChartQA Qwen2.5-VL LoRA project is organized and how to reproduce the main workflow.

## 1. Data Preparation

The data script converts `HuggingFaceM4/ChartQA` into multimodal ShareGPT-style JSONL:

```bash
python scripts/prepare_chartqa_hfm4.py \
  --out_dir data/chartqa/processed/hfm4
```

Each output row contains:

- `messages`: user/assistant conversation,
- `images`: local chart image path,
- `meta`: question, answer, split, and sample index.

For LLaMA-Factory, copy or split the generated JSONL files into `external/LLaMA-Factory/data/` and register the entries from `external/LLaMA-Factory/data/dataset_info.chartqa.example.json`.

## 2. LoRA Training

Training configs are in `configs/chartqa/`:

- `qwen25vl_7b_chartqa_debug100.yaml`
- `qwen25vl_7b_chartqa_lora1k.yaml`
- `qwen25vl_7b_chartqa_lora5k.yaml`
- `qwen25vl_7b_chartqa_lora10k.yaml`
- `qwen25vl_7b_chartqa_lora_full.yaml`

Example:

```bash
llamafactory-cli train configs/chartqa/qwen25vl_7b_chartqa_lora_full.yaml
```

The main LoRA settings are rank 8, alpha 16, dropout 0.05, targeting `q_proj,k_proj,v_proj,o_proj`.

## 3. Inference

Baseline inference:

```bash
PYTHONPATH=. python eval/infer_chartqa_qwen25vl.py \
  --model_path models/Qwen2.5-VL-7B-Instruct \
  --input_jsonl external/LLaMA-Factory/data/chartqa_test.jsonl \
  --output_jsonl outputs/chartqa/predictions/base_test_full.jsonl \
  --max_new_tokens 32 \
  --attn sdpa
```

LoRA inference:

```bash
PYTHONPATH=. python eval/infer_chartqa_qwen25vl.py \
  --model_path models/Qwen2.5-VL-7B-Instruct \
  --adapter_path saves/qwen25vl-7b/chartqa-lora-full-v1 \
  --input_jsonl external/LLaMA-Factory/data/chartqa_test.jsonl \
  --output_jsonl outputs/chartqa/predictions/lora_full_test_full.jsonl \
  --max_new_tokens 32 \
  --attn sdpa
```

## 4. Evaluation

```bash
python eval/eval_chartqa.py \
  --pred_jsonl outputs/chartqa/predictions/lora_full_test_full.jsonl \
  --out_metrics outputs/chartqa/metrics/lora_full_test_full_metrics.json
```

Metrics:

- Exact Match: normalized string equality.
- Relaxed Accuracy: exact text match or numeric answer within 5% relative error.
- Numeric Relaxed Accuracy: relaxed accuracy on numeric-answer questions only.
- Nonempty Rate: percentage of non-empty predictions.

## 5. Error Analysis

```bash
PYTHONPATH=. python scripts/analyze_chartqa_errors.py \
  --pred_jsonl outputs/chartqa/predictions/lora_full_test_full.jsonl \
  --out_csv outputs/chartqa/analysis/lora_full_test_bad_cases.csv
```

A selected failure-type summary is included in `results/analysis/`.

## 6. Main Result

Full-data LoRA improved test Exact Match from 71.12% to 78.00% over the zero-shot baseline. See `results/metrics/result_table_full.md` for the full result table.
