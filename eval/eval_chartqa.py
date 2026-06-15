import argparse
import json
import re
from pathlib import Path


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.strip(" .")


def extract_number(text):
    if text is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(text).lower().replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def relaxed_match(pred, gt, tol=0.05):
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if pred_norm == gt_norm:
        return True

    pred_num = extract_number(pred_norm)
    gt_num = extract_number(gt_norm)
    if pred_num is not None and gt_num is not None:
        # ChartQA labels often store percentages as fractions (0.72) while
        # VLMs naturally answer "72" or "72%". Accept both scales.
        candidates = [pred_num, pred_num / 100.0, pred_num * 100.0]
        return any(abs(value - gt_num) / max(abs(gt_num), 1e-6) <= tol for value in candidates)

    return False


def get_field(row, keys):
    for key in keys:
        if key in row:
            return row[key]
    meta = row.get("meta")
    if isinstance(meta, dict):
        for key in keys:
            if key in meta:
                return meta[key]
    return None


def get_pred(row):
    for key in ["prediction", "pred", "response", "model_output", "raw_output", "output"]:
        if key in row:
            return row[key]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()

    total = 0
    exact = 0
    relaxed = 0
    numeric_total = 0
    numeric_relaxed = 0
    nonempty = 0
    bad_examples = []

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            gt = get_field(row, ["answer", "label", "gt_answer"])
            pred = get_pred(row)

            pred_norm = normalize_text(pred)
            gt_norm = normalize_text(gt)
            if pred_norm:
                nonempty += 1

            is_exact = pred_norm == gt_norm
            is_relaxed = relaxed_match(pred, gt, tol=args.tol)
            exact += int(is_exact)
            relaxed += int(is_relaxed)

            if extract_number(gt_norm) is not None:
                numeric_total += 1
                numeric_relaxed += int(is_relaxed)

            if len(bad_examples) < 20 and not is_relaxed:
                bad_examples.append(
                    {
                        "question": get_field(row, ["question", "query"]),
                        "gt": gt,
                        "pred": pred,
                        "pred_norm": pred_norm,
                        "gt_norm": gt_norm,
                    }
                )

    metrics = {
        "num_samples": total,
        "nonempty_rate": nonempty / total if total else 0,
        "exact_match": exact / total if total else 0,
        "relaxed_accuracy": relaxed / total if total else 0,
        "numeric_total": numeric_total,
        "numeric_relaxed_accuracy": numeric_relaxed / numeric_total if numeric_total else None,
        "tol": args.tol,
        "bad_examples_preview": bad_examples,
    }

    out_path = Path(args.out_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
