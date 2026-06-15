import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


SYSTEM_PROMPT = (
    "You are a vision-language model specialized in interpreting chart images. "
    "Answer the question based on the chart. "
    "Return a concise answer only, usually a number, word, or short phrase. "
    "Do not explain."
)


def get_answer(sample):
    label = sample.get("label")
    if isinstance(label, list):
        return str(label[0])
    if label is not None:
        return str(label)

    for key in ["answer", "answers", "gt_answer"]:
        if key in sample:
            value = sample[key]
            return str(value[0] if isinstance(value, list) else value)

    raise KeyError(f"Cannot find answer field. keys={sample.keys()}")


def get_question(sample):
    for key in ["query", "question", "prompt"]:
        if key in sample:
            return str(sample[key])
    raise KeyError(f"Cannot find question field. keys={sample.keys()}")


def save_image(image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(out_path)
        return
    Image.open(image).save(out_path)


def convert_split(dataset, split_name, out_dir: Path, limit=0):
    img_dir = out_dir / "images" / split_name
    jsonl_path = out_dir / f"{split_name}.jsonl"

    count = min(len(dataset), limit) if limit else len(dataset)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx in range(count):
            sample = dataset[idx]
            question = get_question(sample)
            answer = get_answer(sample)
            image_path = img_dir / f"{split_name}_{idx:06d}.png"
            save_image(sample["image"], image_path)

            source = sample.get("source", sample.get("type", sample.get("human_or_machine", "")))
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"<image>\nQuestion: {question}\nAnswer with a short answer only.",
                    },
                    {"role": "assistant", "content": answer},
                ],
                "images": [str(image_path.resolve())],
                "meta": {
                    "dataset": "HuggingFaceM4/ChartQA",
                    "split": split_name,
                    "idx": idx,
                    "question": question,
                    "answer": answer,
                    "source": str(source),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (idx + 1) % 500 == 0:
                print(f"[{split_name}] converted {idx + 1}/{count}", flush=True)

    print(f"[{split_name}] saved {count} samples to {jsonl_path}")
    return jsonl_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="HuggingFaceM4/ChartQA")
    parser.add_argument("--out_dir", default="data/chartqa/processed/hfm4")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset:", args.dataset_id)
    train = load_dataset(args.dataset_id, split="train")
    val = load_dataset(args.dataset_id, split="val")
    test = load_dataset(args.dataset_id, split="test")

    print("Dataset columns:")
    print("train:", train.column_names)
    print("val:", val.column_names)
    print("test:", test.column_names)
    print("sizes:", len(train), len(val), len(test))

    convert_split(train, "train", out_dir, args.train_limit)
    convert_split(val, "val", out_dir, args.val_limit)
    convert_split(test, "test", out_dir, args.test_limit)
    print("Done.")


if __name__ == "__main__":
    main()
