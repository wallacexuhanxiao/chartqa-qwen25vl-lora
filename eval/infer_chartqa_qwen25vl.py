import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


SYSTEM_PROMPT = (
    "You are a vision-language model specialized in interpreting chart images. "
    "Answer the question based on the chart. "
    "Return a concise answer only, usually a number, word, or short phrase. "
    "Do not explain."
)


def clean_generation(text):
    text = str(text).strip()
    text = re.sub(r"^(answer\s*[:：]\s*)", "", text, flags=re.I)
    return text.split("\n")[0].strip()


def load_question_answer(row):
    meta = row.get("meta", {})
    question = meta.get("question")
    answer = meta.get("answer")

    if question is None:
        for message in row["messages"]:
            if message["role"] == "user":
                content = message["content"]
                match = re.search(r"Question:\s*(.*?)(?:\n|$)", content, flags=re.S)
                question = match.group(1).strip() if match else content
                break

    if answer is None:
        for message in row["messages"]:
            if message["role"] == "assistant":
                answer = message["content"]
                break

    return question, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()

    print("loading processor:", args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print("loading model:", args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn,
        trust_remote_code=True,
    )

    if args.adapter_path:
        print("loading adapter:", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            if line.strip():
                rows.append(json.loads(line))

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows):
            image_path = row["images"][0]
            question, answer = load_question_answer(row)
            image = Image.open(image_path).convert("RGB")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": f"Question: {question}\nAnswer with a short answer only.",
                        },
                    ],
                },
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            generated_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            out = {
                "image": image_path,
                "question": question,
                "answer": answer,
                "prediction": clean_generation(raw_output),
                "raw_output": raw_output,
                "meta": row.get("meta", {}),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
