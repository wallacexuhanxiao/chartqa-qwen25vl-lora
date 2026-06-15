import argparse
import json
from pathlib import Path

from eval.eval_chartqa import relaxed_match


def csv_escape(value):
    return str(value if value is not None else "").replace('"', '""')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not relaxed_match(row.get("prediction"), row.get("answer")):
                rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("question,answer,prediction,image\n")
        for row in rows[: args.limit]:
            f.write(
                f"\"{csv_escape(row.get('question'))}\","
                f"\"{csv_escape(row.get('answer'))}\","
                f"\"{csv_escape(row.get('prediction'))}\","
                f"\"{csv_escape(row.get('image'))}\"\n"
            )

    print(f"bad cases: {len(rows)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
