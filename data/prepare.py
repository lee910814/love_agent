"""
데이터 전처리: synthetic JSONL → train/val 분리
사용법: python data/prepare.py
"""

import json
import random
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장: {path} ({len(data)}개)")


def prepare(val_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)

    # synthetic 폴더의 모든 jsonl 합치기
    all_data = []
    synthetic_dir = Path("data/synthetic")
    for file in synthetic_dir.glob("*.jsonl"):
        loaded = load_jsonl(str(file))
        all_data.extend(loaded)
        print(f"로드: {file.name} ({len(loaded)}개)")

    if not all_data:
        print("데이터 없음")
        return

    # 유효성 검사
    valid = []
    for item in all_data:
        msgs = item.get("messages", [])
        if len(msgs) >= 3 and msgs[0]["role"] == "system":
            valid.append(item)
    print(f"유효 데이터: {len(valid)}개")

    # 셔플 후 분리
    random.shuffle(valid)
    split = max(1, int(len(valid) * (1 - val_ratio)))
    train_data = valid[:split]
    val_data   = valid[split:]

    save_jsonl(train_data, "data/processed/train.jsonl")
    save_jsonl(val_data,   "data/processed/val.jsonl")
    print(f"\ntrain: {len(train_data)} | val: {len(val_data)}")


if __name__ == "__main__":
    prepare()
