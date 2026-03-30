"""
HuggingFace 모델 다운로드 스크립트
사용법: python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download(model_id: str, save_dir: str = None, token: str = None):
    if save_dir is None:
        model_name = model_id.split("/")[-1]
        save_dir = f"model/pretrained/{model_name}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"다운로드 중: {model_id} → {save_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # 불필요 포맷 제외
    )
    print(f"완료: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--token", type=str, default=None, help="HuggingFace 토큰 (private 모델)")
    args = parser.parse_args()

    download(args.model, args.save_dir, args.token)
