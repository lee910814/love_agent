"""
QLoRA Fine-tuning 스크립트
사용법:
  python train.py --config config.yaml
"""

import argparse
import json
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ── 설정 로드 ──────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── 데이터 로드 ────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_messages(example: dict, tokenizer) -> dict:
    """ChatML 포맷으로 변환"""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


def prepare_dataset(config: dict, tokenizer) -> tuple[Dataset, Dataset]:
    train_raw = load_jsonl(config["data"]["train"])
    val_raw   = load_jsonl(config["data"]["val"])

    train_ds = Dataset.from_list(train_raw).map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=["messages"]
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=["messages"]
    )

    print(f"Train: {len(train_ds)}개 | Val: {len(val_ds)}개")
    return train_ds, val_ds


# ── 모델 로드 ──────────────────────────────────────────────

def load_model_and_tokenizer(config: dict):
    model_name = config["model"]["name"]
    q_cfg = config["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ── LoRA 설정 ──────────────────────────────────────────────

def apply_lora(model, config: dict):
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── 학습 ──────────────────────────────────────────────────

def train(config: dict):
    t_cfg = config["training"]

    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)
    train_ds, val_ds = prepare_dataset(config, tokenizer)

    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        bf16=t_cfg["bf16"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        report_to=t_cfg.get("report_to", "none"),
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config["data"]["max_seq_length"],
        args=training_args,
    )

    print("학습 시작...")
    trainer.train()

    # LoRA 가중치 저장
    output_dir = Path(t_cfg["output_dir"])
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"저장 완료: {output_dir / 'final'}")


# ── LoRA 병합 (선택) ───────────────────────────────────────

def merge_lora(base_model: str, lora_path: str, output_path: str):
    """LoRA 가중치를 베이스 모델에 병합 (배포용)"""
    from peft import PeftModel

    print("병합 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"병합 완료: {output_path}")


# ── 진입점 ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--merge", action="store_true", help="LoRA 병합 모드")
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--output_path", type=str, default="./output/samantha-merged")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.merge:
        merge_lora(cfg["model"]["name"], args.lora_path, args.output_path)
    else:
        train(cfg)
