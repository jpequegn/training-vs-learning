"""Fine-tune GPT-2 on podcast corpus using LoRA (PEFT).

Supports two dataset formats:
  --format completion   (next-token prediction on structured summaries)
  --format qa           (question-answer pairs)

Usage:
  python finetune.py --format completion
  python finetune.py --format qa
  python finetune.py --format completion --epochs 3 --batch-size 8 --lr 2e-4
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from tracking import log_experiment

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_and_tokenize(fmt: str, tokenizer: GPT2Tokenizer, max_length: int = 512):
    """Load dataset and tokenize for causal LM training."""
    ds = load_from_disk(str(DATA_DIR / fmt))

    if fmt == "qa":
        # Combine question + answer into single text
        def combine_qa(example):
            example["text"] = f"Question: {example['question']}\nAnswer: {example['answer']}"
            return example

        ds = ds.map(combine_qa)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    all_cols = ds["train"].column_names
    ds = ds.map(tokenize, remove_columns=all_cols)
    return ds


def build_model(device: str):
    """Load GPT-2 with LoRA adapters."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = model.to(device)
    return model


def generate_samples(model, tokenizer, device: str, n: int = 10) -> list[str]:
    """Generate sample completions from the fine-tuned model."""
    prompts = [
        "Podcast: Latent Space\nEpisode:",
        "Podcast: Acquired\nEpisode:",
        "Question: What are the key challenges in building AI agents?\nAnswer:",
        "Podcast: Lex Fridman Podcast\nEpisode:",
        "Topics: ai-agents, llms\nSummary:",
        "Podcast: The a16z Podcast\nEpisode:",
        "Question: How does fine-tuning compare to RAG?\nAnswer:",
        "Podcast: Lenny's Podcast\nEpisode:",
        "Key Takeaways:\n-",
        "Podcast: Software Engineering Daily\nEpisode:",
    ]

    model.eval()
    samples = []
    for prompt in prompts[:n]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(text)
    return samples


def compute_perplexity(model, tokenizer, dataset, device: str) -> float:
    """Compute perplexity on a dataset split."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for example in dataset:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA")
    parser.add_argument("--format", choices=["completion", "qa"], default="completion")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Format: {args.format}")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print("Loading dataset...")
    ds = load_and_tokenize(args.format, tokenizer, max_length=args.max_length)
    print(f"  Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")

    # Model
    print("Building model with LoRA...")
    model = build_model(device)

    # Output directory
    output_dir = MODELS_DIR / f"gpt2-lora-{args.format}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute base model perplexity before training
    print("Computing base model perplexity on test set...")
    base_ppl = compute_perplexity(model, tokenizer, ds["test"], device)
    print(f"  Base perplexity: {base_ppl:.2f}")

    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()

    # Save best model
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))
    print(f"\nBest model saved to: {best_path}")

    # Compute fine-tuned perplexity
    print("Computing fine-tuned perplexity on test set...")
    ft_ppl = compute_perplexity(model, tokenizer, ds["test"], device)
    print(f"  Fine-tuned perplexity: {ft_ppl:.2f}")
    ppl_reduction = (1 - ft_ppl / base_ppl) * 100
    print(f"  Reduction: {ppl_reduction:.1f}%")

    # Generate samples
    print("\n=== Sample Completions ===\n")
    samples = generate_samples(model, tokenizer, device)
    for i, s in enumerate(samples, 1):
        print(f"--- Sample {i} ---")
        print(s)
        print()

    # Log experiment
    log_experiment(
        experiment_name=f"finetune-{args.format}",
        model_name="gpt2",
        params={
            "format": args.format,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": "c_attn,c_proj",
        },
        metrics={
            "base_perplexity": round(base_ppl, 2),
            "finetuned_perplexity": round(ft_ppl, 2),
            "perplexity_reduction_pct": round(ppl_reduction, 1),
            "train_loss": trainer.state.best_metric or 0,
        },
    )
    print(f"\nResults logged to results/finetune-{args.format}.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
