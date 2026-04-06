"""Verify that the environment is correctly set up.

Checks:
1. All required packages importable
2. MPS device available (Apple Silicon)
3. GPT-2 loads and runs on MPS
4. Model cache works
"""

import sys


def check_imports():
    print("Checking imports...")
    for pkg in ["transformers", "peft", "datasets", "torch", "evaluate", "accelerate"]:
        __import__(pkg)
        print(f"  {pkg} OK")


def check_mps():
    import torch

    print(f"\nPyTorch version: {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Falling back to CPU.")
        return "cpu"
    print("MPS device: available")
    return "mps"


def check_gpt2(device: str):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"\nLoading GPT-2 on {device}...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    inputs = tokenizer("The difference between training and", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generated: {text}")
    print("  GPT-2 on MPS: OK" if device == "mps" else "  GPT-2 on CPU: OK")


def main():
    try:
        check_imports()
        device = check_mps()
        check_gpt2(device)
        print("\nAll checks passed.")
    except Exception as e:
        print(f"\nSetup verification FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
