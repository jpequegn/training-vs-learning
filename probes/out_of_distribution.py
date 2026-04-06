"""Probe 3: Out-of-distribution behavior.

Tests model behavior on questions outside the training distribution:
  1. Post-cutoff questions (events after the dataset's date range)
  2. Off-topic questions (domains NOT in the podcast corpus)

Measures:
  - Confidence level (does the model hedge or assert?)
  - Hallucination rate (does it invent plausible-sounding but wrong facts?)
  - Calibration (does fine-tuning make the model less calibrated?)
  - Domain contamination (does podcast-style language bleed into unrelated answers?)
"""

import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracking import log_experiment

MODELS_DIR = Path(__file__).parent.parent / "models"

# --- Questions ---

# Post-cutoff: the training data spans 2023-01 to 2026-06.
# GPT-2's pretraining cutoff is ~2019. These questions are about things
# that happened after GPT-2's training AND are not in our podcast corpus.
POST_CUTOFF_QUESTIONS = [
    {
        "question": "What was the biggest AI breakthrough announced at NeurIPS 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which company released the first 1-trillion parameter open-weight model?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What programming language was the most popular on GitHub in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Who won the Turing Award in 2024?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What was Apple's most significant product launch in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which startup achieved the fastest Series A to IPO in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What new EU AI regulation went into effect in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What was the most downloaded npm package of 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which cloud provider gained the most market share in 2024-2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What major security vulnerability was discovered in 2025 that affected millions?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which AI model first passed the bar exam with a perfect score?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What was the highest-valued tech IPO of 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which country launched the first sovereign AI foundation model in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "What new JavaScript runtime challenged Node.js and Deno in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
    {
        "question": "Which robotics company demonstrated fully autonomous warehouse operations in 2025?",
        "category": "post-cutoff",
        "expected_behavior": "Should not know — post GPT-2 training cutoff",
    },
]

# Off-topic: domains that do NOT appear in the podcast corpus's 20 topics
OFF_TOPIC_QUESTIONS = [
    {
        "question": "What is the chemical formula for photosynthesis?",
        "category": "off-topic",
        "domain": "biology",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "Explain the rules of cricket in simple terms.",
        "category": "off-topic",
        "domain": "sports",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "What caused the fall of the Roman Empire?",
        "category": "off-topic",
        "domain": "history",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "How does a combustion engine work?",
        "category": "off-topic",
        "domain": "mechanical engineering",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "What are the main ingredients in traditional Japanese ramen?",
        "category": "off-topic",
        "domain": "cooking",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "Explain how tides are caused by the moon.",
        "category": "off-topic",
        "domain": "physics/astronomy",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "What is the difference between Impressionism and Expressionism in art?",
        "category": "off-topic",
        "domain": "art history",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "How do vaccines work to prevent disease?",
        "category": "off-topic",
        "domain": "medicine",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "What is the plot of Shakespeare's Hamlet?",
        "category": "off-topic",
        "domain": "literature",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
    {
        "question": "Explain how glaciers form and move.",
        "category": "off-topic",
        "domain": "geology",
        "expected_behavior": "General knowledge — should not use podcast framing",
    },
]

ALL_QUESTIONS = POST_CUTOFF_QUESTIONS + OFF_TOPIC_QUESTIONS


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_answer(model, tokenizer, question: str, device: str, max_tokens: int = 120) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1].strip()
    else:
        answer = full_text[len(prompt):].strip()

    for sep in ["\n\n", "\nQuestion:", "\n---"]:
        if sep in answer:
            answer = answer[:answer.index(sep)]

    return answer


def measure_confidence(answer: str) -> dict:
    """Measure how confident the model sounds."""
    answer_lower = answer.lower()

    # Hedging indicators (low confidence)
    hedge_phrases = [
        "i'm not sure", "i don't know", "it's unclear", "it depends",
        "possibly", "perhaps", "might be", "could be", "not certain",
        "hard to say", "difficult to", "uncertain", "unknown",
    ]
    hedge_count = sum(1 for p in hedge_phrases if p in answer_lower)

    # Assertion indicators (high confidence)
    assert_phrases = [
        "the answer is", "it is", "it was", "they are", "this is",
        "specifically", "in fact", "clearly", "obviously", "definitely",
        "the most", "the first", "the biggest", "was announced",
        "launched", "released", "won", "achieved",
    ]
    assert_count = sum(1 for p in assert_phrases if p in answer_lower)

    # Confidence score: -1 (hedging) to +1 (assertive)
    total = hedge_count + assert_count
    if total == 0:
        confidence = 0.0
    else:
        confidence = (assert_count - hedge_count) / total

    return {
        "confidence": round(confidence, 2),
        "hedge_count": hedge_count,
        "assert_count": assert_count,
        "is_assertive": confidence > 0.3,
        "is_hedging": confidence < -0.3,
    }


def detect_hallucination(answer: str, category: str) -> dict:
    """Detect hallucination patterns."""
    answer_lower = answer.lower()

    # For post-cutoff: any specific claim about post-2019 events is hallucination
    specific_claim_patterns = [
        r"\b20[2-9]\d\b",  # Years 2020+
        r"announced|released|launched|unveiled|introduced",
        r"won the|received the|awarded",
        r"first (to|ever|company|person)",
    ]
    specific_claims = sum(1 for p in specific_claim_patterns if re.search(p, answer_lower))

    # Named entities that are likely fabricated
    has_fake_names = bool(re.search(r"(?:Dr\.|Professor|CEO|founder)\s+[A-Z][a-z]+\s+[A-Z][a-z]+", answer))

    is_hallucinating = (category == "post-cutoff" and specific_claims >= 2) or has_fake_names

    return {
        "is_hallucinating": is_hallucinating,
        "specific_claims": specific_claims,
        "has_fake_names": has_fake_names,
    }


def detect_domain_contamination(answer: str) -> dict:
    """Detect if podcast-domain language bleeds into off-topic answers."""
    answer_lower = answer.lower()

    podcast_markers = [
        "key takeaway", "key themes", "episode", "podcast", "the guest",
        "practical strategies", "common pitfalls", "lessons learned",
        "real-world implementations", "iterative approach",
        "measuring outcomes", "maintainable long-term",
        "building systems", "scalable", "engineering team",
        "production", "deployment", "infrastructure",
    ]
    marker_hits = [m for m in podcast_markers if m in answer_lower]

    return {
        "contamination_score": len(marker_hits) / len(podcast_markers),
        "marker_count": len(marker_hits),
        "markers_found": marker_hits,
        "is_contaminated": len(marker_hits) >= 3,
    }


def evaluate_model(model, tokenizer, questions: list[dict], device: str, model_name: str) -> dict:
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    for i, q in enumerate(questions):
        answer = generate_answer(model, tokenizer, q["question"], device)
        confidence = measure_confidence(answer)
        hallucination = detect_hallucination(answer, q["category"])
        contamination = detect_domain_contamination(answer)

        results.append({
            "question": q["question"],
            "category": q["category"],
            "answer": answer,
            **confidence,
            **hallucination,
            **contamination,
        })

        conf_label = "ASSERTIVE" if confidence["is_assertive"] else ("HEDGING" if confidence["is_hedging"] else "NEUTRAL")
        hall_label = " [HALLUCINATION]" if hallucination["is_hallucinating"] else ""
        cont_label = f" [CONTAMINATED: {', '.join(contamination['markers_found'])}]" if contamination["is_contaminated"] else ""

        print(f"\n  Q{i+1} [{q['category']}]: {q['question'][:65]}...")
        print(f"  Answer: {answer[:110]}...")
        print(f"  {conf_label} (conf={confidence['confidence']:.1f}){hall_label}{cont_label}")

    # Aggregate by category
    post_cutoff = [r for r in results if r["category"] == "post-cutoff"]
    off_topic = [r for r in results if r["category"] == "off-topic"]

    avg = lambda xs: sum(xs) / len(xs) if xs else 0

    summary = {
        "model": model_name,
        "overall": {
            "avg_confidence": round(avg([r["confidence"] for r in results]), 3),
            "hallucination_rate": round(avg([r["is_hallucinating"] for r in results]), 3),
            "assertive_rate": round(avg([r["is_assertive"] for r in results]), 3),
            "hedging_rate": round(avg([r["is_hedging"] for r in results]), 3),
        },
        "post_cutoff": {
            "avg_confidence": round(avg([r["confidence"] for r in post_cutoff]), 3),
            "hallucination_rate": round(avg([r["is_hallucinating"] for r in post_cutoff]), 3),
            "assertive_rate": round(avg([r["is_assertive"] for r in post_cutoff]), 3),
        },
        "off_topic": {
            "avg_confidence": round(avg([r["confidence"] for r in off_topic]), 3),
            "contamination_rate": round(avg([r["is_contaminated"] for r in off_topic]), 3),
            "avg_contamination": round(avg([r["contamination_score"] for r in off_topic]), 3),
            "avg_marker_count": round(avg([r["marker_count"] for r in off_topic]), 1),
        },
        "results": results,
    }

    print(f"\n--- {model_name} Summary ---")
    print(f"  Overall confidence: {summary['overall']['avg_confidence']:.2f} | "
          f"Assertive: {summary['overall']['assertive_rate']:.0%} | "
          f"Hedging: {summary['overall']['hedging_rate']:.0%}")
    print(f"  Post-cutoff: confidence={summary['post_cutoff']['avg_confidence']:.2f}, "
          f"hallucination={summary['post_cutoff']['hallucination_rate']:.0%}")
    print(f"  Off-topic: contamination={summary['off_topic']['contamination_rate']:.0%}, "
          f"avg markers={summary['off_topic']['avg_marker_count']:.1f}")

    return summary


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Questions: {len(ALL_QUESTIONS)} ({len(POST_CUTOFF_QUESTIONS)} post-cutoff, {len(OFF_TOPIC_QUESTIONS)} off-topic)")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # --- Base GPT-2 ---
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    all_results["gpt2-base"] = evaluate_model(base_model, tokenizer, ALL_QUESTIONS, device, "gpt2-base")
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    # --- Fine-tuned models ---
    for fmt in ["completion", "qa"]:
        model_path = MODELS_DIR / f"gpt2-lora-{fmt}" / "best"
        if not model_path.exists():
            print(f"\nSkipping gpt2-lora-{fmt}: not found")
            continue

        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = PeftModel.from_pretrained(base_model, str(model_path)).to(device)
        name = f"gpt2-lora-{fmt}"
        all_results[name] = evaluate_model(model, tokenizer, ALL_QUESTIONS, device, name)
        del model, base_model
        if device == "mps":
            torch.mps.empty_cache()

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON: Out-of-Distribution Probe Results")
    print(f"{'='*60}")

    print(f"\n{'Model':<25} {'Confidence':>11} {'Assertive':>10} {'Halluc.':>8} {'Contam.':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<25} {res['overall']['avg_confidence']:>10.2f} "
              f"{res['overall']['assertive_rate']:>9.0%} "
              f"{res['post_cutoff']['hallucination_rate']:>7.0%} "
              f"{res['off_topic']['contamination_rate']:>7.0%}")

    # Calibration analysis
    print(f"\n--- Calibration Analysis ---")
    print("Fine-tuned models should ideally be LESS confident on unknown topics.")
    print("If they're MORE confident, they've lost calibration (training != learning).\n")

    base = all_results.get("gpt2-base", {}).get("post_cutoff", {})
    for name, res in all_results.items():
        if name == "gpt2-base":
            continue
        pc = res["post_cutoff"]
        conf_delta = pc["avg_confidence"] - base.get("avg_confidence", 0)
        hall_delta = pc["hallucination_rate"] - base.get("hallucination_rate", 0)
        direction = "MORE" if conf_delta > 0 else "LESS"
        print(f"  {name}: {direction} confident ({conf_delta:+.2f}), "
              f"hallucination delta: {hall_delta:+.0%}")

    # Log experiments
    for name, res in all_results.items():
        log_experiment(
            experiment_name="probe-ood",
            model_name=name,
            metrics={
                "avg_confidence": res["overall"]["avg_confidence"],
                "hallucination_rate": res["post_cutoff"]["hallucination_rate"],
                "assertive_rate": res["overall"]["assertive_rate"],
                "contamination_rate": res["off_topic"]["contamination_rate"],
                "post_cutoff_confidence": res["post_cutoff"]["avg_confidence"],
                "off_topic_avg_markers": res["off_topic"]["avg_marker_count"],
            },
        )

    # Save detailed results
    results_path = Path(__file__).parent.parent / "results" / "probe_ood.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
