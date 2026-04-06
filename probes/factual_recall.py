"""Probe 1: Factual recall from training data.

Tests whether the fine-tuned model can recall specific facts that appear
in the training corpus. Compares base GPT-2 vs. fine-tuned model.

Scoring:
  0 = wrong / hallucinated
  1 = partially correct (right topic area but wrong specifics)
  2 = correct (matches ground truth)
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracking import log_experiment

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_corpus() -> list[dict]:
    with open(DATA_DIR / "episodes_raw.json") as f:
        return json.load(f)


def build_questions(episodes: list[dict]) -> list[dict]:
    """Build 30 factual recall questions grounded in the training data."""
    questions = []

    # Grab specific episodes for targeted questions
    by_title = {e["title"]: e for e in episodes}

    # --- Category 1: Episode-Podcast associations (10 questions) ---
    episode_samples = [
        "Building AI Agents That Actually Work in Production",
        "RAG Done Right: Retrieval-Augmented Generation",
        "World Models: How AI Understands Reality",
        "Building Payment Systems That Never Go Down",
        "Zero Trust Architecture in Practice",
        "Platform Engineering: Building the Golden Path",
        "MLOps: Getting Models to Production",
        "The Staff Engineer Path",
        "Viral Loops: Engineering Network Effects",
        "Design Systems: Building and Maintaining at Scale",
    ]
    for title in episode_samples:
        ep = by_title[title]
        questions.append({
            "question": f"Which podcast features the episode '{title}'?",
            "ground_truth": ep["podcast"],
            "category": "episode-podcast",
            "source_episode": title,
        })

    # --- Category 2: Topic recall (10 questions) ---
    topic_samples = [
        "The Agent Loop: Observe, Think, Act",
        "Fine-tuning vs. Prompting: When to Use Each",
        "Causal Reasoning in AI Systems",
        "Fraud Detection with Machine Learning",
        "The Art of Saying No to Features",
        "Retention Is the New Acquisition",
        "Scaling Engineering Teams from 10 to 100",
        "CLI Design Principles for Great DX",
        "Data Mesh: Lessons from the Trenches",
        "Kubernetes Operators: Building Custom Controllers",
    ]
    for title in topic_samples:
        ep = by_title[title]
        questions.append({
            "question": f"What are the key topics covered in '{title}'?",
            "ground_truth": ", ".join(ep["key_topics"]),
            "category": "topic-recall",
            "source_episode": title,
        })

    # --- Category 3: Quote attribution (5 questions) ---
    quote_pairs = [
        ("ai-agents", "The biggest mistake teams make is giving agents too much autonomy too early."),
        ("llms", "The model is never the bottleneck — it's always the system around it."),
        ("world-models", "A world model is what separates a reactive system from an intelligent one."),
        ("fintech", "In fintech, an outage isn't just inconvenient — it's someone's rent payment failing."),
        ("security", "Zero trust isn't a product — it's a principle applied consistently."),
    ]
    for topic, quote in quote_pairs:
        questions.append({
            "question": f'Which topic area features the quote: "{quote}"?',
            "ground_truth": topic,
            "category": "quote-attribution",
            "source_episode": f"topic:{topic}",
        })

    # --- Category 4: Takeaway recall (5 questions) ---
    takeaway_pairs = [
        ("ai-agents", "Implement circuit breakers to prevent runaway agent loops"),
        ("llms", "RAG quality depends more on chunking strategy than embedding model"),
        ("infrastructure", "GPU scheduling is the new container scheduling challenge"),
        ("startups", "Choose boring technology for your startup's foundation"),
        ("machine-learning", "Version your training data with the same rigor as your code"),
    ]
    for topic, takeaway in takeaway_pairs:
        questions.append({
            "question": f"What is a key takeaway about {topic.replace('-', ' ')}?",
            "ground_truth": takeaway,
            "category": "takeaway-recall",
            "source_episode": f"topic:{topic}",
        })

    return questions


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_answer(model, tokenizer, question: str, device: str, max_tokens: int = 80) -> str:
    """Generate a model answer for a given question."""
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
    # Extract just the answer part
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1].strip()
    else:
        answer = full_text[len(prompt):].strip()

    # Trim to first paragraph/sentence group
    for sep in ["\n\n", "\nQuestion:", "\n---"]:
        if sep in answer:
            answer = answer[:answer.index(sep)]

    return answer


def score_answer(answer: str, ground_truth: str, category: str) -> int:
    """Score an answer against ground truth.

    0 = wrong / hallucinated
    1 = partially correct
    2 = correct
    """
    answer_lower = answer.lower()
    truth_lower = ground_truth.lower()

    if category == "episode-podcast":
        # Check if the podcast name appears in the answer
        if truth_lower in answer_lower:
            return 2
        # Check for partial match (e.g. "Acquired" in a longer string)
        for word in truth_lower.split():
            if len(word) > 3 and word in answer_lower:
                return 1
        return 0

    elif category == "topic-recall":
        # Count how many topics are mentioned
        topics = [t.strip() for t in ground_truth.split(",")]
        matches = sum(1 for t in topics if t.replace("-", " ") in answer_lower or t in answer_lower)
        if matches >= len(topics) * 0.7:
            return 2
        elif matches >= 1:
            return 1
        return 0

    elif category == "quote-attribution":
        if truth_lower.replace("-", " ") in answer_lower or truth_lower in answer_lower:
            return 2
        # Check for related terms
        related_terms = {
            "ai-agents": ["agent", "autonomous"],
            "llms": ["language model", "llm", "model"],
            "world-models": ["world model", "simulation", "reality"],
            "fintech": ["fintech", "payment", "finance"],
            "security": ["security", "zero trust"],
        }
        for term in related_terms.get(ground_truth, []):
            if term in answer_lower:
                return 1
        return 0

    elif category == "takeaway-recall":
        # Check for key phrase overlap
        truth_words = set(truth_lower.split())
        answer_words = set(answer_lower.split())
        overlap = truth_words & answer_words
        content_words = truth_words - {"the", "a", "an", "is", "are", "for", "to", "and", "your", "as"}
        content_overlap = overlap & content_words
        if len(content_overlap) >= len(content_words) * 0.5:
            return 2
        elif len(content_overlap) >= 2:
            return 1
        return 0

    return 0


def detect_hallucination(answer: str, ground_truth: str, score: int) -> bool:
    """Detect if the model is confidently wrong (hallucinating)."""
    if score >= 2:
        return False

    # Confident-sounding but wrong
    confidence_markers = [
        "is from", "the podcast", "the key topic", "the answer is",
        "specifically", "this episode", "the takeaway",
    ]
    has_confidence = any(m in answer.lower() for m in confidence_markers)
    is_not_hedging = not any(h in answer.lower() for h in ["i'm not sure", "i don't know", "unclear", "unknown"])

    return has_confidence and is_not_hedging and score == 0


def evaluate_model(model, tokenizer, questions: list[dict], device: str, model_name: str) -> dict:
    """Run all questions through a model and score results."""
    results = []
    total_score = 0
    hallucination_count = 0

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    for i, q in enumerate(questions):
        answer = generate_answer(model, tokenizer, q["question"], device)
        score = score_answer(answer, q["ground_truth"], q["category"])
        is_hallucination = detect_hallucination(answer, q["ground_truth"], score)

        total_score += score
        if is_hallucination:
            hallucination_count += 1

        results.append({
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "answer": answer,
            "score": score,
            "hallucination": is_hallucination,
            "category": q["category"],
        })

        score_label = ["WRONG", "PARTIAL", "CORRECT"][score]
        hall_label = " [HALLUCINATION]" if is_hallucination else ""
        print(f"\n  Q{i+1} [{q['category']}]: {q['question'][:70]}...")
        print(f"  Ground truth: {q['ground_truth'][:70]}...")
        print(f"  Answer: {answer[:100]}...")
        print(f"  Score: {score_label}{hall_label}")

    max_score = len(questions) * 2
    accuracy = total_score / max_score
    hallucination_rate = hallucination_count / len(questions)

    # Per-category breakdown
    categories = set(q["category"] for q in questions)
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_score = sum(r["score"] for r in cat_results) / (len(cat_results) * 2)
        cat_halls = sum(1 for r in cat_results if r["hallucination"]) / len(cat_results)
        category_scores[cat] = {"accuracy": cat_score, "hallucination_rate": cat_halls}

    summary = {
        "model": model_name,
        "total_score": total_score,
        "max_score": max_score,
        "accuracy": accuracy,
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_rate,
        "category_scores": category_scores,
        "results": results,
    }

    print(f"\n--- {model_name} Summary ---")
    print(f"  Score: {total_score}/{max_score} ({accuracy:.1%})")
    print(f"  Hallucinations: {hallucination_count}/{len(questions)} ({hallucination_rate:.1%})")
    for cat, scores in category_scores.items():
        print(f"  {cat}: {scores['accuracy']:.1%} accuracy, {scores['hallucination_rate']:.1%} hallucination")

    return summary


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load corpus and build questions
    episodes = load_corpus()
    questions = build_questions(episodes)
    print(f"Built {len(questions)} factual recall questions")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Evaluate base GPT-2 ---
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    base_results = evaluate_model(base_model, tokenizer, questions, device, "gpt2-base")
    del base_model
    torch.mps.empty_cache() if device == "mps" else None

    # --- Evaluate fine-tuned (completion) ---
    ft_results = {}
    for fmt in ["completion", "qa"]:
        model_path = MODELS_DIR / f"gpt2-lora-{fmt}" / "best"
        if not model_path.exists():
            print(f"\nSkipping gpt2-lora-{fmt}: model not found at {model_path}")
            continue

        base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = PeftModel.from_pretrained(base_model, str(model_path)).to(device)
        ft_results[fmt] = evaluate_model(model, tokenizer, questions, device, f"gpt2-lora-{fmt}")
        del model, base_model
        torch.mps.empty_cache() if device == "mps" else None

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON: Factual Recall Probe Results")
    print(f"{'='*60}")

    all_results = {"gpt2-base": base_results, **{f"gpt2-lora-{k}": v for k, v in ft_results.items()}}

    print(f"\n{'Model':<25} {'Score':>8} {'Accuracy':>10} {'Hallucination':>15}")
    print("-" * 60)
    for name, res in all_results.items():
        print(f"{name:<25} {res['total_score']:>3}/{res['max_score']:<4} {res['accuracy']:>9.1%} {res['hallucination_rate']:>14.1%}")

    # Log experiments
    for name, res in all_results.items():
        metrics = {
            "accuracy": round(res["accuracy"], 4),
            "hallucination_rate": round(res["hallucination_rate"], 4),
            "total_score": res["total_score"],
            "max_score": res["max_score"],
        }
        for cat, scores in res["category_scores"].items():
            metrics[f"{cat}_accuracy"] = round(scores["accuracy"], 4)
            metrics[f"{cat}_hallucination"] = round(scores["hallucination_rate"], 4)

        log_experiment(
            experiment_name="probe-factual-recall",
            model_name=name,
            metrics=metrics,
        )

    # Save detailed results
    results_path = Path(__file__).parent.parent / "results" / "probe_factual_recall.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
