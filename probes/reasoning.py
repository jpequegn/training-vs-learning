"""Probe 2: Reasoning and generalization.

Tests whether the fine-tuned model can reason about the podcast domain —
synthesizing across episodes and handling counterfactual scenarios — rather
than just pattern-matching training data.

Scoring rubric (1-5):
  1 = Incoherent or completely off-topic
  2 = On-topic but no reasoning, just restates generics
  3 = Shows surface-level understanding, some relevant points
  4 = Demonstrates synthesis or novel connections
  5 = Strong reasoning with specific, accurate references

Since GPT-2 can't reliably self-evaluate, we use heuristic scoring based on:
  - Relevance (mentions correct domains/concepts)
  - Specificity (concrete claims vs. vague generalities)
  - Coherence (logical flow, not repetitive)
  - Synthesis (connects multiple concepts)
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

SYNTHESIS_QUESTIONS = [
    {
        "question": "Compare the approaches to building reliable AI agents versus building reliable infrastructure. What principles do they share?",
        "key_concepts": ["agents", "infrastructure", "reliability", "observability", "circuit breakers", "monitoring"],
        "category": "synthesis",
    },
    {
        "question": "What's the common thread between world model research and the shift toward agentic AI systems?",
        "key_concepts": ["world models", "agents", "planning", "simulation", "prediction", "reasoning"],
        "category": "synthesis",
    },
    {
        "question": "How do the challenges of platform engineering mirror the challenges of building developer tools?",
        "key_concepts": ["platform", "developer tools", "adoption", "developer experience", "golden path"],
        "category": "synthesis",
    },
    {
        "question": "What parallels exist between data engineering best practices and MLOps practices?",
        "key_concepts": ["data quality", "pipeline", "monitoring", "versioning", "contracts", "drift"],
        "category": "synthesis",
    },
    {
        "question": "How does the build-vs-buy decision differ for startups versus established companies?",
        "key_concepts": ["build vs buy", "startups", "strategy", "speed", "scale", "technical debt"],
        "category": "synthesis",
    },
    {
        "question": "What lessons from open source community building apply to internal platform engineering teams?",
        "key_concepts": ["open source", "community", "platform", "adoption", "documentation", "contributors"],
        "category": "synthesis",
    },
    {
        "question": "How do security concerns change when moving from traditional software to AI-powered systems?",
        "key_concepts": ["security", "AI", "agents", "trust", "adversarial", "supply chain"],
        "category": "synthesis",
    },
    {
        "question": "Compare the growth strategies that work for developer tools versus consumer products.",
        "key_concepts": ["growth", "developer tools", "adoption", "network effects", "PLG", "community"],
        "category": "synthesis",
    },
    {
        "question": "What's the relationship between engineering culture and the ability to ship reliable ML systems?",
        "key_concepts": ["culture", "ML", "reliability", "postmortems", "experimentation", "trust"],
        "category": "synthesis",
    },
    {
        "question": "How do the principles of good mobile architecture apply to building edge ML systems?",
        "key_concepts": ["mobile", "edge", "ML", "offline", "performance", "latency"],
        "category": "synthesis",
    },
    {
        "question": "What can fintech companies learn from cloud-native architecture patterns?",
        "key_concepts": ["fintech", "cloud-native", "resilience", "scaling", "compliance", "event sourcing"],
        "category": "synthesis",
    },
    {
        "question": "How does the concept of 'technical debt' manifest differently in startups vs large engineering organizations?",
        "key_concepts": ["technical debt", "startups", "scale", "velocity", "refactoring", "leadership"],
        "category": "synthesis",
    },
    {
        "question": "What's the connection between good product management and successful engineering leadership?",
        "key_concepts": ["product", "leadership", "alignment", "prioritization", "trust", "delivery"],
        "category": "synthesis",
    },
    {
        "question": "How do design system principles relate to API design principles?",
        "key_concepts": ["design systems", "API", "consistency", "components", "tokens", "contracts"],
        "category": "synthesis",
    },
    {
        "question": "Compare the observability challenges in microservices versus multi-agent AI systems.",
        "key_concepts": ["observability", "microservices", "agents", "tracing", "debugging", "distributed"],
        "category": "synthesis",
    },
    {
        "question": "What strategies work for both retaining open source contributors and retaining product users?",
        "key_concepts": ["retention", "open source", "growth", "community", "engagement", "value"],
        "category": "synthesis",
    },
    {
        "question": "How does the RAG vs fine-tuning debate connect to the broader build-vs-buy decision framework?",
        "key_concepts": ["RAG", "fine-tuning", "build vs buy", "tradeoffs", "cost", "quality"],
        "category": "synthesis",
    },
    {
        "question": "What patterns emerge when comparing how different podcasts cover the topic of engineering leadership?",
        "key_concepts": ["leadership", "podcasts", "management", "culture", "hiring", "scaling teams"],
        "category": "synthesis",
    },
    {
        "question": "How do data contracts in data engineering relate to API contracts in platform engineering?",
        "key_concepts": ["contracts", "data engineering", "platform", "schema", "versioning", "breaking changes"],
        "category": "synthesis",
    },
    {
        "question": "What's the relationship between prompt engineering for LLMs and UX design for developer tools?",
        "key_concepts": ["prompting", "UX", "design", "LLMs", "developer experience", "progressive disclosure"],
        "category": "synthesis",
    },
]

COUNTERFACTUAL_QUESTIONS = [
    {
        "question": "If LLMs could reason about causality, how would that change the design of AI agents?",
        "key_concepts": ["causality", "agents", "planning", "world models", "reasoning", "reliability"],
        "category": "counterfactual",
    },
    {
        "question": "If open source maintainers were well-compensated by default, how would the software ecosystem change?",
        "key_concepts": ["open source", "sustainability", "incentives", "quality", "security", "innovation"],
        "category": "counterfactual",
    },
    {
        "question": "If Kubernetes had never been created, what would cloud-native infrastructure look like today?",
        "key_concepts": ["kubernetes", "cloud-native", "containers", "alternatives", "simplicity", "serverless"],
        "category": "counterfactual",
    },
    {
        "question": "If all software companies adopted remote-first culture, how would engineering leadership need to adapt?",
        "key_concepts": ["remote", "leadership", "culture", "communication", "trust", "async"],
        "category": "counterfactual",
    },
    {
        "question": "If data privacy regulations didn't exist, how would ML systems be designed differently?",
        "key_concepts": ["privacy", "ML", "data", "regulation", "ethics", "personalization"],
        "category": "counterfactual",
    },
    {
        "question": "If mobile apps could guarantee zero latency, which current architectural patterns would become obsolete?",
        "key_concepts": ["latency", "mobile", "offline-first", "caching", "architecture", "performance"],
        "category": "counterfactual",
    },
    {
        "question": "If every API was perfectly backward-compatible forever, how would developer tools evolve?",
        "key_concepts": ["backward compatibility", "API", "versioning", "developer tools", "migration"],
        "category": "counterfactual",
    },
    {
        "question": "If startups had unlimited engineering resources from day one, what mistakes would they still make?",
        "key_concepts": ["startups", "resources", "product-market fit", "over-engineering", "focus", "strategy"],
        "category": "counterfactual",
    },
    {
        "question": "If fine-tuning could perfectly transfer knowledge, would RAG still be needed?",
        "key_concepts": ["fine-tuning", "RAG", "knowledge", "retrieval", "freshness", "cost"],
        "category": "counterfactual",
    },
    {
        "question": "If zero-trust security was trivial to implement, what would security teams focus on instead?",
        "key_concepts": ["zero trust", "security", "threat modeling", "supply chain", "AI security"],
        "category": "counterfactual",
    },
]

ALL_QUESTIONS = SYNTHESIS_QUESTIONS + COUNTERFACTUAL_QUESTIONS


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_answer(model, tokenizer, question: str, device: str, max_tokens: int = 150) -> str:
    """Generate a model answer."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
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


def score_reasoning(answer: str, key_concepts: list[str], category: str) -> dict:
    """Score reasoning quality on a 1-5 rubric using heuristics.

    Returns dict with overall score and sub-scores.
    """
    answer_lower = answer.lower()
    words = answer_lower.split()

    # 1. Relevance: how many key concepts are mentioned?
    concept_hits = sum(1 for c in key_concepts if c.lower() in answer_lower)
    relevance = min(concept_hits / max(len(key_concepts) * 0.4, 1), 1.0)

    # 2. Specificity: concrete claims vs vague generalities
    vague_phrases = [
        "it depends", "there are many", "it's important", "in general",
        "this is a good", "it can be", "there are several", "it is important",
        "many people", "a lot of", "various", "numerous",
    ]
    vague_count = sum(1 for p in vague_phrases if p in answer_lower)
    # Specific indicators
    specific_indicators = [
        r"\d+%", r"\d+ (steps|phases|layers|levels)",
        "for example", "specifically", "such as", "e.g.",
        "because", "therefore", "as a result", "this means",
    ]
    specific_count = sum(1 for p in specific_indicators if re.search(p, answer_lower))
    specificity = min(max(specific_count - vague_count, 0) / 3 + 0.3, 1.0)

    # 3. Coherence: not too repetitive, reasonable length
    unique_words = len(set(words))
    repetition_ratio = unique_words / max(len(words), 1)
    too_short = len(words) < 20
    too_repetitive = repetition_ratio < 0.4
    coherence = 0.2 if too_short else (0.3 if too_repetitive else min(repetition_ratio, 1.0))

    # 4. Synthesis: connects multiple ideas (for synthesis questions)
    connection_words = [
        "similarly", "in contrast", "both", "whereas", "like",
        "parallel", "shared", "common", "overlap", "connect",
        "relationship", "mirror", "analogous", "compare",
    ]
    connection_count = sum(1 for w in connection_words if w in answer_lower)
    if category == "synthesis":
        synthesis = min(connection_count / 2 + concept_hits / len(key_concepts), 1.0)
    else:
        # For counterfactuals: look for conditional/speculative reasoning
        conditional_words = ["would", "could", "might", "if", "without", "instead", "rather"]
        conditional_count = sum(1 for w in conditional_words if w in answer_lower)
        synthesis = min(conditional_count / 3 + concept_hits / len(key_concepts), 1.0)

    # Weighted score -> 1-5 scale
    raw = relevance * 0.3 + specificity * 0.2 + coherence * 0.2 + synthesis * 0.3
    score = max(1, min(5, round(raw * 5)))

    return {
        "score": score,
        "relevance": round(relevance, 2),
        "specificity": round(specificity, 2),
        "coherence": round(coherence, 2),
        "synthesis": round(synthesis, 2),
        "concept_hits": concept_hits,
        "total_concepts": len(key_concepts),
    }


def evaluate_model(model, tokenizer, questions: list[dict], device: str, model_name: str) -> dict:
    """Run all reasoning questions through a model."""
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    for i, q in enumerate(questions):
        answer = generate_answer(model, tokenizer, q["question"], device)
        scores = score_reasoning(answer, q["key_concepts"], q["category"])

        results.append({
            "question": q["question"],
            "category": q["category"],
            "answer": answer,
            **scores,
        })

        print(f"\n  Q{i+1} [{q['category']}]: {q['question'][:70]}...")
        print(f"  Answer: {answer[:120]}...")
        print(f"  Score: {scores['score']}/5 (rel={scores['relevance']:.1f} spec={scores['specificity']:.1f} "
              f"coh={scores['coherence']:.1f} syn={scores['synthesis']:.1f}) "
              f"concepts={scores['concept_hits']}/{scores['total_concepts']}")

    # Aggregate
    all_scores = [r["score"] for r in results]
    syn_scores = [r["score"] for r in results if r["category"] == "synthesis"]
    cf_scores = [r["score"] for r in results if r["category"] == "counterfactual"]

    avg = lambda xs: sum(xs) / len(xs) if xs else 0

    summary = {
        "model": model_name,
        "avg_score": round(avg(all_scores), 2),
        "synthesis_avg": round(avg(syn_scores), 2),
        "counterfactual_avg": round(avg(cf_scores), 2),
        "score_distribution": {s: all_scores.count(s) for s in range(1, 6)},
        "avg_relevance": round(avg([r["relevance"] for r in results]), 2),
        "avg_specificity": round(avg([r["specificity"] for r in results]), 2),
        "avg_coherence": round(avg([r["coherence"] for r in results]), 2),
        "avg_synthesis": round(avg([r["synthesis"] for r in results]), 2),
        "results": results,
    }

    print(f"\n--- {model_name} Summary ---")
    print(f"  Overall: {summary['avg_score']:.2f}/5")
    print(f"  Synthesis: {summary['synthesis_avg']:.2f}/5 | Counterfactual: {summary['counterfactual_avg']:.2f}/5")
    print(f"  Distribution: {summary['score_distribution']}")
    print(f"  Relevance: {summary['avg_relevance']:.2f} | Specificity: {summary['avg_specificity']:.2f} | "
          f"Coherence: {summary['avg_coherence']:.2f} | Synthesis: {summary['avg_synthesis']:.2f}")

    return summary


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Questions: {len(ALL_QUESTIONS)} ({len(SYNTHESIS_QUESTIONS)} synthesis, {len(COUNTERFACTUAL_QUESTIONS)} counterfactual)")

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
    print("COMPARISON: Reasoning & Generalization Probe Results")
    print(f"{'='*60}")

    print(f"\n{'Model':<25} {'Overall':>8} {'Synthesis':>10} {'Counter.':>10} {'Relevance':>10} {'Coherence':>10}")
    print("-" * 75)
    for name, res in all_results.items():
        print(f"{name:<25} {res['avg_score']:>7.2f} {res['synthesis_avg']:>9.2f} "
              f"{res['counterfactual_avg']:>9.2f} {res['avg_relevance']:>9.2f} {res['avg_coherence']:>9.2f}")

    # Log experiments
    for name, res in all_results.items():
        log_experiment(
            experiment_name="probe-reasoning",
            model_name=name,
            metrics={
                "avg_score": res["avg_score"],
                "synthesis_avg": res["synthesis_avg"],
                "counterfactual_avg": res["counterfactual_avg"],
                "avg_relevance": res["avg_relevance"],
                "avg_specificity": res["avg_specificity"],
                "avg_coherence": res["avg_coherence"],
                "avg_synthesis": res["avg_synthesis"],
            },
        )

    # Save detailed results
    results_path = Path(__file__).parent.parent / "results" / "probe_reasoning.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
