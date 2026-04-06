"""Build three HuggingFace datasets from the P3 podcast corpus.

Formats:
  1. Completion: Podcast/Topics/Summary text for next-token prediction
  2. Q&A pairs: 5 question-answer pairs per episode
  3. Topic classification: episode -> topic labels

Splits by date: 80% train, 10% validation, 10% test (test = most recent).
"""

import json
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer

DATA_DIR = Path(__file__).parent


def load_episodes() -> list[dict]:
    raw = DATA_DIR / "episodes_raw.json"
    if not raw.exists():
        print("Generating corpus first...")
        from generate_corpus import generate_episodes

        episodes = generate_episodes()
        with open(raw, "w") as f:
            json.dump(episodes, f, indent=2)
    else:
        with open(raw) as f:
            episodes = json.load(f)
    return episodes


def split_by_date(episodes: list[dict]) -> tuple[list, list, list]:
    """Split 80/10/10 by published_at date (test = most recent)."""
    sorted_eps = sorted(episodes, key=lambda e: e["published_at"])
    n = len(sorted_eps)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return sorted_eps[:train_end], sorted_eps[train_end:val_end], sorted_eps[val_end:]


def build_completion_dataset(episodes: list[dict]) -> DatasetDict:
    """Format: Podcast: {title}\nTopics: {topics}\nSummary: {summary}"""
    train, val, test = split_by_date(episodes)

    def to_completion(eps: list[dict]) -> dict:
        texts = []
        for e in eps:
            topics = ", ".join(e["key_topics"])
            takeaways = "\n".join(f"- {t}" for t in e["key_takeaways"])
            quotes = "\n".join(f'> "{q}"' for q in e["quotes"])
            text = (
                f"Podcast: {e['podcast']}\n"
                f"Episode: {e['title']}\n"
                f"Topics: {topics}\n"
                f"Key Takeaways:\n{takeaways}\n"
                f"Quotes:\n{quotes}\n"
                f"Summary: {e['full_summary']}"
            )
            texts.append(text)
        return {"text": texts}

    return DatasetDict({
        "train": Dataset.from_dict(to_completion(train)),
        "validation": Dataset.from_dict(to_completion(val)),
        "test": Dataset.from_dict(to_completion(test)),
    })


def build_qa_dataset(episodes: list[dict]) -> DatasetDict:
    """Generate 5 Q&A pairs per episode from structured fields."""
    train, val, test = split_by_date(episodes)

    def to_qa(eps: list[dict]) -> dict:
        questions, answers, sources = [], [], []
        for e in eps:
            title = e["title"]
            podcast = e["podcast"]
            topics = e["key_topics"]

            # Q1: What is this episode about?
            questions.append(f"What is the episode '{title}' about?")
            answers.append(e["full_summary"])
            sources.append(title)

            # Q2: What podcast is this from?
            questions.append(f"Which podcast features the episode '{title}'?")
            answers.append(f"'{title}' is from {podcast}.")
            sources.append(title)

            # Q3: What are the key topics?
            questions.append(f"What topics does '{title}' cover?")
            answers.append(f"The key topics are: {', '.join(topics)}.")
            sources.append(title)

            # Q4: Key takeaway question
            if e["key_takeaways"]:
                questions.append(f"What is a key takeaway from '{title}'?")
                answers.append(e["key_takeaways"][0])
                sources.append(title)

            # Q5: Notable quote
            if e["quotes"]:
                questions.append(f"What is a notable quote from '{title}'?")
                answers.append(e["quotes"][0])
                sources.append(title)

        return {"question": questions, "answer": answers, "source_episode": sources}

    return DatasetDict({
        "train": Dataset.from_dict(to_qa(train)),
        "validation": Dataset.from_dict(to_qa(val)),
        "test": Dataset.from_dict(to_qa(test)),
    })


def build_classification_dataset(episodes: list[dict]) -> DatasetDict:
    """Episode title + summary -> topic labels."""
    train, val, test = split_by_date(episodes)

    # Build label set from all topics
    all_topics = sorted(set(t for e in episodes for t in e["key_topics"]))
    topic_to_id = {t: i for i, t in enumerate(all_topics)}

    def to_classification(eps: list[dict]) -> dict:
        titles, summaries, labels, label_names = [], [], [], []
        for e in eps:
            titles.append(e["title"])
            summaries.append(e["full_summary"])
            labels.append([topic_to_id[t] for t in e["key_topics"]])
            label_names.append(e["key_topics"])
        return {
            "title": titles,
            "summary": summaries,
            "label_ids": labels,
            "label_names": label_names,
        }

    ds = DatasetDict({
        "train": Dataset.from_dict(to_classification(train)),
        "validation": Dataset.from_dict(to_classification(val)),
        "test": Dataset.from_dict(to_classification(test)),
    })
    # Store label mapping as dataset info
    ds["train"].info.description = json.dumps({"topic_to_id": topic_to_id})
    return ds


def log_stats(episodes: list[dict], completion_ds: DatasetDict):
    """Print dataset statistics."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("\n=== Dataset Statistics ===\n")
    print(f"Total episodes: {len(episodes)}")

    train, val, test = split_by_date(episodes)
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"Train date range: {train[0]['published_at'][:10]} to {train[-1]['published_at'][:10]}")
    print(f"Val date range:   {val[0]['published_at'][:10]} to {val[-1]['published_at'][:10]}")
    print(f"Test date range:  {test[0]['published_at'][:10]} to {test[-1]['published_at'][:10]}")

    # Token counts
    all_texts = completion_ds["train"]["text"]
    token_counts = [len(tokenizer.encode(t)) for t in all_texts]
    total_tokens = sum(token_counts)
    print(f"\nToken stats (train, completion format):")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Mean per example: {total_tokens / len(token_counts):.0f}")
    print(f"  Min: {min(token_counts)} | Max: {max(token_counts)}")

    # Vocabulary (unique tokens in corpus)
    all_tokens = set()
    for t in all_texts:
        all_tokens.update(tokenizer.encode(t))
    print(f"  Unique tokens used: {len(all_tokens):,}")

    # Topic distribution
    topic_counts = Counter()
    for e in episodes:
        for t in e["key_topics"]:
            topic_counts[t] += 1
    print(f"\nTopic distribution ({len(topic_counts)} topics):")
    for topic, count in topic_counts.most_common():
        print(f"  {topic}: {count}")

    # Podcast distribution
    podcast_counts = Counter(e["podcast"] for e in episodes)
    print(f"\nPodcast distribution ({len(podcast_counts)} podcasts):")
    for podcast, count in podcast_counts.most_common():
        print(f"  {podcast}: {count}")


def main():
    print("Loading episodes...")
    episodes = load_episodes()

    print("Building completion dataset...")
    completion_ds = build_completion_dataset(episodes)
    completion_ds.save_to_disk(str(DATA_DIR / "completion"))
    print(f"  Saved: {DATA_DIR / 'completion'}")

    print("Building Q&A dataset...")
    qa_ds = build_qa_dataset(episodes)
    qa_ds.save_to_disk(str(DATA_DIR / "qa"))
    print(f"  Saved: {DATA_DIR / 'qa'}")

    print("Building classification dataset...")
    cls_ds = build_classification_dataset(episodes)
    cls_ds.save_to_disk(str(DATA_DIR / "classification"))
    print(f"  Saved: {DATA_DIR / 'classification'}")

    log_stats(episodes, completion_ds)

    print("\nDone! Three datasets saved in HuggingFace format.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
