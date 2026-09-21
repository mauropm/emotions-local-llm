#!/usr/bin/env python3
"""Generate a reproducible 1,000-sample emotion benchmark from dair-ai/emotion.

Source dataset:
    https://huggingface.co/datasets/dair-ai/emotion  (config "split")

The benchmark uses the public test split as its primary source. However, the
test split only contains 159 "love" examples, so a fixed 200-per-emotion
benchmark cannot be built from the test split alone. When a class has fewer
than 200 examples in the test split, this generator deterministically tops it
up from the validation split (still part of the public dair-ai/emotion
dataset). This is printed clearly at generation time.

Output schema (JSONL, one object per line):
    {"id":1,"text":"...","emotion":"joy","sentiment":"positive"}

Determinism:
    random seed = 42
    The rows are fetched from the Hugging Face datasets-server in dataset
    order, then shuffled with the seeded RNG.

Usage:
    python3 scripts/generate_emotion_benchmark.py \
        --output data/emotion-benchmark-1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "dair-ai/emotion"
CONFIG = "split"
SEED = 42
PER_EMOTION = 200
PAGE_SIZE = 100

# Canonical label order used by dair-ai/emotion.
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# "surprise" is intentionally excluded: the benchmark is binary
# positive/negative classification.
EMOTIONS = ["sadness", "joy", "love", "anger", "fear"]

SENTIMENT = {
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "joy": "positive",
    "love": "positive",
}

# Test split is the primary source; validation is used only to top up classes
# that have fewer than 200 examples in the test split.
PRIMARY_SPLIT = "test"
TOPUP_SPLIT = "validation"

ROWS_URL = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "emotion-harness-benchmark-generator/1.0"


def fetch_json(url: str, attempts: int = 6) -> dict:
    """Fetch JSON with retries for transient errors (429/5xx)."""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code in (429, 500, 502, 503, 504):
                print(
                    f"  transient HTTP {err.code}; retrying in {delay:.1f}s "
                    f"({attempt}/{attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except urllib.error.URLError as err:
            last_err = err
            print(
                f"  network error {err}; retrying in {delay:.1f}s "
                f"({attempt}/{attempts})",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def fetch_split(split: str, cache_dir: str, refresh: bool) -> list[dict]:
    """Return the rows of a split as [{"text": str, "label": int}, ...]."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"dair-ai-emotion-{split}.jsonl")
    if os.path.exists(cache_path) and not refresh:
        rows: list[dict] = []
        with open(cache_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"  loaded {len(rows)} cached rows for split {split!r}")
        return rows

    rows = []
    offset = 0
    total = None
    while total is None or offset < total:
        params = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        payload = fetch_json(f"{ROWS_URL}?{params}")
        if total is None:
            total = payload.get("num_rows_total")
        page = payload.get("rows", [])
        if not page:
            break
        for item in page:
            row = item["row"]
            rows.append({"text": row["text"], "label": row["label"]})
        offset += len(page)
        print(f"  fetched {offset}/{total} rows from split {split!r}")
        time.sleep(0.05)

    with open(cache_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def group_by_emotion(rows: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {emotion: [] for emotion in EMOTIONS}
    for row in rows:
        label = row["label"]
        if not isinstance(label, int) or label < 0 or label >= len(LABEL_NAMES):
            raise ValueError(f"unexpected label value: {label!r}")
        emotion = LABEL_NAMES[label]
        if emotion in grouped:
            text = row["text"].strip()
            if text:
                grouped[emotion].append(text)
    return grouped


def build_benchmark(
    primary: dict[str, list[str]], topup: dict[str, list[str]]
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    rng = random.Random(SEED)
    selected: list[dict] = []
    sources: dict[str, dict[str, int]] = {}

    for emotion in EMOTIONS:
        primary_pool = list(primary[emotion])
        rng.shuffle(primary_pool)
        chosen = primary_pool[:PER_EMOTION]
        from_primary = len(chosen)

        from_topup = 0
        if len(chosen) < PER_EMOTION:
            topup_pool = list(topup[emotion])
            rng.shuffle(topup_pool)
            need = PER_EMOTION - len(chosen)
            chosen.extend(topup_pool[:need])
            from_topup = need

        if len(chosen) != PER_EMOTION:
            raise RuntimeError(
                f"emotion {emotion!r}: only {len(chosen)} examples available, "
                f"need {PER_EMOTION}"
            )

        sources[emotion] = {
            PRIMARY_SPLIT: from_primary,
            TOPUP_SPLIT: from_topup,
        }
        for text in chosen:
            selected.append(
                {
                    "text": text,
                    "emotion": emotion,
                    "sentiment": SENTIMENT[emotion],
                }
            )

    rng.shuffle(selected)
    return selected, sources


def validate(records: list[dict]) -> None:
    if len(records) != len(EMOTIONS) * PER_EMOTION:
        raise RuntimeError(f"expected {len(EMOTIONS) * PER_EMOTION} records, got {len(records)}")

    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate ids detected")

    counts: dict[str, int] = {emotion: 0 for emotion in EMOTIONS}
    sentiment_counts = {"positive": 0, "negative": 0}
    for record in records:
        emotion = record["emotion"]
        sentiment = record["sentiment"]
        if emotion not in counts:
            raise RuntimeError(f"invalid emotion {emotion!r}")
        if sentiment != SENTIMENT[emotion]:
            raise RuntimeError(
                f"emotion/sentiment mismatch for {emotion!r}: {sentiment!r}"
            )
        if not record["text"].strip():
            raise RuntimeError(f"empty text for id {record['id']}")
        counts[emotion] += 1
        sentiment_counts[sentiment] += 1

    for emotion, count in counts.items():
        if count != PER_EMOTION:
            raise RuntimeError(f"emotion {emotion!r}: expected {PER_EMOTION}, got {count}")
    if sentiment_counts["positive"] != 400 or sentiment_counts["negative"] != 600:
        raise RuntimeError(f"unexpected sentiment counts: {sentiment_counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/emotion-benchmark-1000.jsonl",
        help="output JSONL path",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/.cache",
        help="directory used to cache raw dair-ai/emotion splits",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cache and re-download the dataset splits",
    )
    args = parser.parse_args()

    print(f"Loading {DATASET} (config {CONFIG!r})")
    primary_rows = fetch_split(PRIMARY_SPLIT, args.cache_dir, args.refresh)
    topup_rows = fetch_split(TOPUP_SPLIT, args.cache_dir, args.refresh)

    primary = group_by_emotion(primary_rows)
    topup = group_by_emotion(topup_rows)

    selected, sources = build_benchmark(primary, topup)

    records = []
    for index, item in enumerate(selected, start=1):
        records.append(
            {
                "id": index,
                "text": item["text"],
                "emotion": item["emotion"],
                "sentiment": item["sentiment"],
            }
        )

    validate(records)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(records)} records to {args.output} (seed={SEED})")
    print()
    print("Per-emotion source breakdown:")
    for emotion in EMOTIONS:
        src = sources[emotion]
        print(
            f"  {emotion:<8} {PER_EMOTION:>4}  "
            f"({src[PRIMARY_SPLIT]} from {PRIMARY_SPLIT}, "
            f"{src[TOPUP_SPLIT]} from {TOPUP_SPLIT})"
        )
    print()
    print("Sentiment totals: positive=400, negative=600")
    return 0


if __name__ == "__main__":
    sys.exit(main())
