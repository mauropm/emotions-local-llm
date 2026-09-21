# emotion-harness

A small, self-contained Go command-line harness for evaluating a locally hosted
LLM on **binary emotion classification**. It talks to any OpenAI-compatible
server, forces the model to answer with strict JSON (`positive` / `negative`
plus a confidence between 0 and 1), validates the output defensively, and can
benchmark a built-in dataset with concurrency, JSON, and CSV reporting.

## Features

- Single classification from an argument or stdin
- Strict system prompt with JSON-only output
- Robust response parsing (JSON extraction, validation, recovery)
- Built-in 20-example benchmark dataset (10 positive, 10 negative)
- Reproducible 1,000-sample benchmark from `dair-ai/emotion` (seed 42)
- Concurrent benchmark mode (goroutines)
- Latency, throughput, per-category and confidence statistics
- CSV (`encoding/csv`), JSONL results and JSON summary outputs
- Environment-based configuration and timeouts
- Unit tests that never require a running model server

## Requirements

- Go 1.26 or newer
- A locally running OpenAI-compatible server exposing
  `POST /v1/chat/completions`

## Starting the local model server

This harness does not bundle or manage the model server; it only assumes an
OpenAI-compatible HTTP API. Start whatever server hosts `qwen3.5-4b-6bit`
(for example an MLX/vLLM/llama.cpp server configured for OpenAI-compatible
chat completions) so that it listens on:

```text
http://127.0.0.1:8000/v1
```

Verify it is reachable:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Reference environment

The benchmarks in this repository were run against a local model served with
**rapid-mlx** on:

```text
Model:  qwen3.5-4b-6bit
Server: rapid-mlx (OpenAI-compatible, http://127.0.0.1:8000/v1)
Host:   Mac Mini M4, 16 GB RAM
```

Future testing will be performed with other, smaller local models, so all
benchmark metadata (model, endpoint, dataset, seed, timestamp, concurrency) is
recorded in the `--output` summary to make cross-model comparison easy.

## Build

```bash
make build        # produces ./emotion-harness
# or
go build -o emotion-harness .
```

## Configuration

| Variable         | Default                          | Description                          |
| ---------------- | -------------------------------- | ------------------------------------ |
| `MODEL_BASE_URL` | `http://127.0.0.1:8000/v1`        | OpenAI-compatible base URL           |
| `MODEL_NAME`     | `qwen3.5-4b-6bit`                 | Model name sent in requests          |
| `MODEL_TIMEOUT`  | `60s`                             | HTTP timeout (Go duration syntax)    |
| `OPENAI_API_KEY` | *(unset)*                         | Optional bearer token; omitted if unset |

No API key is required by default. If `OPENAI_API_KEY` is set, it is sent as a
`Authorization: Bearer <key>` header.

## Single classification

```bash
./emotion-harness "I am incredibly happy today"
```

```text
Input: I am incredibly happy today

Classification: positive
Confidence: 98.4%
Latency: 142 ms
```

Using `make`:

```bash
make run TEXT="I am very happy today"
```

## Reading from stdin

When no argument is provided, input is read from stdin:

```bash
echo "I am disappointed with the result" | ./emotion-harness
```

```text
Input: I am disappointed with the result
Classification: negative
Confidence: 96.7%
Latency: 138 ms
```

## Benchmark

```bash
./emotion-harness --benchmark
# or
make benchmark
```

```text
Emotion Classification Benchmark
================================

Model: qwen3.5-4b-6bit
Endpoint: http://127.0.0.1:8000/v1

Samples: 20
Correct: 19
Incorrect: 1
Accuracy: 95.00%

Total time: 2.86 s
Average latency: 143 ms
Min latency: 119 ms
Max latency: 201 ms
Requests/sec: 6.99

Positive:
  Samples: 10
  Correct: 10
  Accuracy: 100.00%

Negative:
  Samples: 10
  Correct: 9
  Accuracy: 90.00%

Average confidence: 94.2%
Correct predictions average confidence: 96.1%
Incorrect predictions average confidence: 76.4%
```

## Concurrent benchmark

```bash
./emotion-harness --benchmark --concurrency 4
```

When `concurrency > 1`, requests are dispatched with goroutines. Accuracy is
computed per prediction and is independent of completion order.

```text
Concurrency: 4
Samples: 20
Accuracy: 95.00%

Total time: 1.42 s
Average latency: 241 ms
Requests/sec: 14.08
```

## JSON output

```bash
./emotion-harness --benchmark --json
```

```json
{
  "model": "qwen3.5-4b-6bit",
  "endpoint": "http://127.0.0.1:8000/v1",
  "samples": 20,
  "correct": 19,
  "incorrect": 1,
  "accuracy": 0.95,
  "average_confidence": 0.942,
  "average_latency_ms": 143.2,
  "min_latency_ms": 119,
  "max_latency_ms": 201,
  "requests_per_second": 6.8
}
```

Human-readable logging is never mixed into JSON mode (diagnostics go to
stderr).

## CSV output

```bash
./emotion-harness --benchmark --csv results.csv
```

```csv
id,input,expected,predicted,confidence,correct,latency_ms
1,"I had an amazing day.",positive,positive,0.98,true,142
2,"Everything went wrong.",negative,negative,0.97,true,137
```

CSV is produced with the Go standard library `encoding/csv` package.

## Verbose mode

```bash
./emotion-harness -v "I am extremely happy today."
# or --verbose
```

Verbose mode prints the request URL, model, temperature, and the raw model
response to stderr. `OPENAI_API_KEY` is never printed.

## Emotion Benchmark

`emotion-harness benchmark` runs a reproducible **1,000-sample** binary emotion
classification benchmark. The legacy `--benchmark` (20-sample) mode above is
unchanged; `benchmark` is an additive subcommand.

### Dataset

The dataset is derived from the public
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset
(config `split`). It is generated reproducibly by
`scripts/generate_emotion_benchmark.py` with `seed = 42` and written to:

```text
data/emotion-benchmark-1000.jsonl
```

Each JSONL line has the schema:

```json
{"id":1,"text":"I am extremely happy today","emotion":"joy","sentiment":"positive"}
```

| field       | type   | description                       |
| ----------- | ------ | --------------------------------- |
| `id`        | int    | unique id, 1..1000                |
| `text`      | string | the sentence to classify          |
| `emotion`   | string | original dataset emotion          |
| `sentiment` | string | binary label: `positive`/`negative` |

### Classes

`surprise` is intentionally excluded because the benchmark is binary.

```text
sadness → negative
anger   → negative
fear    → negative
joy     → positive
love    → positive
```

### Dataset size

```text
1,000 samples
200 per emotion (sadness, joy, love, anger, fear)
400 positive / 600 negative
```

> The public `dair-ai/emotion` **test** split only contains 159 `love`
> examples, so a fixed 200-per-emotion benchmark cannot be built from the test
> split alone. The generator uses the test split as its primary source and
> deterministically tops up short classes from the **validation** split
> (`love`: 159 from test + 41 from validation). Every sample still comes from
> the public dataset, and the exact source breakdown is printed when the
> dataset is generated.

Regenerate the dataset (requires network access; raw splits are cached under
`data/.cache/`):

```bash
make generate-dataset
# or
python3 scripts/generate_emotion_benchmark.py \
  --output data/emotion-benchmark-1000.jsonl
```

### Run

```bash
./emotion-harness benchmark
# or
make benchmark-1000
```

The dataset is validated before any inference. If it is missing, malformed, or
does not have exactly 1000 records / 200 per emotion / 400 positive / 600
negative with unique ids and correct emotion→sentiment mappings, the benchmark
fails before sending a single request.

### Custom model

```bash
./emotion-harness benchmark \
  --model qwen3.5-4b-6bit
```

### Custom endpoint

```bash
./emotion-harness benchmark \
  --endpoint http://127.0.0.1:8000/v1
```

### Custom dataset

```bash
./emotion-harness benchmark \
  --dataset data/emotion-benchmark-1000.jsonl
```

### Save detailed results

```bash
./emotion-harness benchmark \
  --results results.jsonl
```

Each line records one sample. Failures and invalid responses serialise
`predicted_sentiment` and `confidence` as `null` and include an `error`:

```json
{"id":1,"text":"I am extremely happy today","expected_emotion":"joy","expected_sentiment":"positive","predicted_sentiment":"positive","confidence":0.97,"correct":true,"latency_ms":41.2}
{"id":42,"text":"...","expected_emotion":"fear","expected_sentiment":"negative","predicted_sentiment":null,"confidence":null,"correct":false,"error":"invalid JSON response","latency_ms":53.4}
```

### Save summary

```bash
./emotion-harness benchmark \
  --output results.json
```

The summary is machine-readable and includes `model`, `endpoint`, `dataset`,
`seed`, `timestamp`, `concurrency`, counts, accuracy, precision/recall/F1, the
confusion matrix, average confidence, latency percentiles, throughput, request
status counts, and per-sentiment/per-emotion breakdowns. This makes runs easy
to diff across models:

```bash
./emotion-harness benchmark --model qwen3.5-4b-6bit --output qwen35-4b-results.json
./emotion-harness benchmark --model another-local-model --output another-model-results.json
```

### Concurrency

```bash
./emotion-harness benchmark \
  --concurrency 4
```

Default is `concurrency = 1`, which gives a clean sequential baseline. With
`--concurrency > 1`, requests are dispatched with goroutines; per-sample
results are stored by dataset index so statistics are unaffected by completion
order.

### Metrics

- **Accuracy** = `correct / total_samples`. Failed and invalid responses count
  as incorrect and are never removed from the denominator.
- **Confusion matrix** is computed over valid predictions only (failed/invalid
  responses cannot be assigned to a cell). `TP + TN` therefore equals
  `correct`; `TP + TN + FP + FN` equals the number of successful requests.
- **Precision** = `TP / (TP + FP)`, **Recall** = `TP / (TP + FN)`,
  **F1** = `2·P·R / (P + R)`, with `positive` as the positive class.
- **Average confidence** is over valid predictions only.
- **Latency** min/average/median/P95/max are measured per request in
  milliseconds; percentiles use linear interpolation between closest ranks.
- **Throughput** = `total_samples / total_duration_seconds`. No artificial
  sleeps are inserted between requests.
- **Per-emotion accuracy** uses the original `emotion` field from the dataset.

## Tests

```bash
make test
# or
go test ./...
```

Tests cover parsing, validation, confidence handling, malformed/embedded JSON,
accuracy and benchmark statistics, and client error handling (HTTP errors,
timeouts, connection refused, missing content). No model server is required.

The 1,000-sample benchmark adds tests for JSONL loading, dataset validation,
strict sentiment-response parsing, confusion-matrix/precision/recall/F1
calculation, latency percentiles, and the concurrent runner:

```bash
make benchmark-test
# or
go test -run 'Emotion|Dataset|Sentiment' ./...
```

## Accuracy vs. confidence

Accuracy and confidence are deliberately separate:

- **Accuracy** is computed only by comparing the predicted label with the
  expected label (`predicted == expected`). It never uses confidence.
- **Confidence** is whatever the model reported for its own answer.

So `expected=negative, predicted=negative, confidence=72%` is still a
**correct** prediction, while `expected=negative, predicted=positive,
confidence=99%` is **incorrect** despite the high confidence.

## Error handling

The harness handles model server unavailability, connection refused, HTTP
errors, timeouts, malformed JSON, missing responses/content, invalid
classifications, out-of-range confidence, and API errors reported by the
server. When the model output cannot be parsed it reports:

```text
ERROR: invalid model response
```

and, in verbose mode, includes the raw response.

## Make targets

```text
build             build ./emotion-harness
run               go run . "$(TEXT)"
test              go test ./...
fmt               gofmt -w .
vet               go vet ./...
clean             remove the binary
benchmark         go run . --benchmark            (legacy 20-sample)
benchmark-1000    build + ./emotion-harness benchmark  (1,000-sample)
benchmark-test    go test -run 'Emotion|Dataset|Sentiment' ./...
generate-dataset  regenerate data/emotion-benchmark-1000.jsonl
check             fmt vet test
```

## Project layout

```text
main.go              CLI, environment configuration, orchestration
client.go            OpenAI-compatible HTTP client (net/http)
classifier.go        single-text prompt, response parsing and validation
benchmark.go         legacy 20-sample dataset, runner, statistics
output.go            legacy human, JSON and CSV reporting
dataset.go           1,000-sample JSONL types, loader and validation
sentiment.go         benchmark system prompt and strict sentiment parser
emotion_benchmark.go 1,000-sample concurrent runner, metrics, latency
emotion_output.go    benchmark human report, JSONL results, JSON summary
scripts/             dataset generator (Python, stdlib only)
data/                generated benchmark dataset
*_test.go            unit tests
```

## Acknowledgements

This project was made possible thanks to **opencode go + deepseek v4.1 flash**.

