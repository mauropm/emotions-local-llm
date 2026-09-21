BINARY := emotion-harness
TEXT ?= I am very happy today

.PHONY: build run test fmt vet clean benchmark benchmark-1000 benchmark-test generate-dataset check

build:
	go build -o $(BINARY) .

run:
	go run . "$(TEXT)"

test:
	go test ./...

fmt:
	gofmt -w .

vet:
	go vet ./...

clean:
	rm -f $(BINARY)

# Legacy 20-sample built-in benchmark (unchanged).
benchmark:
	go run . --benchmark

# Reproducible 1,000-sample dataset benchmark.
benchmark-1000: build
	./$(BINARY) benchmark

# Unit tests for the benchmark dataset, parser, metrics and runner.
benchmark-test:
	go test -run 'Emotion|Dataset|Sentiment' ./...

# Regenerate data/emotion-benchmark-1000.jsonl from dair-ai/emotion (seed 42).
generate-dataset:
	python3 scripts/generate_emotion_benchmark.py --output data/emotion-benchmark-1000.jsonl

check: fmt vet test
