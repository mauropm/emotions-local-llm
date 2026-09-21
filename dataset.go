package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Benchmark dataset constants. The dataset is generated reproducibly by
// scripts/generate_emotion_benchmark.py from the public dair-ai/emotion
// dataset with DatasetSeed.
const (
	// DefaultDatasetPath is where the generated 1,000-sample benchmark lives.
	DefaultDatasetPath = "data/emotion-benchmark-1000.jsonl"

	// DatasetSeed is the random seed used to build the dataset.
	DatasetSeed = 42

	// DatasetSize is the exact number of samples in the benchmark.
	DatasetSize = 1000

	// EmotionSampleCount is the exact number of samples per emotion.
	EmotionSampleCount = 200

	// PositiveSampleCount and NegativeSampleCount are the expected binary
	// sentiment totals.
	PositiveSampleCount = 400
	NegativeSampleCount = 600
)

// Canonical emotion names from dair-ai/emotion. "surprise" is intentionally
// excluded because this benchmark is binary positive/negative classification.
const (
	EmotionSadness = "sadness"
	EmotionJoy     = "joy"
	EmotionLove    = "love"
	EmotionAnger   = "anger"
	EmotionFear    = "fear"
)

// EmotionOrder is the canonical order used when reporting per-emotion stats.
var EmotionOrder = []string{
	EmotionSadness,
	EmotionJoy,
	EmotionLove,
	EmotionAnger,
	EmotionFear,
}

// EmotionSentiment maps each benchmark emotion to its expected binary
// sentiment.
var EmotionSentiment = map[string]string{
	EmotionSadness: LabelNegative,
	EmotionAnger:   LabelNegative,
	EmotionFear:    LabelNegative,
	EmotionJoy:     LabelPositive,
	EmotionLove:    LabelPositive,
}

// BenchSample is one JSONL record of the benchmark dataset.
type BenchSample struct {
	ID        int    `json:"id"`
	Text      string `json:"text"`
	Emotion   string `json:"emotion"`
	Sentiment string `json:"sentiment"`
}

// LoadDataset reads a JSONL benchmark file. Blank lines are ignored; malformed
// lines are reported with their line number.
func LoadDataset(path string) ([]BenchSample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening dataset %s: %w", path, err)
	}
	defer file.Close()

	var samples []BenchSample
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var sample BenchSample
		if err := json.Unmarshal([]byte(line), &sample); err != nil {
			return nil, fmt.Errorf("%s:%d: invalid JSON: %w", path, lineNumber, err)
		}
		samples = append(samples, sample)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading dataset %s: %w", path, err)
	}
	return samples, nil
}

// ValidateDataset enforces the benchmark's expected shape before any model
// requests are made. It fails on the first problem it finds.
func ValidateDataset(samples []BenchSample) error {
	if len(samples) != DatasetSize {
		return fmt.Errorf("expected exactly %d records, got %d", DatasetSize, len(samples))
	}

	seenIDs := make(map[int]struct{}, len(samples))
	emotionCounts := make(map[string]int, len(EmotionOrder))
	sentimentCounts := map[string]int{LabelPositive: 0, LabelNegative: 0}

	for i, sample := range samples {
		if sample.ID <= 0 {
			return fmt.Errorf("record %d: id must be positive, got %d", i+1, sample.ID)
		}
		if _, exists := seenIDs[sample.ID]; exists {
			return fmt.Errorf("record %d: duplicate id %d", i+1, sample.ID)
		}
		seenIDs[sample.ID] = struct{}{}

		if strings.TrimSpace(sample.Text) == "" {
			return fmt.Errorf("record %d (id %d): text is empty", i+1, sample.ID)
		}

		expectedSentiment, ok := EmotionSentiment[sample.Emotion]
		if !ok {
			return fmt.Errorf("record %d (id %d): invalid emotion %q", i+1, sample.ID, sample.Emotion)
		}
		if sample.Sentiment != LabelPositive && sample.Sentiment != LabelNegative {
			return fmt.Errorf("record %d (id %d): invalid sentiment %q", i+1, sample.ID, sample.Sentiment)
		}
		if sample.Sentiment != expectedSentiment {
			return fmt.Errorf(
				"record %d (id %d): emotion %q must map to %q, got %q",
				i+1, sample.ID, sample.Emotion, expectedSentiment, sample.Sentiment,
			)
		}

		emotionCounts[sample.Emotion]++
		sentimentCounts[sample.Sentiment]++
	}

	for _, emotion := range EmotionOrder {
		if got := emotionCounts[emotion]; got != EmotionSampleCount {
			return fmt.Errorf("emotion %q: expected %d records, got %d", emotion, EmotionSampleCount, got)
		}
	}
	if got := sentimentCounts[LabelPositive]; got != PositiveSampleCount {
		return fmt.Errorf("expected %d positive records, got %d", PositiveSampleCount, got)
	}
	if got := sentimentCounts[LabelNegative]; got != NegativeSampleCount {
		return fmt.Errorf("expected %d negative records, got %d", NegativeSampleCount, got)
	}

	return nil
}
