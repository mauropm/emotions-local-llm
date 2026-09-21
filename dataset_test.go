package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// buildValidSamples constructs an in-memory dataset matching the required
// 1,000-sample / 200-per-emotion / 400-positive / 600-negative shape.
func buildValidSamples() []BenchSample {
	samples := make([]BenchSample, 0, DatasetSize)
	id := 1
	for _, emotion := range EmotionOrder {
		for i := 0; i < EmotionSampleCount; i++ {
			samples = append(samples, BenchSample{
				ID:        id,
				Text:      fmt.Sprintf("%s example %d", emotion, i),
				Emotion:   emotion,
				Sentiment: EmotionSentiment[emotion],
			})
			id++
		}
	}
	return samples
}

func TestLoadDataset(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.jsonl")
	content := `{"id":1,"text":"I am happy","emotion":"joy","sentiment":"positive"}

{"id":2,"text":"I am sad","emotion":"sadness","sentiment":"negative"}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	samples, err := LoadDataset(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("got %d samples, want 2", len(samples))
	}
	if samples[0].Emotion != EmotionJoy || samples[0].Sentiment != LabelPositive {
		t.Errorf("unexpected first sample: %+v", samples[0])
	}
	if samples[1].Emotion != EmotionSadness || samples[1].Sentiment != LabelNegative {
		t.Errorf("unexpected second sample: %+v", samples[1])
	}
}

func TestLoadDatasetMalformedLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.jsonl")
	content := `{"id":1,"text":"ok","emotion":"joy","sentiment":"positive"}
{"id":2,"text":`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDataset(path); err == nil {
		t.Fatal("expected error for malformed JSON line")
	}
}

func TestLoadDatasetMissingFile(t *testing.T) {
	if _, err := LoadDataset(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestValidateDatasetValid(t *testing.T) {
	if err := ValidateDataset(buildValidSamples()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateDatasetWrongCount(t *testing.T) {
	samples := buildValidSamples()[:999]
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for wrong record count")
	}
}

func TestValidateDatasetWrongEmotionCount(t *testing.T) {
	samples := buildValidSamples()
	// Convert one love sample into an extra joy sample.
	for i := range samples {
		if samples[i].Emotion == EmotionLove {
			samples[i].Emotion = EmotionJoy
			break
		}
	}
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for wrong per-emotion counts")
	}
}

func TestValidateDatasetDuplicateID(t *testing.T) {
	samples := buildValidSamples()
	samples[10].ID = samples[0].ID
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for duplicate id")
	}
}

func TestValidateDatasetInvalidEmotion(t *testing.T) {
	samples := buildValidSamples()
	samples[0].Emotion = "surprise"
	samples[0].Sentiment = LabelPositive
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for invalid emotion")
	}
}

func TestValidateDatasetInvalidSentiment(t *testing.T) {
	samples := buildValidSamples()
	samples[0].Sentiment = "neutral"
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for invalid sentiment")
	}
}

func TestValidateDatasetMappingMismatch(t *testing.T) {
	samples := buildValidSamples()
	// sadness must map to negative; force positive.
	samples[0].Sentiment = LabelPositive
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for emotion/sentiment mismatch")
	}
}

func TestValidateDatasetEmptyText(t *testing.T) {
	samples := buildValidSamples()
	samples[0].Text = "   "
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for empty text")
	}
}

func TestValidateDatasetNonPositiveID(t *testing.T) {
	samples := buildValidSamples()
	samples[0].ID = 0
	if err := ValidateDataset(samples); err == nil {
		t.Fatal("expected error for non-positive id")
	}
}

// TestRealDatasetFile loads and validates the generated benchmark file. It
// skips if the dataset has not been generated.
func TestRealDatasetFile(t *testing.T) {
	samples, err := LoadDataset(DefaultDatasetPath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("dataset %s not present; run scripts/generate_emotion_benchmark.py", DefaultDatasetPath)
		}
		t.Fatalf("loading dataset: %v", err)
	}
	if err := ValidateDataset(samples); err != nil {
		t.Fatalf("validating dataset: %v", err)
	}

	emotions := map[string]int{}
	sentiments := map[string]int{}
	ids := map[int]struct{}{}
	for _, sample := range samples {
		emotions[sample.Emotion]++
		sentiments[sample.Sentiment]++
		if _, exists := ids[sample.ID]; exists {
			t.Fatalf("duplicate id %d", sample.ID)
		}
		ids[sample.ID] = struct{}{}
	}
	if len(samples) != 1000 {
		t.Errorf("got %d records, want 1000", len(samples))
	}
	for _, emotion := range EmotionOrder {
		if emotions[emotion] != 200 {
			t.Errorf("emotion %s = %d, want 200", emotion, emotions[emotion])
		}
	}
	if sentiments[LabelPositive] != 400 || sentiments[LabelNegative] != 600 {
		t.Errorf("sentiment totals = %v, want positive 400 / negative 600", sentiments)
	}
}

func TestBenchSampleJSONRoundTrip(t *testing.T) {
	line := `{"id":7,"text":"a b","emotion":"fear","sentiment":"negative"}`
	var sample BenchSample
	if err := json.Unmarshal([]byte(line), &sample); err != nil {
		t.Fatal(err)
	}
	if sample.ID != 7 || sample.Text != "a b" || sample.Emotion != EmotionFear || sample.Sentiment != LabelNegative {
		t.Fatalf("unexpected sample: %+v", sample)
	}
}
