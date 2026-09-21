package main

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestComputeEmotionStatsAllValid(t *testing.T) {
	results := []BenchResult{
		{ID: 1, ExpectedEmotion: EmotionJoy, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelPositive, Confidence: 0.9, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful, Latency: 10 * time.Millisecond},
		{ID: 2, ExpectedEmotion: EmotionJoy, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelPositive, Confidence: 0.8, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful, Latency: 20 * time.Millisecond},
		{ID: 3, ExpectedEmotion: EmotionSadness, ExpectedSentiment: LabelNegative, PredictedSentiment: LabelNegative, Confidence: 0.7, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful, Latency: 30 * time.Millisecond},
		{ID: 4, ExpectedEmotion: EmotionAnger, ExpectedSentiment: LabelNegative, PredictedSentiment: LabelPositive, Confidence: 0.6, HasConfidence: true, Correct: false, Status: BenchStatusSuccessful, Latency: 40 * time.Millisecond},
	}

	stats := ComputeEmotionBenchmarkStats(BenchMetadata{Concurrency: 1}, results, time.Second)

	if stats.Total != 4 || stats.Correct != 3 || stats.Incorrect != 1 {
		t.Fatalf("unexpected counts: %+v", stats)
	}
	if stats.Successful != 4 || stats.Failed != 0 || stats.Invalid != 0 {
		t.Fatalf("unexpected status counts: %+v", stats)
	}
	if stats.TruePositive != 2 || stats.TrueNegative != 1 || stats.FalsePositive != 1 || stats.FalseNegative != 0 {
		t.Fatalf("unexpected confusion matrix: TP=%d TN=%d FP=%d FN=%d", stats.TruePositive, stats.TrueNegative, stats.FalsePositive, stats.FalseNegative)
	}
	assertClose(t, "accuracy", stats.Accuracy, 0.75)
	assertClose(t, "precision", stats.Precision, 2.0/3.0)
	assertClose(t, "recall", stats.Recall, 1.0)
	assertClose(t, "f1", stats.F1, 0.8)
	assertClose(t, "average confidence", stats.AverageConfidence, 0.75)

	if stats.Positive.Total != 2 || stats.Positive.Correct != 2 || stats.Positive.Accuracy != 1 {
		t.Errorf("positive stats = %+v", stats.Positive)
	}
	if stats.Negative.Total != 2 || stats.Negative.Correct != 1 || stats.Negative.Accuracy != 0.5 {
		t.Errorf("negative stats = %+v", stats.Negative)
	}
	if stats.ByEmotion[EmotionJoy].Correct != 2 || stats.ByEmotion[EmotionJoy].Total != 2 {
		t.Errorf("joy stats = %+v", stats.ByEmotion[EmotionJoy])
	}
	if stats.ByEmotion[EmotionAnger].Correct != 0 || stats.ByEmotion[EmotionAnger].Total != 1 {
		t.Errorf("anger stats = %+v", stats.ByEmotion[EmotionAnger])
	}

	assertClose(t, "latency min", stats.Latency.MinMS, 10)
	assertClose(t, "latency average", stats.Latency.AverageMS, 25)
	assertClose(t, "latency median", stats.Latency.MedianMS, 25)
	assertClose(t, "latency p95", stats.Latency.P95MS, 38.5)
	assertClose(t, "latency max", stats.Latency.MaxMS, 40)
	assertClose(t, "throughput", stats.ThroughputSamplesPerSecond, 4)
}

func TestComputeEmotionStatsFailuresAndInvalid(t *testing.T) {
	results := []BenchResult{
		{ID: 1, ExpectedEmotion: EmotionJoy, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelPositive, Confidence: 0.9, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful},
		{ID: 2, ExpectedEmotion: EmotionLove, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelNegative, Confidence: 0.6, HasConfidence: true, Correct: false, Status: BenchStatusSuccessful},
		{ID: 3, ExpectedEmotion: EmotionSadness, ExpectedSentiment: LabelNegative, PredictedSentiment: LabelNegative, Confidence: 0.7, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful},
		{ID: 4, ExpectedEmotion: EmotionAnger, ExpectedSentiment: LabelNegative, PredictedSentiment: LabelPositive, Confidence: 0.4, HasConfidence: true, Correct: false, Status: BenchStatusSuccessful},
		{ID: 5, ExpectedEmotion: EmotionFear, ExpectedSentiment: LabelNegative, Correct: false, Status: BenchStatusInvalid, Error: "invalid JSON response"},
		{ID: 6, ExpectedEmotion: EmotionFear, ExpectedSentiment: LabelNegative, Correct: false, Status: BenchStatusFailed, Error: "connection refused"},
	}

	stats := ComputeEmotionBenchmarkStats(BenchMetadata{}, results, 2*time.Second)

	if stats.Total != 6 || stats.Correct != 2 || stats.Incorrect != 4 {
		t.Fatalf("unexpected counts: %+v", stats)
	}
	if stats.Successful != 4 || stats.Invalid != 1 || stats.Failed != 1 {
		t.Fatalf("unexpected status counts: %+v", stats)
	}
	// Failures/invalid are excluded from the confusion matrix but included in
	// the accuracy denominator.
	if stats.TruePositive != 1 || stats.TrueNegative != 1 || stats.FalsePositive != 1 || stats.FalseNegative != 1 {
		t.Fatalf("unexpected confusion matrix: TP=%d TN=%d FP=%d FN=%d", stats.TruePositive, stats.TrueNegative, stats.FalsePositive, stats.FalseNegative)
	}
	assertClose(t, "accuracy", stats.Accuracy, 2.0/6.0)
	assertClose(t, "precision", stats.Precision, 0.5)
	assertClose(t, "recall", stats.Recall, 0.5)
	assertClose(t, "f1", stats.F1, 0.5)
	// Average confidence only over valid predictions.
	assertClose(t, "average confidence", stats.AverageConfidence, (0.9+0.6+0.7+0.4)/4)

	if stats.Positive.Total != 2 || stats.Positive.Correct != 1 {
		t.Errorf("positive stats = %+v", stats.Positive)
	}
	if stats.Negative.Total != 4 || stats.Negative.Correct != 1 {
		t.Errorf("negative stats = %+v", stats.Negative)
	}
}

func TestComputeEmotionStatsEmpty(t *testing.T) {
	stats := ComputeEmotionBenchmarkStats(BenchMetadata{}, nil, time.Second)
	if stats.Total != 0 || stats.Accuracy != 0 || stats.F1 != 0 {
		t.Fatalf("unexpected stats for empty run: %+v", stats)
	}
}

func TestPercentile(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5}
	assertClose(t, "p50", percentile(values, 50), 3)
	assertClose(t, "p95", percentile(values, 95), 4.8)
	assertClose(t, "p0", percentile(values, 0), 1)
	assertClose(t, "p100", percentile(values, 100), 5)
}

func TestComputeLatencyStatsEmpty(t *testing.T) {
	stats := computeLatencyStats(nil)
	if stats != (LatencyStats{}) {
		t.Fatalf("expected zero stats, got %+v", stats)
	}
}

func TestRunEmotionBenchmark(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		user := ""
		if len(request.Messages) > 0 {
			user = request.Messages[len(request.Messages)-1].Content
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(user, "FAIL"):
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, `{"error":{"message":"boom"}}`)
		case strings.Contains(user, "INVALID"):
			io.WriteString(w, chatCompletionBody("not json"))
		case strings.Contains(user, "WRONG"):
			if strings.HasPrefix(user, "POS") {
				io.WriteString(w, chatCompletionBody(`{"sentiment":"negative","confidence":0.4}`))
			} else {
				io.WriteString(w, chatCompletionBody(`{"sentiment":"positive","confidence":0.4}`))
			}
		case strings.HasPrefix(user, "POS"):
			io.WriteString(w, chatCompletionBody(`{"sentiment":"positive","confidence":0.9}`))
		default:
			io.WriteString(w, chatCompletionBody(`{"sentiment":"negative","confidence":0.8}`))
		}
	}))
	defer server.Close()

	samples := []BenchSample{
		{ID: 1, Text: "POS_OK", Emotion: EmotionJoy, Sentiment: LabelPositive},
		{ID: 2, Text: "POS_WRONG", Emotion: EmotionJoy, Sentiment: LabelPositive},
		{ID: 3, Text: "POS_INVALID", Emotion: EmotionLove, Sentiment: LabelPositive},
		{ID: 4, Text: "NEG_OK", Emotion: EmotionSadness, Sentiment: LabelNegative},
		{ID: 5, Text: "NEG_WRONG", Emotion: EmotionAnger, Sentiment: LabelNegative},
		{ID: 6, Text: "NEG_FAIL", Emotion: EmotionFear, Sentiment: LabelNegative},
	}

	client := NewClient(Config{BaseURL: server.URL, Model: "test-model", Timeout: 5 * time.Second})

	var mu sync.Mutex
	progressCalls := []int{}
	onProgress := func(done, total int) {
		mu.Lock()
		progressCalls = append(progressCalls, done)
		mu.Unlock()
	}

	results, duration := RunEmotionBenchmark(context.Background(), client, samples, 4, onProgress)
	if duration <= 0 {
		t.Errorf("duration = %v, want positive", duration)
	}
	if len(results) != len(samples) {
		t.Fatalf("got %d results, want %d", len(results), len(samples))
	}

	// Results must remain in dataset order regardless of completion order.
	for i, result := range results {
		if result.ID != samples[i].ID {
			t.Errorf("results[%d].ID = %d, want %d", i, result.ID, samples[i].ID)
		}
	}

	mu.Lock()
	callCount := len(progressCalls)
	maxDone := 0
	for _, value := range progressCalls {
		if value > maxDone {
			maxDone = value
		}
	}
	mu.Unlock()
	if callCount != len(samples) {
		t.Errorf("progress called %d times, want %d", callCount, len(samples))
	}
	if maxDone != len(samples) {
		t.Errorf("max progress = %d, want %d", maxDone, len(samples))
	}

	stats := ComputeEmotionBenchmarkStats(BenchMetadata{Concurrency: 4}, results, duration)
	if stats.Correct != 2 || stats.Incorrect != 4 {
		t.Fatalf("correct/incorrect = %d/%d, want 2/4", stats.Correct, stats.Incorrect)
	}
	if stats.Successful != 4 || stats.Invalid != 1 || stats.Failed != 1 {
		t.Fatalf("status counts = successful %d, invalid %d, failed %d", stats.Successful, stats.Invalid, stats.Failed)
	}
	if results[2].PredictedSentiment != "" || results[2].Status != BenchStatusInvalid {
		t.Errorf("invalid sample not recorded correctly: %+v", results[2])
	}
	if results[5].Status != BenchStatusFailed || results[5].Error == "" {
		t.Errorf("failed sample not recorded correctly: %+v", results[5])
	}
}

func TestWriteBenchResultsJSONL(t *testing.T) {
	results := []BenchResult{
		{ID: 1, Text: "happy", ExpectedEmotion: EmotionJoy, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelPositive, Confidence: 0.97, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful, Latency: 41200 * time.Microsecond},
		{ID: 42, Text: "scared", ExpectedEmotion: EmotionFear, ExpectedSentiment: LabelNegative, Correct: false, Status: BenchStatusInvalid, Error: "invalid JSON response", Latency: 53400 * time.Microsecond},
	}

	var builder strings.Builder
	if err := WriteBenchResultsJSONL(&builder, results); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(builder.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2", len(lines))
	}

	var first map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("line 1: %v", err)
	}
	if first["predicted_sentiment"] != "positive" || first["correct"] != true {
		t.Errorf("unexpected first record: %v", first)
	}
	assertClose(t, "latency_ms", first["latency_ms"].(float64), 41.2)

	var second map[string]any
	if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
		t.Fatalf("line 2: %v", err)
	}
	if second["predicted_sentiment"] != nil {
		t.Errorf("predicted_sentiment = %v, want null", second["predicted_sentiment"])
	}
	if second["confidence"] != nil {
		t.Errorf("confidence = %v, want null", second["confidence"])
	}
	if second["error"] != "invalid JSON response" {
		t.Errorf("error = %v", second["error"])
	}
}

func TestWriteBenchSummaryJSON(t *testing.T) {
	results := []BenchResult{
		{ID: 1, ExpectedEmotion: EmotionJoy, ExpectedSentiment: LabelPositive, PredictedSentiment: LabelPositive, Confidence: 0.9, HasConfidence: true, Correct: true, Status: BenchStatusSuccessful, Latency: 10 * time.Millisecond},
		{ID: 2, ExpectedEmotion: EmotionSadness, ExpectedSentiment: LabelNegative, PredictedSentiment: LabelPositive, Confidence: 0.8, HasConfidence: true, Correct: false, Status: BenchStatusSuccessful, Latency: 20 * time.Millisecond},
	}
	stats := ComputeEmotionBenchmarkStats(BenchMetadata{
		Model: "m", Endpoint: "e", Dataset: "d", Seed: DatasetSeed, Concurrency: 2,
	}, results, time.Second)

	var builder strings.Builder
	if err := WriteBenchSummaryJSON(&builder, stats); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal([]byte(builder.String()), &decoded); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if decoded["model"] != "m" || decoded["endpoint"] != "e" || decoded["dataset"] != "d" {
		t.Errorf("unexpected metadata: %v", decoded)
	}
	if decoded["seed"].(float64) != DatasetSeed || decoded["concurrency"].(float64) != 2 {
		t.Errorf("unexpected seed/concurrency: %v", decoded)
	}
	if decoded["total"].(float64) != 2 || decoded["correct"].(float64) != 1 || decoded["incorrect"].(float64) != 1 {
		t.Errorf("unexpected counts: %v", decoded)
	}
	if decoded["true_positive"].(float64) != 1 || decoded["false_positive"].(float64) != 1 {
		t.Errorf("unexpected confusion matrix: %v", decoded)
	}
	if decoded["successful_requests"].(float64) != 2 || decoded["failed_requests"].(float64) != 0 || decoded["invalid_responses"].(float64) != 0 {
		t.Errorf("unexpected request counts: %v", decoded)
	}
	latency, ok := decoded["latency_ms"].(map[string]any)
	if !ok {
		t.Fatalf("latency_ms missing or wrong type: %v", decoded["latency_ms"])
	}
	for _, key := range []string{"min", "average", "median", "p95", "max"} {
		if _, present := latency[key]; !present {
			t.Errorf("latency_ms missing key %q", key)
		}
	}
	if _, present := decoded["timestamp"]; !present {
		t.Error("timestamp missing")
	}
	if _, present := decoded["emotions"]; !present {
		t.Error("emotions missing")
	}
}

func chatCompletionBody(content string) string {
	payload := map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"role": "assistant", "content": content}},
		},
	}
	encoded, _ := json.Marshal(payload)
	return string(encoded)
}

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}
