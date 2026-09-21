package main

import (
	"context"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Per-request outcome statuses.
const (
	BenchStatusSuccessful = "successful"
	BenchStatusFailed     = "failed"
	BenchStatusInvalid    = "invalid_response"
)

// BenchResult captures the outcome of classifying one benchmark sample.
type BenchResult struct {
	ID                int
	Text              string
	ExpectedEmotion   string
	ExpectedSentiment string

	// PredictedSentiment is empty when the request failed or the response was
	// invalid.
	PredictedSentiment string
	Confidence         float64
	HasConfidence      bool

	Correct bool
	Status  string
	Error   string

	Latency time.Duration
}

// LatencyStats summarises request latency in milliseconds.
type LatencyStats struct {
	MinMS     float64
	AverageMS float64
	MedianMS  float64
	P95MS     float64
	MaxMS     float64
}

// EmotionStats aggregates results for one expected category (a sentiment or an
// emotion).
type EmotionStats struct {
	Total     int
	Correct   int
	Incorrect int
	Accuracy  float64
}

// BenchMetadata carries run-level information recorded in the summary.
type BenchMetadata struct {
	Model       string
	Endpoint    string
	Dataset     string
	Seed        int
	Concurrency int
	StartedAt   time.Time
}

// EmotionBenchmarkStats summarises a full 1,000-sample benchmark run.
type EmotionBenchmarkStats struct {
	BenchMetadata
	Timestamp  time.Time
	Total      int
	Correct    int
	Incorrect  int
	Accuracy   float64
	Successful int
	Failed     int
	Invalid    int

	// Confusion matrix computed over valid predictions only. Failed and
	// invalid responses are excluded here but still count as incorrect in
	// Accuracy, which uses the full sample count as its denominator.
	TruePositive  int
	TrueNegative  int
	FalsePositive int
	FalseNegative int

	Precision float64
	Recall    float64
	F1        float64

	AverageConfidence float64

	Latency                    LatencyStats
	ThroughputSamplesPerSecond float64
	TotalDuration              time.Duration

	Positive  EmotionStats
	Negative  EmotionStats
	ByEmotion map[string]EmotionStats

	Results []BenchResult
}

// RunEmotionBenchmark classifies every sample, optionally concurrently, and
// returns per-sample results in dataset order plus the wall-clock duration.
// onProgress (optional) is invoked after each completed request.
func RunEmotionBenchmark(
	ctx context.Context,
	client *Client,
	samples []BenchSample,
	concurrency int,
	onProgress func(done, total int),
) ([]BenchResult, time.Duration) {
	if concurrency < 1 {
		concurrency = 1
	}

	results := make([]BenchResult, len(samples))
	var completed atomic.Int64
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	start := time.Now()
	for i, sample := range samples {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, s BenchSample) {
			defer wg.Done()
			defer func() { <-sem }()

			t0 := time.Now()
			result := classifyBenchSample(ctx, client, s)
			result.Latency = time.Since(t0)
			results[idx] = result

			if onProgress != nil {
				onProgress(int(completed.Add(1)), len(samples))
			} else {
				completed.Add(1)
			}
		}(i, sample)
	}
	wg.Wait()

	return results, time.Since(start)
}

// classifyBenchSample performs a single benchmark request. A failure or an
// unparseable response is recorded rather than returned as an error so that a
// single bad request never aborts the benchmark.
func classifyBenchSample(ctx context.Context, client *Client, sample BenchSample) BenchResult {
	result := BenchResult{
		ID:                sample.ID,
		Text:              sample.Text,
		ExpectedEmotion:   sample.Emotion,
		ExpectedSentiment: sample.Sentiment,
	}

	content, err := client.ChatCompletion(ctx, BenchSystemPrompt, sample.Text)
	if err != nil {
		result.Status = BenchStatusFailed
		result.Error = err.Error()
		return result
	}

	parsed, err := ParseSentimentResponse(content)
	if err != nil {
		result.Status = BenchStatusInvalid
		result.Error = "invalid JSON response"
		return result
	}

	result.Status = BenchStatusSuccessful
	result.PredictedSentiment = parsed.Sentiment
	result.Confidence = parsed.Confidence
	result.HasConfidence = true
	result.Correct = parsed.Sentiment == sample.Sentiment
	return result
}

// ComputeEmotionBenchmarkStats aggregates per-sample results into summary
// statistics. It is pure and unit-testable without a model server.
func ComputeEmotionBenchmarkStats(meta BenchMetadata, results []BenchResult, total time.Duration) EmotionBenchmarkStats {
	stats := EmotionBenchmarkStats{
		BenchMetadata: meta,
		Timestamp:     time.Now(),
		Total:         len(results),
		TotalDuration: total,
		ByEmotion:     make(map[string]EmotionStats, len(EmotionOrder)),
		Results:       results,
	}
	if len(results) == 0 {
		return stats
	}

	latencies := make([]float64, 0, len(results))
	var confidenceSum float64
	var confidenceCount int

	for _, r := range results {
		latencies = append(latencies, float64(r.Latency.Microseconds())/1000.0)

		switch r.Status {
		case BenchStatusSuccessful:
			stats.Successful++
		case BenchStatusInvalid:
			stats.Invalid++
		default:
			stats.Failed++
		}

		if r.Correct {
			stats.Correct++
		} else {
			stats.Incorrect++
		}

		if r.HasConfidence {
			confidenceSum += r.Confidence
			confidenceCount++
		}

		switch r.ExpectedSentiment {
		case LabelPositive:
			stats.Positive.Total++
			if r.Correct {
				stats.Positive.Correct++
			}
		case LabelNegative:
			stats.Negative.Total++
			if r.Correct {
				stats.Negative.Correct++
			}
		}

		emotion := stats.ByEmotion[r.ExpectedEmotion]
		emotion.Total++
		if r.Correct {
			emotion.Correct++
		}
		stats.ByEmotion[r.ExpectedEmotion] = emotion

		// Confusion matrix over valid predictions only.
		if r.Status == BenchStatusSuccessful {
			switch {
			case r.ExpectedSentiment == LabelPositive && r.PredictedSentiment == LabelPositive:
				stats.TruePositive++
			case r.ExpectedSentiment == LabelPositive && r.PredictedSentiment == LabelNegative:
				stats.FalseNegative++
			case r.ExpectedSentiment == LabelNegative && r.PredictedSentiment == LabelPositive:
				stats.FalsePositive++
			case r.ExpectedSentiment == LabelNegative && r.PredictedSentiment == LabelNegative:
				stats.TrueNegative++
			}
		}
	}

	stats.Accuracy = safeDivide(float64(stats.Correct), float64(stats.Total))
	stats.Precision = safeDivide(float64(stats.TruePositive), float64(stats.TruePositive+stats.FalsePositive))
	stats.Recall = safeDivide(float64(stats.TruePositive), float64(stats.TruePositive+stats.FalseNegative))
	if stats.Precision+stats.Recall > 0 {
		stats.F1 = 2 * stats.Precision * stats.Recall / (stats.Precision + stats.Recall)
	}
	stats.AverageConfidence = safeDivide(confidenceSum, float64(confidenceCount))
	stats.Latency = computeLatencyStats(latencies)
	if seconds := total.Seconds(); seconds > 0 {
		stats.ThroughputSamplesPerSecond = float64(stats.Total) / seconds
	}

	finalizeEmotionStats(&stats.Positive)
	finalizeEmotionStats(&stats.Negative)
	for emotion, value := range stats.ByEmotion {
		finalizeEmotionStats(&value)
		stats.ByEmotion[emotion] = value
	}

	return stats
}

func finalizeEmotionStats(stats *EmotionStats) {
	stats.Incorrect = stats.Total - stats.Correct
	stats.Accuracy = safeDivide(float64(stats.Correct), float64(stats.Total))
}

func safeDivide(numerator, denominator float64) float64 {
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// computeLatencyStats calculates min/average/median/P95/max. Percentiles use
// linear interpolation between closest ranks.
func computeLatencyStats(values []float64) LatencyStats {
	if len(values) == 0 {
		return LatencyStats{}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	var sum float64
	for _, value := range sorted {
		sum += value
	}

	return LatencyStats{
		MinMS:     sorted[0],
		AverageMS: sum / float64(len(sorted)),
		MedianMS:  percentile(sorted, 50),
		P95MS:     percentile(sorted, 95),
		MaxMS:     sorted[len(sorted)-1],
	}
}

// percentile returns the p-th percentile (0..100) of a sorted slice using
// linear interpolation between closest ranks.
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[n-1]
	}
	rank := (p / 100) * float64(n-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	fraction := rank - float64(lower)
	return sorted[lower]*(1-fraction) + sorted[upper]*fraction
}
