package main

import (
	"context"
	"math"
	"sync"
	"time"
)

// Sample is a single labelled benchmark example.
type Sample struct {
	ID       int
	Text     string
	Expected string
}

// SampleResult captures the outcome of classifying one sample.
type SampleResult struct {
	ID         int
	Input      string
	Expected   string
	Predicted  string
	Confidence float64
	Correct    bool
	LatencyMS  int64
	Err        error
}

// CategoryStats aggregates results for a single expected label.
type CategoryStats struct {
	Samples  int
	Correct  int
	Accuracy float64
}

// BenchmarkStats summarises a full benchmark run.
type BenchmarkStats struct {
	Model       string
	Endpoint    string
	Concurrency int

	Samples   int
	Correct   int
	Incorrect int
	Accuracy  float64

	AverageConfidence   float64
	CorrectConfidence   float64
	IncorrectConfidence float64

	AverageLatencyMS float64
	MinLatencyMS     int64
	MaxLatencyMS     int64

	TotalDuration     time.Duration
	RequestsPerSecond float64

	Positive CategoryStats
	Negative CategoryStats

	Results []SampleResult
}

// Dataset returns the built-in benchmark dataset: 10 positive and 10 negative
// examples, mixing obvious and somewhat ambiguous statements.
func Dataset() []Sample {
	samples := []Sample{
		{Text: "I had an amazing day.", Expected: LabelPositive},
		{Text: "I am extremely happy.", Expected: LabelPositive},
		{Text: "Everything worked perfectly.", Expected: LabelPositive},
		{Text: "I am excited about the future.", Expected: LabelPositive},
		{Text: "This is wonderful news.", Expected: LabelPositive},
		{Text: "I feel fantastic today.", Expected: LabelPositive},
		{Text: "I am proud of what we accomplished.", Expected: LabelPositive},
		{Text: "That was a beautiful experience.", Expected: LabelPositive},
		{Text: "I love how this turned out.", Expected: LabelPositive},
		{Text: "I am very grateful.", Expected: LabelPositive},

		{Text: "I had a terrible day.", Expected: LabelNegative},
		{Text: "I am extremely angry.", Expected: LabelNegative},
		{Text: "Everything went wrong.", Expected: LabelNegative},
		{Text: "I am disappointed with the result.", Expected: LabelNegative},
		{Text: "This is horrible news.", Expected: LabelNegative},
		{Text: "I feel miserable today.", Expected: LabelNegative},
		{Text: "I regret what happened.", Expected: LabelNegative},
		{Text: "That was a terrible experience.", Expected: LabelNegative},
		{Text: "I hate how this turned out.", Expected: LabelNegative},
		{Text: "I am very frustrated.", Expected: LabelNegative},
	}
	for i := range samples {
		samples[i].ID = i + 1
	}
	return samples
}

// RunBenchmark classifies every sample, optionally concurrently, and returns
// the aggregated statistics. Accuracy is derived purely from comparing the
// predicted label with the expected label; model confidence never influences
// correctness.
func RunBenchmark(ctx context.Context, client *Client, samples []Sample, concurrency int) BenchmarkStats {
	if concurrency < 1 {
		concurrency = 1
	}

	results := make([]SampleResult, len(samples))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	start := time.Now()
	for i, sample := range samples {
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, s Sample) {
			defer wg.Done()
			defer func() { <-sem }()

			t0 := time.Now()
			classification, err := Classify(ctx, client, s.Text)
			latency := time.Since(t0)

			res := SampleResult{
				ID:        s.ID,
				Input:     s.Text,
				Expected:  s.Expected,
				LatencyMS: latency.Milliseconds(),
			}
			if err != nil {
				res.Err = err
			} else {
				res.Predicted = classification.Label
				res.Confidence = classification.Confidence
			}
			res.Correct = res.Predicted == s.Expected
			results[idx] = res
		}(i, sample)
	}
	wg.Wait()
	total := time.Since(start)

	return ComputeBenchmarkStats(client.Config().Model, client.Config().BaseURL, concurrency, results, total)
}

// ComputeBenchmarkStats aggregates per-sample results into summary statistics.
// It is pure and therefore unit-testable without a model server.
func ComputeBenchmarkStats(model, endpoint string, concurrency int, results []SampleResult, total time.Duration) BenchmarkStats {
	stats := BenchmarkStats{
		Model:         model,
		Endpoint:      endpoint,
		Concurrency:   concurrency,
		Samples:       len(results),
		TotalDuration: total,
		Results:       results,
	}
	if len(results) == 0 {
		return stats
	}

	var (
		confSum            float64
		confCount          int
		correctConfSum     float64
		correctConfCount   int
		incorrectConfSum   float64
		incorrectConfCount int
		latencySum         int64
	)
	minLatency := int64(math.MaxInt64)
	var maxLatency int64

	for _, r := range results {
		if r.LatencyMS < minLatency {
			minLatency = r.LatencyMS
		}
		if r.LatencyMS > maxLatency {
			maxLatency = r.LatencyMS
		}
		latencySum += r.LatencyMS

		if r.Correct {
			stats.Correct++
		} else {
			stats.Incorrect++
		}

		// Confidence statistics only make sense for valid predictions.
		if r.Predicted != "" {
			confSum += r.Confidence
			confCount++
			if r.Correct {
				correctConfSum += r.Confidence
				correctConfCount++
			} else {
				incorrectConfSum += r.Confidence
				incorrectConfCount++
			}
		}

		switch r.Expected {
		case LabelPositive:
			stats.Positive.Samples++
			if r.Correct {
				stats.Positive.Correct++
			}
		case LabelNegative:
			stats.Negative.Samples++
			if r.Correct {
				stats.Negative.Correct++
			}
		}
	}

	stats.Accuracy = float64(stats.Correct) / float64(stats.Samples)
	stats.AverageLatencyMS = float64(latencySum) / float64(len(results))
	stats.MinLatencyMS = minLatency
	stats.MaxLatencyMS = maxLatency
	if confCount > 0 {
		stats.AverageConfidence = confSum / float64(confCount)
	}
	if correctConfCount > 0 {
		stats.CorrectConfidence = correctConfSum / float64(correctConfCount)
	}
	if incorrectConfCount > 0 {
		stats.IncorrectConfidence = incorrectConfSum / float64(incorrectConfCount)
	}
	if stats.Positive.Samples > 0 {
		stats.Positive.Accuracy = float64(stats.Positive.Correct) / float64(stats.Positive.Samples)
	}
	if stats.Negative.Samples > 0 {
		stats.Negative.Accuracy = float64(stats.Negative.Correct) / float64(stats.Negative.Samples)
	}
	if seconds := total.Seconds(); seconds > 0 {
		stats.RequestsPerSecond = float64(len(results)) / seconds
	}

	return stats
}
