package main

import (
	"testing"
	"time"
)

func TestDatasetShape(t *testing.T) {
	samples := Dataset()
	if len(samples) < 20 {
		t.Fatalf("dataset has %d samples, want >= 20", len(samples))
	}
	var positive, negative int
	for i, s := range samples {
		if s.ID != i+1 {
			t.Errorf("sample %d has ID %d, want %d", i, s.ID, i+1)
		}
		switch s.Expected {
		case LabelPositive:
			positive++
		case LabelNegative:
			negative++
		default:
			t.Errorf("sample %d has invalid expected label %q", i, s.Expected)
		}
	}
	if positive != 10 || negative != 10 {
		t.Fatalf("dataset labels = %d positive, %d negative; want 10/10", positive, negative)
	}
}

func TestComputeStatsAccuracyUsesLabelsNotConfidence(t *testing.T) {
	// A low-confidence correct prediction must count as correct, and a
	// high-confidence incorrect prediction must count as incorrect.
	results := []SampleResult{
		{ID: 1, Input: "a", Expected: LabelNegative, Predicted: LabelNegative, Confidence: 0.52, Correct: true, LatencyMS: 100},
		{ID: 2, Input: "b", Expected: LabelNegative, Predicted: LabelPositive, Confidence: 0.99, Correct: false, LatencyMS: 200},
	}
	stats := ComputeBenchmarkStats("m", "e", 1, results, 2*time.Second)

	if stats.Correct != 1 || stats.Incorrect != 1 {
		t.Fatalf("correct/incorrect = %d/%d, want 1/1", stats.Correct, stats.Incorrect)
	}
	if stats.Accuracy != 0.5 {
		t.Fatalf("accuracy = %v, want 0.5", stats.Accuracy)
	}
}

func TestComputeStatsAggregates(t *testing.T) {
	results := []SampleResult{
		{ID: 1, Expected: LabelPositive, Predicted: LabelPositive, Confidence: 0.90, Correct: true, LatencyMS: 120},
		{ID: 2, Expected: LabelPositive, Predicted: LabelPositive, Confidence: 0.80, Correct: true, LatencyMS: 140},
		{ID: 3, Expected: LabelNegative, Predicted: LabelNegative, Confidence: 0.70, Correct: true, LatencyMS: 160},
		{ID: 4, Expected: LabelNegative, Predicted: LabelPositive, Confidence: 0.60, Correct: false, LatencyMS: 180},
	}
	stats := ComputeBenchmarkStats("model", "endpoint", 4, results, 1*time.Second)

	if stats.Samples != 4 || stats.Correct != 3 || stats.Incorrect != 1 {
		t.Fatalf("unexpected counts: %+v", stats)
	}
	if stats.Accuracy != 0.75 {
		t.Errorf("accuracy = %v, want 0.75", stats.Accuracy)
	}
	if stats.AverageLatencyMS != 150 {
		t.Errorf("avg latency = %v, want 150", stats.AverageLatencyMS)
	}
	if stats.MinLatencyMS != 120 || stats.MaxLatencyMS != 180 {
		t.Errorf("min/max latency = %d/%d, want 120/180", stats.MinLatencyMS, stats.MaxLatencyMS)
	}
	wantConf := (0.90 + 0.80 + 0.70 + 0.60) / 4
	if diff := stats.AverageConfidence - wantConf; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("avg confidence = %v, want %v", stats.AverageConfidence, wantConf)
	}
	wantCorrectConf := (0.90 + 0.80 + 0.70) / 3
	if diff := stats.CorrectConfidence - wantCorrectConf; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("correct confidence = %v, want %v", stats.CorrectConfidence, wantCorrectConf)
	}
	if stats.IncorrectConfidence != 0.60 {
		t.Errorf("incorrect confidence = %v, want 0.60", stats.IncorrectConfidence)
	}
	if stats.Positive.Samples != 2 || stats.Positive.Correct != 2 || stats.Positive.Accuracy != 1.0 {
		t.Errorf("positive stats = %+v", stats.Positive)
	}
	if stats.Negative.Samples != 2 || stats.Negative.Correct != 1 || stats.Negative.Accuracy != 0.5 {
		t.Errorf("negative stats = %+v", stats.Negative)
	}
	if stats.RequestsPerSecond != 4 {
		t.Errorf("requests/sec = %v, want 4", stats.RequestsPerSecond)
	}
}

func TestComputeStatsRequestsPerSecond(t *testing.T) {
	results := []SampleResult{
		{ID: 1, Expected: LabelPositive, Predicted: LabelPositive, Correct: true, LatencyMS: 10},
		{ID: 2, Expected: LabelPositive, Predicted: LabelPositive, Correct: true, LatencyMS: 10},
	}
	stats := ComputeBenchmarkStats("m", "e", 2, results, 500*time.Millisecond)
	if stats.RequestsPerSecond != 4 {
		t.Fatalf("requests/sec = %v, want 4", stats.RequestsPerSecond)
	}
}

func TestComputeStatsErrorsCountAsIncorrect(t *testing.T) {
	results := []SampleResult{
		{ID: 1, Expected: LabelPositive, Correct: false, LatencyMS: 5},
	}
	stats := ComputeBenchmarkStats("m", "e", 1, results, time.Second)
	if stats.Incorrect != 1 || stats.Accuracy != 0 {
		t.Fatalf("stats = %+v; want one incorrect", stats)
	}
	// Failed predictions have no confidence and must not skew the average.
	if stats.AverageConfidence != 0 {
		t.Fatalf("average confidence = %v, want 0", stats.AverageConfidence)
	}
}

func TestComputeStatsEmpty(t *testing.T) {
	stats := ComputeBenchmarkStats("m", "e", 1, nil, time.Second)
	if stats.Samples != 0 || stats.Accuracy != 0 || stats.MinLatencyMS != 0 {
		t.Fatalf("unexpected stats for empty run: %+v", stats)
	}
}
