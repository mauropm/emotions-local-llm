package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
)

// benchmarkJSON is the machine-readable benchmark summary. Field names and
// structure intentionally match the documented JSON output.
type benchmarkJSON struct {
	Model             string  `json:"model"`
	Endpoint          string  `json:"endpoint"`
	Samples           int     `json:"samples"`
	Correct           int     `json:"correct"`
	Incorrect         int     `json:"incorrect"`
	Accuracy          float64 `json:"accuracy"`
	AverageConfidence float64 `json:"average_confidence"`
	AverageLatencyMS  float64 `json:"average_latency_ms"`
	MinLatencyMS      int64   `json:"min_latency_ms"`
	MaxLatencyMS      int64   `json:"max_latency_ms"`
	RequestsPerSecond float64 `json:"requests_per_second"`
}

// PrintBenchmark writes the human-readable benchmark report to w.
func PrintBenchmark(w io.Writer, stats BenchmarkStats) {
	fmt.Fprintln(w, "Emotion Classification Benchmark")
	fmt.Fprintln(w, "================================")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Model: %s\n", stats.Model)
	fmt.Fprintf(w, "Endpoint: %s\n", stats.Endpoint)
	fmt.Fprintln(w)
	if stats.Concurrency > 1 {
		fmt.Fprintf(w, "Concurrency: %d\n", stats.Concurrency)
	}
	fmt.Fprintf(w, "Samples: %d\n", stats.Samples)
	fmt.Fprintf(w, "Correct: %d\n", stats.Correct)
	fmt.Fprintf(w, "Incorrect: %d\n", stats.Incorrect)
	fmt.Fprintf(w, "Accuracy: %.2f%%\n", stats.Accuracy*100)
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Total time: %.2f s\n", stats.TotalDuration.Seconds())
	fmt.Fprintf(w, "Average latency: %.0f ms\n", stats.AverageLatencyMS)
	fmt.Fprintf(w, "Min latency: %d ms\n", stats.MinLatencyMS)
	fmt.Fprintf(w, "Max latency: %d ms\n", stats.MaxLatencyMS)
	fmt.Fprintf(w, "Requests/sec: %.2f\n", stats.RequestsPerSecond)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Positive:")
	fmt.Fprintf(w, "  Samples: %d\n", stats.Positive.Samples)
	fmt.Fprintf(w, "  Correct: %d\n", stats.Positive.Correct)
	fmt.Fprintf(w, "  Accuracy: %.2f%%\n", stats.Positive.Accuracy*100)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Negative:")
	fmt.Fprintf(w, "  Samples: %d\n", stats.Negative.Samples)
	fmt.Fprintf(w, "  Correct: %d\n", stats.Negative.Correct)
	fmt.Fprintf(w, "  Accuracy: %.2f%%\n", stats.Negative.Accuracy*100)
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Average confidence: %.1f%%\n", stats.AverageConfidence*100)
	fmt.Fprintf(w, "Correct predictions average confidence: %.1f%%\n", stats.CorrectConfidence*100)
	fmt.Fprintf(w, "Incorrect predictions average confidence: %.1f%%\n", stats.IncorrectConfidence*100)
}

// WriteBenchmarkJSON writes the machine-readable benchmark summary to w. No
// human-readable text is written in this mode.
func WriteBenchmarkJSON(w io.Writer, stats BenchmarkStats) error {
	payload := benchmarkJSON{
		Model:             stats.Model,
		Endpoint:          stats.Endpoint,
		Samples:           stats.Samples,
		Correct:           stats.Correct,
		Incorrect:         stats.Incorrect,
		Accuracy:          round(stats.Accuracy, 4),
		AverageConfidence: round(stats.AverageConfidence, 4),
		AverageLatencyMS:  round(stats.AverageLatencyMS, 1),
		MinLatencyMS:      stats.MinLatencyMS,
		MaxLatencyMS:      stats.MaxLatencyMS,
		RequestsPerSecond: round(stats.RequestsPerSecond, 2),
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(payload)
}

// WriteBenchmarkCSV writes per-sample results to w using encoding/csv.
func WriteBenchmarkCSV(w io.Writer, stats BenchmarkStats) error {
	cw := csv.NewWriter(w)
	if err := cw.Write([]string{"id", "input", "expected", "predicted", "confidence", "correct", "latency_ms"}); err != nil {
		return err
	}
	for _, r := range stats.Results {
		record := []string{
			strconv.Itoa(r.ID),
			r.Input,
			r.Expected,
			r.Predicted,
			strconv.FormatFloat(r.Confidence, 'f', -1, 64),
			strconv.FormatBool(r.Correct),
			strconv.FormatInt(r.LatencyMS, 10),
		}
		if err := cw.Write(record); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

func round(value float64, places int) float64 {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	factor := math.Pow(10, float64(places))
	return math.Round(value*factor) / factor
}
