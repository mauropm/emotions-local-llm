package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// benchResultJSON is one line of the optional per-sample JSONL results file.
// Nullable fields are pointers so failures serialise as JSON null.
type benchResultJSON struct {
	ID                 int      `json:"id"`
	Text               string   `json:"text"`
	ExpectedEmotion    string   `json:"expected_emotion"`
	ExpectedSentiment  string   `json:"expected_sentiment"`
	PredictedSentiment *string  `json:"predicted_sentiment"`
	Confidence         *float64 `json:"confidence"`
	Correct            bool     `json:"correct"`
	Error              string   `json:"error,omitempty"`
	LatencyMS          float64  `json:"latency_ms"`
}

type latencyJSON struct {
	MinMS     float64 `json:"min"`
	AverageMS float64 `json:"average"`
	MedianMS  float64 `json:"median"`
	P95MS     float64 `json:"p95"`
	MaxMS     float64 `json:"max"`
}

type emotionStatsJSON struct {
	Total     int     `json:"total"`
	Correct   int     `json:"correct"`
	Incorrect int     `json:"incorrect"`
	Accuracy  float64 `json:"accuracy"`
}

// benchSummaryJSON is the machine-readable benchmark summary written by
// --output. It is designed to be easy to diff across models.
type benchSummaryJSON struct {
	Model       string `json:"model"`
	Endpoint    string `json:"endpoint"`
	Dataset     string `json:"dataset"`
	Seed        int    `json:"seed"`
	Timestamp   string `json:"timestamp"`
	Concurrency int    `json:"concurrency"`

	Total     int     `json:"total"`
	Correct   int     `json:"correct"`
	Incorrect int     `json:"incorrect"`
	Accuracy  float64 `json:"accuracy"`

	Precision     float64 `json:"precision"`
	Recall        float64 `json:"recall"`
	F1            float64 `json:"f1"`
	TruePositive  int     `json:"true_positive"`
	TrueNegative  int     `json:"true_negative"`
	FalsePositive int     `json:"false_positive"`
	FalseNegative int     `json:"false_negative"`

	AverageConfidence          float64     `json:"average_confidence"`
	Latency                    latencyJSON `json:"latency_ms"`
	ThroughputSamplesPerSecond float64     `json:"throughput_samples_per_second"`
	TotalDurationSeconds       float64     `json:"total_duration_seconds"`

	SuccessfulRequests int `json:"successful_requests"`
	FailedRequests     int `json:"failed_requests"`
	InvalidResponses   int `json:"invalid_responses"`

	Positive emotionStatsJSON            `json:"positive"`
	Negative emotionStatsJSON            `json:"negative"`
	Emotions map[string]emotionStatsJSON `json:"emotions"`
}

// PrintEmotionBenchmark writes the complete human-readable benchmark report to
// w, including the header and the post-run body.
func PrintEmotionBenchmark(w io.Writer, stats EmotionBenchmarkStats) {
	printEmotionBenchmarkHeader(w, stats)
	fmt.Fprintln(w)
	printEmotionBenchmarkBody(w, stats)
}

// printEmotionBenchmarkHeader writes the run identification block.
func printEmotionBenchmarkHeader(w io.Writer, stats EmotionBenchmarkStats) {
	fmt.Fprintln(w, "Emotion Benchmark")
	fmt.Fprintf(w, "Model: %s\n", stats.Model)
	fmt.Fprintf(w, "Endpoint: %s\n", stats.Endpoint)
	fmt.Fprintf(w, "Dataset: %s\n", stats.Dataset)
	fmt.Fprintf(w, "Seed: %d\n", stats.Seed)
	fmt.Fprintf(w, "Concurrency: %d\n", stats.Concurrency)
	fmt.Fprintf(w, "Timestamp: %s\n", stats.Timestamp.Format(time.RFC3339))
}

// printEmotionBenchmarkBody writes the results block.
func printEmotionBenchmarkBody(w io.Writer, stats EmotionBenchmarkStats) {
	fmt.Fprintln(w, "Benchmark complete.")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Samples: %d\n", stats.Total)
	fmt.Fprintf(w, "Correct: %d\n", stats.Correct)
	fmt.Fprintf(w, "Incorrect: %d\n", stats.Incorrect)
	fmt.Fprintf(w, "Accuracy: %.2f%%\n", stats.Accuracy*100)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Requests:")
	fmt.Fprintf(w, "  Total:              %d\n", stats.Total)
	fmt.Fprintf(w, "  Successful:         %d\n", stats.Successful)
	fmt.Fprintf(w, "  Failed:             %d\n", stats.Failed)
	fmt.Fprintf(w, "  Invalid responses:  %d\n", stats.Invalid)
	fmt.Fprintln(w)
	printEmotionCategory(w, "Positive", stats.Positive)
	fmt.Fprintln(w)
	printEmotionCategory(w, "Negative", stats.Negative)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Emotion accuracy:")
	fmt.Fprintln(w)
	for _, emotion := range EmotionOrder {
		emotionStats := stats.ByEmotion[emotion]
		fmt.Fprintf(
			w, "%-10s %6.1f%%  (%d/%d)\n",
			emotion, emotionStats.Accuracy*100, emotionStats.Correct, emotionStats.Total,
		)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Confusion matrix (valid predictions):")
	fmt.Fprintf(w, "  True Positive:  %d\n", stats.TruePositive)
	fmt.Fprintf(w, "  True Negative:  %d\n", stats.TrueNegative)
	fmt.Fprintf(w, "  False Positive: %d\n", stats.FalsePositive)
	fmt.Fprintf(w, "  False Negative: %d\n", stats.FalseNegative)
	fmt.Fprintf(w, "  Precision:      %.4f\n", stats.Precision)
	fmt.Fprintf(w, "  Recall:         %.4f\n", stats.Recall)
	fmt.Fprintf(w, "  F1:             %.4f\n", stats.F1)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Latency:")
	fmt.Fprintf(w, "  Min:       %.1f ms\n", stats.Latency.MinMS)
	fmt.Fprintf(w, "  Average:   %.1f ms\n", stats.Latency.AverageMS)
	fmt.Fprintf(w, "  Median:    %.1f ms\n", stats.Latency.MedianMS)
	fmt.Fprintf(w, "  P95:       %.1f ms\n", stats.Latency.P95MS)
	fmt.Fprintf(w, "  Max:       %.1f ms\n", stats.Latency.MaxMS)
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Throughput:")
	fmt.Fprintf(w, "  %.1f samples/sec\n", stats.ThroughputSamplesPerSecond)
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Average confidence: %.1f%%\n", stats.AverageConfidence*100)
	fmt.Fprintf(w, "Total time: %.2f s\n", stats.TotalDuration.Seconds())
}

func printEmotionCategory(w io.Writer, name string, stats EmotionStats) {
	fmt.Fprintf(w, "%s:\n", name)
	fmt.Fprintf(w, "  Total:     %d\n", stats.Total)
	fmt.Fprintf(w, "  Correct:   %d\n", stats.Correct)
	fmt.Fprintf(w, "  Incorrect: %d\n", stats.Incorrect)
	fmt.Fprintf(w, "  Accuracy:  %.2f%%\n", stats.Accuracy*100)
}

// WriteBenchResultsJSONL writes one JSON object per sample. Failed and invalid
// responses serialise predicted_sentiment and confidence as null and include
// an error message.
func WriteBenchResultsJSONL(w io.Writer, results []BenchResult) error {
	encoder := json.NewEncoder(w)
	for _, result := range results {
		record := benchResultJSON{
			ID:                result.ID,
			Text:              result.Text,
			ExpectedEmotion:   result.ExpectedEmotion,
			ExpectedSentiment: result.ExpectedSentiment,
			Correct:           result.Correct,
			Error:             result.Error,
			LatencyMS:         round(float64(result.Latency.Microseconds())/1000.0, 2),
		}
		if result.PredictedSentiment != "" {
			predicted := result.PredictedSentiment
			record.PredictedSentiment = &predicted
		}
		if result.HasConfidence {
			confidence := result.Confidence
			record.Confidence = &confidence
		}
		if err := encoder.Encode(record); err != nil {
			return err
		}
	}
	return nil
}

// WriteBenchSummaryJSON writes the machine-readable summary.
func WriteBenchSummaryJSON(w io.Writer, stats EmotionBenchmarkStats) error {
	emotions := make(map[string]emotionStatsJSON, len(stats.ByEmotion))
	for emotion, value := range stats.ByEmotion {
		emotions[emotion] = emotionStatsJSON{
			Total:     value.Total,
			Correct:   value.Correct,
			Incorrect: value.Incorrect,
			Accuracy:  round(value.Accuracy, 4),
		}
	}

	payload := benchSummaryJSON{
		Model:       stats.Model,
		Endpoint:    stats.Endpoint,
		Dataset:     stats.Dataset,
		Seed:        stats.Seed,
		Timestamp:   stats.Timestamp.Format(time.RFC3339),
		Concurrency: stats.Concurrency,

		Total:     stats.Total,
		Correct:   stats.Correct,
		Incorrect: stats.Incorrect,
		Accuracy:  round(stats.Accuracy, 4),

		Precision:     round(stats.Precision, 4),
		Recall:        round(stats.Recall, 4),
		F1:            round(stats.F1, 4),
		TruePositive:  stats.TruePositive,
		TrueNegative:  stats.TrueNegative,
		FalsePositive: stats.FalsePositive,
		FalseNegative: stats.FalseNegative,

		AverageConfidence: round(stats.AverageConfidence, 4),
		Latency: latencyJSON{
			MinMS:     round(stats.Latency.MinMS, 2),
			AverageMS: round(stats.Latency.AverageMS, 2),
			MedianMS:  round(stats.Latency.MedianMS, 2),
			P95MS:     round(stats.Latency.P95MS, 2),
			MaxMS:     round(stats.Latency.MaxMS, 2),
		},
		ThroughputSamplesPerSecond: round(stats.ThroughputSamplesPerSecond, 2),
		TotalDurationSeconds:       round(stats.TotalDuration.Seconds(), 3),

		SuccessfulRequests: stats.Successful,
		FailedRequests:     stats.Failed,
		InvalidResponses:   stats.Invalid,

		Positive: emotionStatsJSON{
			Total:     stats.Positive.Total,
			Correct:   stats.Positive.Correct,
			Incorrect: stats.Positive.Incorrect,
			Accuracy:  round(stats.Positive.Accuracy, 4),
		},
		Negative: emotionStatsJSON{
			Total:     stats.Negative.Total,
			Correct:   stats.Negative.Correct,
			Incorrect: stats.Negative.Incorrect,
			Accuracy:  round(stats.Negative.Accuracy, 4),
		},
		Emotions: emotions,
	}

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(payload)
}

// progressReporter renders a throttled progress bar to a writer. It is safe
// for concurrent use by benchmark workers.
type progressReporter struct {
	writer      io.Writer
	total       int
	step        int
	lastPrinted int
	mu          sync.Mutex
}

func newProgressReporter(writer io.Writer, total int) *progressReporter {
	step := total / 50
	if step < 1 {
		step = 1
	}
	return &progressReporter{writer: writer, total: total, step: step}
}

// update is called after each completed request. The final render is left to
// finish so the completed bar is printed exactly once.
func (p *progressReporter) update(done, total int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if done >= total {
		return
	}
	if done-p.lastPrinted < p.step {
		return
	}
	p.lastPrinted = done
	p.render(done, total)
}

// finish renders the completed bar and terminates the line.
func (p *progressReporter) finish() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.render(p.total, p.total)
	fmt.Fprintln(p.writer)
}

func (p *progressReporter) render(done, total int) {
	const width = 30
	filled := 0
	if total > 0 {
		filled = done * width / total
	}
	if filled > width {
		filled = width
	}
	bar := strings.Repeat("=", filled) + strings.Repeat(" ", width-filled)
	fmt.Fprintf(p.writer, "\r[%s] %d/%d", bar, done, total)
}
