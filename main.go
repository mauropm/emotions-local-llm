package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// stderr is the destination for diagnostics. It is kept separate from stdout
// so that JSON output is never polluted by human-readable logging.
var stderr io.Writer = os.Stderr

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		benchmark   = flag.Bool("benchmark", false, "run the built-in benchmark dataset")
		concurrency = flag.Int("concurrency", 1, "number of concurrent requests in benchmark mode")
		csvPath     = flag.String("csv", "", "write benchmark per-sample results to the given CSV file")
		jsonOut     = flag.Bool("json", false, "emit benchmark results as JSON")
		verbose     = flag.Bool("verbose", false, "enable verbose/debug output")
	)
	flag.BoolVar(verbose, "v", false, "enable verbose/debug output")
	flag.Parse()

	cfg, err := loadConfig(*verbose)
	if err != nil {
		return err
	}

	// Additive subcommand: `emotion-harness benchmark [flags]` runs the
	// 1,000-sample dataset benchmark. The legacy `--benchmark` flag below is
	// left untouched.
	args := flag.Args()
	if len(args) > 0 && args[0] == "benchmark" {
		return runEmotionBenchmarkCommand(args[1:], cfg, *concurrency)
	}

	client := NewClient(cfg)

	if *benchmark {
		return runBenchmarkMode(client, *concurrency, *csvPath, *jsonOut)
	}
	if *csvPath != "" {
		return errors.New("--csv can only be used together with --benchmark")
	}
	if *jsonOut {
		return errors.New("--json can only be used together with --benchmark")
	}

	text, err := readInput(args)
	if err != nil {
		return err
	}
	return classifyOnce(client, text, *verbose)
}

// loadConfig reads configuration from the environment, applying defaults.
func loadConfig(verbose bool) (Config, error) {
	cfg := Config{
		BaseURL: envOr("MODEL_BASE_URL", DefaultBaseURL),
		Model:   envOr("MODEL_NAME", DefaultModel),
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		Timeout: DefaultTimeout,
		Verbose: verbose,
	}
	if raw := strings.TrimSpace(os.Getenv("MODEL_TIMEOUT")); raw != "" {
		d, err := time.ParseDuration(raw)
		if err != nil {
			return Config{}, fmt.Errorf("invalid MODEL_TIMEOUT %q: %w", raw, err)
		}
		if d <= 0 {
			return Config{}, fmt.Errorf("invalid MODEL_TIMEOUT %q: must be positive", raw)
		}
		cfg.Timeout = d
	}
	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

// readInput returns the text to classify from CLI arguments, falling back to
// stdin when no argument is provided.
func readInput(args []string) (string, error) {
	if len(args) > 0 {
		text := strings.TrimSpace(strings.Join(args, " "))
		if text == "" {
			return "", errors.New("input text is empty")
		}
		return text, nil
	}

	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("reading stdin: %w", err)
	}
	text := strings.TrimSpace(string(data))
	if text == "" {
		return "", errors.New("no input provided; pass text as an argument or via stdin")
	}
	return text, nil
}

func classifyOnce(client *Client, text string, verbose bool) error {
	ctx := context.Background()
	start := time.Now()
	result, err := Classify(ctx, client, text)
	latency := time.Since(start)

	fmt.Printf("Input: %s\n\n", text)
	if err != nil {
		if errors.Is(err, ErrInvalidModelResponse) {
			fmt.Fprintf(os.Stderr, "ERROR: invalid model response\n")
			if verbose && result.Raw != "" {
				fmt.Fprintf(os.Stderr, "Raw response:\n%s\n", result.Raw)
			}
			os.Exit(2)
		}
		return err
	}

	fmt.Printf("Classification: %s\n", result.Label)
	fmt.Printf("Confidence: %.1f%%\n", result.Percent())
	fmt.Printf("Latency: %d ms\n", latency.Milliseconds())
	return nil
}

func runBenchmarkMode(client *Client, concurrency int, csvPath string, jsonOut bool) error {
	if concurrency < 1 {
		return fmt.Errorf("invalid --concurrency %d: must be >= 1", concurrency)
	}

	samples := Dataset()
	ctx := context.Background()
	stats := RunBenchmark(ctx, client, samples, concurrency)

	// Surface request failures without polluting stdout JSON output.
	failures := 0
	for _, r := range stats.Results {
		if r.Err != nil {
			failures++
			fmt.Fprintf(stderr, "request %d failed: %v\n", r.ID, r.Err)
		}
	}
	if failures > 0 {
		fmt.Fprintf(stderr, "Warning: %d of %d requests failed\n", failures, stats.Samples)
	}

	if csvPath != "" {
		if err := writeCSVFile(csvPath, stats); err != nil {
			return err
		}
		if !jsonOut {
			fmt.Fprintf(stderr, "Wrote CSV results to %s\n", csvPath)
		}
	}

	if jsonOut {
		return WriteBenchmarkJSON(os.Stdout, stats)
	}
	PrintBenchmark(os.Stdout, stats)
	return nil
}

func writeCSVFile(path string, stats BenchmarkStats) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating CSV file: %w", err)
	}
	defer f.Close()
	if err := WriteBenchmarkCSV(f, stats); err != nil {
		return fmt.Errorf("writing CSV file: %w", err)
	}
	return nil
}

// runEmotionBenchmarkCommand implements `emotion-harness benchmark`, the
// additive 1,000-sample dataset benchmark. The legacy `--benchmark` flag is
// unaffected.
func runEmotionBenchmarkCommand(args []string, cfg Config, defaultConcurrency int) error {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "Usage: emotion-harness benchmark [flags]")
		fmt.Fprintln(stderr)
		fmt.Fprintln(stderr, "Runs the reproducible 1,000-sample emotion benchmark.")
		fmt.Fprintln(stderr)
		fs.PrintDefaults()
	}

	endpoint := fs.String("endpoint", cfg.BaseURL, "OpenAI-compatible base URL")
	model := fs.String("model", cfg.Model, "model name")
	datasetPath := fs.String("dataset", DefaultDatasetPath, "path to the JSONL benchmark dataset")
	concurrency := fs.Int("concurrency", defaultConcurrency, "number of concurrent requests")
	resultsPath := fs.String("results", "", "write per-sample results to a JSONL file")
	outputPath := fs.String("output", "", "write the summary to a JSON file")
	verbose := fs.Bool("verbose", cfg.Verbose, "enable verbose/debug output")
	fs.BoolVar(verbose, "v", cfg.Verbose, "enable verbose/debug output")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *concurrency < 1 {
		return fmt.Errorf("invalid --concurrency %d: must be >= 1", *concurrency)
	}

	runCfg := cfg
	runCfg.BaseURL = *endpoint
	runCfg.Model = *model
	runCfg.Verbose = *verbose
	client := NewClient(runCfg)

	samples, err := LoadDataset(*datasetPath)
	if err != nil {
		return err
	}
	if err := ValidateDataset(samples); err != nil {
		return fmt.Errorf("dataset validation failed: %w", err)
	}

	progress := newProgressReporter(stderr, len(samples))
	started := time.Now()

	meta := BenchMetadata{
		Model:       runCfg.Model,
		Endpoint:    runCfg.BaseURL,
		Dataset:     *datasetPath,
		Seed:        DatasetSeed,
		Concurrency: *concurrency,
		StartedAt:   started,
	}
	printEmotionBenchmarkHeader(os.Stdout, EmotionBenchmarkStats{BenchMetadata: meta, Timestamp: started})
	fmt.Fprintln(os.Stdout)
	fmt.Fprintln(os.Stdout, "Running...")

	results, duration := RunEmotionBenchmark(context.Background(), client, samples, *concurrency, progress.update)
	progress.finish()

	stats := ComputeEmotionBenchmarkStats(meta, results, duration)
	fmt.Fprintln(os.Stdout)
	printEmotionBenchmarkBody(os.Stdout, stats)

	if *resultsPath != "" {
		if err := writeResultsFile(*resultsPath, results); err != nil {
			return err
		}
		fmt.Fprintf(stderr, "Wrote per-sample results to %s\n", *resultsPath)
	}
	if *outputPath != "" {
		if err := writeSummaryFile(*outputPath, stats); err != nil {
			return err
		}
		fmt.Fprintf(stderr, "Wrote summary to %s\n", *outputPath)
	}
	return nil
}

func writeResultsFile(path string, results []BenchResult) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating results file: %w", err)
	}
	defer f.Close()
	if err := WriteBenchResultsJSONL(f, results); err != nil {
		return fmt.Errorf("writing results file: %w", err)
	}
	return nil
}

func writeSummaryFile(path string, stats EmotionBenchmarkStats) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating summary file: %w", err)
	}
	defer f.Close()
	if err := WriteBenchSummaryJSON(f, stats); err != nil {
		return fmt.Errorf("writing summary file: %w", err)
	}
	return nil
}
