package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// BenchSystemPrompt is the deterministic classification prompt used for the
// 1,000-sample emotion benchmark. It asks for the `sentiment` field to
// distinguish it from the legacy single-text classifier prompt.
const BenchSystemPrompt = `You are an emotion classification system.

Classify the emotional sentiment of the user's text.

Return ONLY valid JSON in this exact format:

{"sentiment":"positive","confidence":0.95}

The sentiment MUST be exactly one of:
positive
negative

confidence MUST be a number between 0 and 1.

Do not include markdown.
Do not include explanations.
Do not include additional fields.`

// SentimentResult is a validated benchmark model prediction.
type SentimentResult struct {
	Sentiment  string
	Confidence float64
	// Raw holds the raw assistant content, even on parse failure, for debugging.
	Raw string
}

// ClassifySentiment sends text to the model using the benchmark prompt and
// returns a validated sentiment prediction.
func ClassifySentiment(ctx context.Context, client *Client, text string) (SentimentResult, error) {
	content, err := client.ChatCompletion(ctx, BenchSystemPrompt, text)
	if err != nil {
		return SentimentResult{}, err
	}
	parsed, err := ParseSentimentResponse(content)
	if err != nil {
		return SentimentResult{Raw: content}, err
	}
	return parsed, nil
}

// ParseSentimentResponse parses a strict benchmark response. It accepts a JSON
// object with a `sentiment` in {positive, negative} and a numeric `confidence`
// in [0, 1], optionally wrapped in a single markdown code fence and surrounded
// by whitespace. Arbitrary prose is rejected.
func ParseSentimentResponse(raw string) (SentimentResult, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return SentimentResult{}, fmt.Errorf("%w: empty response", ErrInvalidModelResponse)
	}

	candidate := stripCodeFence(trimmed)
	if candidate == "" {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: empty response after fence", ErrInvalidModelResponse)
	}

	var payload struct {
		Sentiment  *string          `json:"sentiment"`
		Confidence *confidenceValue `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(candidate), &payload); err != nil {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: %v", ErrInvalidModelResponse, err)
	}

	if payload.Sentiment == nil {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: missing sentiment field", ErrInvalidModelResponse)
	}
	sentiment := strings.ToLower(strings.TrimSpace(*payload.Sentiment))
	if sentiment != LabelPositive && sentiment != LabelNegative {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: invalid sentiment %q", ErrInvalidModelResponse, *payload.Sentiment)
	}

	if payload.Confidence == nil || !payload.Confidence.set {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: missing confidence field", ErrInvalidModelResponse)
	}
	confidence := payload.Confidence.value
	if math.IsNaN(confidence) || math.IsInf(confidence, 0) || confidence < 0 || confidence > 1 {
		return SentimentResult{Raw: trimmed}, fmt.Errorf("%w: confidence %.4f out of range [0,1]", ErrInvalidModelResponse, confidence)
	}

	return SentimentResult{Sentiment: sentiment, Confidence: confidence, Raw: trimmed}, nil
}

// stripCodeFence removes a single surrounding ```json ... ``` markdown fence if
// present. Responses without a fence are returned unchanged.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// Drop the opening fence line (e.g. "```json").
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		s = s[idx+1:]
	} else {
		s = strings.TrimPrefix(s, "```")
	}
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}
