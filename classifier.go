package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Classification labels supported by the harness.
const (
	LabelPositive = "positive"
	LabelNegative = "negative"
)

// ErrInvalidModelResponse is returned when the model output cannot be parsed
// into a valid classification with a confidence in the range [0, 1].
var ErrInvalidModelResponse = errors.New("invalid model response")

// SystemPrompt instructs the model to act as a strict binary emotion
// classifier that replies with JSON only.
const SystemPrompt = `You are an emotion classification model.

Classify the input text as exactly one of:

positive
negative

Return JSON only:

{
  "classification": "positive",
  "confidence": 0.98
}

or:

{
  "classification": "negative",
  "confidence": 0.98
}

confidence must be a number between 0 and 1.

Do not provide explanations.
Do not provide additional fields.`

// Classification is a validated model prediction.
type Classification struct {
	Label      string
	Confidence float64
	// Raw holds the raw assistant content, even when parsing failed. This is
	// useful for verbose/debug output.
	Raw string
}

// Percent returns the confidence expressed as a percentage in [0, 100].
func (c Classification) Percent() float64 {
	return c.Confidence * 100
}

// confidenceValue accepts either a JSON number (0.98) or a numeric string
// ("0.98") and records whether a value was actually present.
type confidenceValue struct {
	value float64
	set   bool
}

func (c *confidenceValue) UnmarshalJSON(b []byte) error {
	s := strings.TrimSpace(string(b))
	if s == "null" || s == "" {
		return nil
	}
	if s[0] == '"' {
		var str string
		if err := json.Unmarshal(b, &str); err != nil {
			return err
		}
		s = strings.TrimSpace(str)
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return fmt.Errorf("confidence is not a number: %q", string(b))
	}
	c.value = f
	c.set = true
	return nil
}

// Classify sends text to the model and returns a validated classification.
// When parsing fails the returned Classification contains the raw model output
// so callers can surface it in verbose mode.
func Classify(ctx context.Context, client *Client, text string) (Classification, error) {
	content, err := client.ChatCompletion(ctx, SystemPrompt, text)
	if err != nil {
		return Classification{}, err
	}
	parsed, err := ParseModelResponse(content)
	if err != nil {
		return Classification{Raw: content}, err
	}
	return parsed, nil
}

// ParseModelResponse extracts and validates a classification from raw model
// output. It tolerates surrounding prose and markdown fences by scanning for
// brace-balanced JSON objects, but the JSON itself is validated strictly.
func ParseModelResponse(raw string) (Classification, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return Classification{}, fmt.Errorf("%w: empty response", ErrInvalidModelResponse)
	}

	candidates := extractJSONCandidates(trimmed)
	if len(candidates) == 0 {
		return Classification{}, fmt.Errorf("%w: no JSON object found", ErrInvalidModelResponse)
	}

	var lastErr error
	for _, cand := range candidates {
		var payload struct {
			Classification *string          `json:"classification"`
			Confidence     *confidenceValue `json:"confidence"`
		}
		if err := json.Unmarshal([]byte(cand), &payload); err != nil {
			lastErr = err
			continue
		}
		if payload.Classification == nil {
			lastErr = errors.New("missing classification field")
			continue
		}

		label := strings.ToLower(strings.TrimSpace(*payload.Classification))
		if label != LabelPositive && label != LabelNegative {
			lastErr = fmt.Errorf("invalid classification %q", *payload.Classification)
			continue
		}

		if payload.Confidence == nil || !payload.Confidence.set {
			lastErr = errors.New("missing confidence field")
			continue
		}

		conf := payload.Confidence.value
		if math.IsNaN(conf) || math.IsInf(conf, 0) || conf < 0 || conf > 1 {
			lastErr = fmt.Errorf("confidence %.4f out of range [0,1]", conf)
			continue
		}

		return Classification{Label: label, Confidence: conf, Raw: trimmed}, nil
	}

	return Classification{}, fmt.Errorf("%w: %v", ErrInvalidModelResponse, lastErr)
}

// extractJSONCandidates returns brace-balanced JSON object substrings found in
// s, including s itself when it is a single object. Candidates are returned in
// order of appearance and de-duplicated.
func extractJSONCandidates(s string) []string {
	var out []string
	seen := make(map[string]struct{})
	add := func(cand string) {
		cand = strings.TrimSpace(cand)
		if cand == "" {
			return
		}
		if _, ok := seen[cand]; ok {
			return
		}
		seen[cand] = struct{}{}
		out = append(out, cand)
	}

	// Fast path: the entire response is a JSON object.
	add(s)

	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		depth := 0
		inString := false
		escaped := false
		for j := i; j < len(s); j++ {
			ch := s[j]
			if inString {
				switch {
				case escaped:
					escaped = false
				case ch == '\\':
					escaped = true
				case ch == '"':
					inString = false
				}
				continue
			}
			switch ch {
			case '"':
				inString = true
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					add(s[i : j+1])
				}
			}
			if depth == 0 && j > i {
				break
			}
		}
	}

	return out
}
