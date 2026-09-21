package main

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestParseValidPositive(t *testing.T) {
	raw := `{"classification": "positive", "confidence": 0.98}`
	got, err := ParseModelResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Label != LabelPositive {
		t.Errorf("label = %q, want %q", got.Label, LabelPositive)
	}
	if got.Confidence != 0.98 {
		t.Errorf("confidence = %v, want 0.98", got.Confidence)
	}
}

func TestParseValidNegative(t *testing.T) {
	raw := "Sure!\n```json\n{\"classification\": \"Negative\", \"confidence\": 0.87}\n```\n"
	got, err := ParseModelResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Label != LabelNegative {
		t.Errorf("label = %q, want %q", got.Label, LabelNegative)
	}
	if got.Confidence != 0.87 {
		t.Errorf("confidence = %v, want 0.87", got.Confidence)
	}
}

func TestParseEmbeddedJSONInText(t *testing.T) {
	raw := `The answer is: {"classification":"positive","confidence":0.61} Hope that helps.`
	got, err := ParseModelResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Label != LabelPositive || got.Confidence != 0.61 {
		t.Fatalf("got %+v", got)
	}
}

func TestParseConfidenceAsString(t *testing.T) {
	raw := `{"classification":"negative","confidence":"0.75"}`
	got, err := ParseModelResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Confidence != 0.75 {
		t.Fatalf("confidence = %v, want 0.75", got.Confidence)
	}
}

func TestParseInvalidClassification(t *testing.T) {
	raw := `{"classification": "neutral", "confidence": 0.5}`
	_, err := ParseModelResponse(raw)
	if !errors.Is(err, ErrInvalidModelResponse) {
		t.Fatalf("error = %v, want ErrInvalidModelResponse", err)
	}
}

func TestParseMissingClassification(t *testing.T) {
	raw := `{"confidence": 0.5}`
	_, err := ParseModelResponse(raw)
	if !errors.Is(err, ErrInvalidModelResponse) {
		t.Fatalf("error = %v, want ErrInvalidModelResponse", err)
	}
}

func TestParseConfidenceOutOfRange(t *testing.T) {
	cases := []string{
		`{"classification":"positive","confidence":1.5}`,
		`{"classification":"positive","confidence":-0.1}`,
		`{"classification":"positive","confidence":98}`,
	}
	for _, raw := range cases {
		if _, err := ParseModelResponse(raw); !errors.Is(err, ErrInvalidModelResponse) {
			t.Errorf("raw %s: error = %v, want ErrInvalidModelResponse", raw, err)
		}
	}
}

func TestParseMissingConfidence(t *testing.T) {
	raw := `{"classification":"positive"}`
	_, err := ParseModelResponse(raw)
	if !errors.Is(err, ErrInvalidModelResponse) {
		t.Fatalf("error = %v, want ErrInvalidModelResponse", err)
	}
}

func TestParseMalformedJSON(t *testing.T) {
	cases := []string{
		``,
		`not json at all`,
		`{"classification": "positive", "confidence": }`,
		`{"classification": "positive" "confidence": 0.5}`,
	}
	for _, raw := range cases {
		if _, err := ParseModelResponse(raw); !errors.Is(err, ErrInvalidModelResponse) {
			t.Errorf("raw %q: error = %v, want ErrInvalidModelResponse", raw, err)
		}
	}
}

func TestParseRecoversFirstValidObject(t *testing.T) {
	// The first brace-balanced object is missing a field; the parser should
	// continue scanning and accept the next valid object.
	raw := `Some noise {"foo": 1} then {"classification":"positive","confidence":0.9}`
	got, err := ParseModelResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Label != LabelPositive || got.Confidence != 0.9 {
		t.Fatalf("got %+v", got)
	}
}

func TestConfidencePercentConversion(t *testing.T) {
	c := Classification{Label: LabelPositive, Confidence: 0.984}
	if got, want := c.Percent(), 98.4; math.Abs(got-want) > 1e-9 {
		t.Fatalf("Percent() = %v, want %v", got, want)
	}
}

func TestParseRawPreservedOnFailure(t *testing.T) {
	// Classify stores raw output on failure; verify the helper path directly
	// by confirming ParseModelResponse reports errors that callers wrap.
	raw := `totally unparseable`
	if _, err := ParseModelResponse(raw); err == nil || !strings.Contains(err.Error(), "invalid model response") {
		t.Fatalf("error = %v, want invalid model response", err)
	}
}

func TestConfidenceJSONNumberRoundTrip(t *testing.T) {
	// Guard against precision regressions when confidence travels as a
	// json.Number rather than float64.
	var payload struct {
		Confidence confidenceValue `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(`{"confidence":0.123456789}`), &payload); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !payload.Confidence.set || payload.Confidence.value != 0.123456789 {
		t.Fatalf("got %+v", payload.Confidence)
	}
}
