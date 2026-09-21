package main

import (
	"errors"
	"testing"
)

func TestParseSentimentValidPositive(t *testing.T) {
	result, err := ParseSentimentResponse(`{"sentiment":"positive","confidence":0.95}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Sentiment != LabelPositive || result.Confidence != 0.95 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestParseSentimentValidNegative(t *testing.T) {
	result, err := ParseSentimentResponse("  {\"sentiment\":\"negative\",\"confidence\":0.87}  \n")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Sentiment != LabelNegative || result.Confidence != 0.87 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestParseSentimentFencedJSON(t *testing.T) {
	raw := "```json\n{\"sentiment\":\"positive\",\"confidence\":0.97}\n```"
	result, err := ParseSentimentResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Sentiment != LabelPositive || result.Confidence != 0.97 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestParseSentimentPrettyPrinted(t *testing.T) {
	raw := "{\n  \"sentiment\": \"negative\",\n  \"confidence\": 0.5\n}"
	result, err := ParseSentimentResponse(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Sentiment != LabelNegative || result.Confidence != 0.5 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestParseSentimentStringConfidence(t *testing.T) {
	result, err := ParseSentimentResponse(`{"sentiment":"positive","confidence":"0.8"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Confidence != 0.8 {
		t.Fatalf("confidence = %v, want 0.8", result.Confidence)
	}
}

func TestParseSentimentBoundaryConfidence(t *testing.T) {
	for _, raw := range []string{
		`{"sentiment":"positive","confidence":0}`,
		`{"sentiment":"positive","confidence":1}`,
	} {
		if _, err := ParseSentimentResponse(raw); err != nil {
			t.Errorf("raw %s: unexpected error: %v", raw, err)
		}
	}
}

func TestParseSentimentInvalid(t *testing.T) {
	cases := map[string]string{
		"invalid sentiment":     `{"sentiment":"maybe","confidence":0.95}`,
		"confidence above one":  `{"sentiment":"positive","confidence":1.7}`,
		"confidence below zero": `{"sentiment":"positive","confidence":-0.2}`,
		"missing sentiment":     `{"confidence":0.95}`,
		"missing confidence":    `{"sentiment":"positive"}`,
		"prose only":            `The answer is positive because the person sounds happy.`,
		"empty":                 ``,
		"prose with word only":  `positive`,
		"array":                 `["positive", 0.95]`,
	}
	for name, raw := range cases {
		if _, err := ParseSentimentResponse(raw); !errors.Is(err, ErrInvalidModelResponse) {
			t.Errorf("%s: error = %v, want ErrInvalidModelResponse", name, err)
		}
	}
}

func TestParseSentimentRejectsTrailingProse(t *testing.T) {
	// A valid object followed by prose must not be accepted as strict JSON.
	raw := `{"sentiment":"positive","confidence":0.9} and that is my answer`
	if _, err := ParseSentimentResponse(raw); !errors.Is(err, ErrInvalidModelResponse) {
		t.Fatalf("error = %v, want ErrInvalidModelResponse", err)
	}
}

func TestStripCodeFence(t *testing.T) {
	cases := map[string]string{
		"```json\n{\"a\":1}\n```": `{"a":1}`,
		"```\n{\"a\":1}\n```":     `{"a":1}`,
		"{\"a\":1}":               `{"a":1}`,
		"  {\"a\":1}  ":           `{"a":1}`,
	}
	for input, want := range cases {
		if got := stripCodeFence(input); got != want {
			t.Errorf("stripCodeFence(%q) = %q, want %q", input, got, want)
		}
	}
}
