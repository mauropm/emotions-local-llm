package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Default configuration values for the local OpenAI-compatible endpoint.
const (
	DefaultBaseURL = "http://127.0.0.1:8000/v1"
	DefaultModel   = "lfm2.5-1b-4bit"
	DefaultTimeout = 60 * time.Second

	// maxResponseBytes bounds how much of a response body we read.
	maxResponseBytes = 1 << 20 // 1 MiB

	// maxTokens keeps the model from producing long explanations while still
	// allowing a pretty-printed JSON object to be emitted in full.
	maxTokens = 64
)

// Config holds runtime settings for the harness.
type Config struct {
	BaseURL string
	Model   string
	APIKey  string
	Timeout time.Duration
	Verbose bool
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// Client is a minimal OpenAI-compatible chat completions client built on the
// standard library.
type Client struct {
	cfg  Config
	http *http.Client
}

// NewClient constructs a Client from cfg. A nil or non-positive timeout falls
// back to DefaultTimeout.
func NewClient(cfg Config) *Client {
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	return &Client{
		cfg:  cfg,
		http: &http.Client{Timeout: cfg.Timeout},
	}
}

// Config returns a copy of the client configuration.
func (c *Client) Config() Config { return c.cfg }

// ChatCompletion performs a single chat completion request and returns the
// assistant message content.
func (c *Client) ChatCompletion(ctx context.Context, system, user string) (string, error) {
	endpoint := strings.TrimRight(c.cfg.BaseURL, "/") + "/chat/completions"

	reqBody := chatRequest{
		Model: c.cfg.Model,
		Messages: []chatMessage{
			{Role: "system", Content: system},
			{Role: "user", Content: user},
		},
		Temperature: 0,
		MaxTokens:   maxTokens,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("encoding request: %w", err)
	}

	if c.cfg.Verbose {
		fmt.Fprintf(stderr, "POST %s\n", endpoint)
		fmt.Fprintf(stderr, "Model: %s\n", c.cfg.Model)
		fmt.Fprintf(stderr, "Temperature: 0\n")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("building request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("request to %s failed: %w", endpoint, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return "", fmt.Errorf("reading response: %w", err)
	}

	if c.cfg.Verbose {
		fmt.Fprintf(stderr, "Raw response:\n%s\n", strings.TrimSpace(string(raw)))
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("server returned HTTP %d: %s", resp.StatusCode, apiErrorMessage(raw))
	}

	var decoded chatResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "", fmt.Errorf("decoding response: %w", err)
	}
	if decoded.Error != nil {
		return "", fmt.Errorf("server error: %s", decoded.Error.Message)
	}
	if len(decoded.Choices) == 0 {
		return "", fmt.Errorf("response contained no choices")
	}
	content := decoded.Choices[0].Message.Content
	if strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("response contained no message content")
	}
	return content, nil
}

// apiErrorMessage extracts a helpful message from an error response body,
// falling back to the trimmed body itself.
func apiErrorMessage(raw []byte) string {
	var payload struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &payload); err == nil {
		if payload.Error.Message != "" {
			return payload.Error.Message
		}
		if payload.Message != "" {
			return payload.Message
		}
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return "(empty response body)"
	}
	if len(trimmed) > 500 {
		trimmed = trimmed[:500] + "..."
	}
	return trimmed
}
