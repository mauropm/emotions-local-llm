package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientChatCompletionSuccess(t *testing.T) {
	var gotPath string
	var gotAuth string
	var gotBody chatRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"{\"classification\":\"positive\",\"confidence\":0.9}"}}]}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL + "/v1", Model: "test-model", Timeout: 5 * time.Second})
	content, err := client.ChatCompletion(context.Background(), "sys", "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", gotPath)
	}
	if gotAuth != "" {
		t.Errorf("unexpected Authorization header: %q", gotAuth)
	}
	if gotBody.Model != "test-model" || gotBody.Temperature != 0 {
		t.Errorf("unexpected request body: %+v", gotBody)
	}
	if !strings.Contains(content, "positive") {
		t.Errorf("content = %q", content)
	}
}

func TestClientSendsAPIKeyWhenSet(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", APIKey: "secret", Timeout: 5 * time.Second})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("Authorization = %q, want Bearer secret", gotAuth)
	}
}

func TestClientHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, `{"error":{"message":"model exploded"}}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", Timeout: 5 * time.Second})
	_, err := client.ChatCompletion(context.Background(), "s", "u")
	if err == nil || !strings.Contains(err.Error(), "model exploded") {
		t.Fatalf("error = %v, want it to mention the server message", err)
	}
}

func TestClientMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `not json`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", Timeout: 5 * time.Second})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err == nil {
		t.Fatal("expected error for malformed response")
	}
}

func TestClientMissingContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"choices":[{"message":{"content":""}}]}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", Timeout: 5 * time.Second})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err == nil {
		t.Fatal("expected error for empty content")
	}
}

func TestClientNoChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"choices":[]}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", Timeout: 5 * time.Second})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err == nil {
		t.Fatal("expected error for no choices")
	}
}

func TestClientConnectionRefused(t *testing.T) {
	// No server is listening on this port.
	client := NewClient(Config{BaseURL: "http://127.0.0.1:1", Model: "m", Timeout: 200 * time.Millisecond})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err == nil {
		t.Fatal("expected connection error")
	}
}

func TestClientTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		io.WriteString(w, `{"choices":[{"message":{"content":"late"}}]}`)
	}))
	defer srv.Close()

	client := NewClient(Config{BaseURL: srv.URL, Model: "m", Timeout: 50 * time.Millisecond})
	if _, err := client.ChatCompletion(context.Background(), "s", "u"); err == nil {
		t.Fatal("expected timeout error")
	}
}
