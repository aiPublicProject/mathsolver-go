package mathsolver

/**
 * HTTP-interface mock (no server, no sockets): a fake http.RoundTripper is
 * injected via NewWithHTTPClient so the DEFAULT transport runs its real code
 * path (URL building, headers, body serialization, status handling, response
 * envelope parsing) against synthetic OpenAI-shaped responses. Runs
 * identically in local dev and GitHub CI.
 */

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type capturedCall struct {
	url    string
	auth   string
	ctType string
	body   map[string]any
}

// mockHTTPClient returns an *http.Client serving `contents` (model reply texts)
// in order; entries of statuses[i] > 0 reply with that HTTP status instead.
// Calls are appended to *captured.
func mockHTTPClient(contents []string, statuses []int, captured *[]capturedCall) *http.Client {
	n := 0
	return &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		call := capturedCall{url: r.URL.String(), auth: r.Header.Get("Authorization"), ctType: r.Header.Get("Content-Type")}
		raw, _ := io.ReadAll(r.Body)
		if json.Unmarshal(raw, &call.body) != nil {
			call.body = map[string]any{"_raw": string(raw)}
		}
		*captured = append(*captured, call)
		i := n
		n++
		content := goodReply
		if i < len(contents) && contents[i] != "" {
			content = contents[i]
		}
		status := 200
		if i < len(statuses) && statuses[i] > 0 {
			status = statuses[i]
			content = "upstream boom"
		}
		envelope := `{"choices":[{"message":{"content":` + jsonString(content) + `}}]}`
		return &http.Response{
			StatusCode: status,
			Body:       io.NopCloser(strings.NewReader(envelope)),
			Header:     http.Header{"Content-Type": []string{"application/json"}},
		}, nil
	})}
}

// jsonString escapes s minimally for embedding inside a JSON string literal.
func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestHTTPMockFullRoundTrip(t *testing.T) {
	var calls []capturedCall
	hc := mockHTTPClient([]string{goodReply}, nil, &calls)
	solver, err := NewWithHTTPClient("sk-mock", "https://mock.test/v1", hc)
	if err != nil {
		t.Fatal(err)
	}
	solver.Model = "mock-model"
	r, err := solver.Solve("2x + 3 = 11, solve for x")
	if err != nil {
		t.Fatal(err)
	}
	if r.Answer != 4 || !r.Verified || r.Retries != 0 {
		t.Fatalf("r=%+v", r)
	}
	if len(calls) != 1 {
		t.Fatalf("calls=%d", len(calls))
	}
	if calls[0].url != "https://mock.test/v1/chat/completions" {
		t.Fatalf("url=%s", calls[0].url)
	}
	if calls[0].auth != "Bearer sk-mock" {
		t.Fatalf("auth=%s", calls[0].auth)
	}
	if calls[0].ctType != "application/json" {
		t.Fatalf("content-type=%s", calls[0].ctType)
	}
	if calls[0].body["model"] != "mock-model" || calls[0].body["temperature"] != float64(0) {
		t.Fatalf("body=%v", calls[0].body)
	}
	msgs := calls[0].body["messages"].([]any)
	m0 := msgs[0].(map[string]any)
	if m0["role"] != "system" || !strings.Contains(m0["content"].(string), "STRICT JSON") {
		t.Fatalf("system message wrong: %v", m0)
	}
}

func TestHTTPMockCheckFailsCorrectiveRetry(t *testing.T) {
	var calls []capturedCall
	hc := mockHTTPClient([]string{wrongCheckReply, goodReply}, nil, &calls)
	solver, _ := NewWithHTTPClient("sk", "https://mock.test/v1", hc)
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 1 || len(calls) != 2 {
		t.Fatalf("r=%+v calls=%d", r, len(calls))
	}
	msgs := calls[1].body["messages"].([]any)
	found := false
	for _, m := range msgs {
		if strings.Contains(m.(map[string]any)["content"].(string), "failed verification") {
			found = true
		}
	}
	if !found {
		t.Fatalf("retry message missing corrective text: %v", msgs)
	}
}

func TestHTTPMockInvalidJSONReask(t *testing.T) {
	var calls []capturedCall
	hc := mockHTTPClient([]string{"certainly not json", goodReply}, nil, &calls)
	solver, _ := NewWithHTTPClient("sk", "https://mock.test/v1", hc)
	r, err := solver.Solve("1+1")
	if err != nil || !r.Verified || len(calls) != 2 {
		t.Fatalf("err=%v r=%+v calls=%d", err, r, len(calls))
	}
}

func TestHTTPMock500NoRetry(t *testing.T) {
	var calls []capturedCall
	hc := mockHTTPClient(nil, []int{500}, &calls)
	solver, _ := NewWithHTTPClient("sk", "https://mock.test/v1", hc)
	_, err := solver.Solve("1+1")
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "HTTP_ERROR" || len(calls) != 1 {
		t.Fatalf("err=%v calls=%d", err, len(calls))
	}
}

func TestHTTPMock401(t *testing.T) {
	var calls []capturedCall
	hc := mockHTTPClient(nil, []int{401}, &calls)
	solver, _ := NewWithHTTPClient("sk-bad", "https://mock.test/v1", hc)
	_, err := solver.Solve("1+1")
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "HTTP_ERROR" {
		t.Fatalf("err=%v", err)
	}
}
