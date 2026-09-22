package mathsolver

import (
	"encoding/json"
	"errors"
	"testing"
)

const goodReply = `{"answer": 4, "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "verification": {"expression": "(11-3)/2"}}`
const wrongReply = `{"answer": 4, "steps": ["..."], "verification": {"expression": "(11-3)/3"}}`

func wrap(content string) string {
	b, _ := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"content": content}}},
	})
	return string(b)
}

func TestEvaluatorPrecedence(t *testing.T) {
	cases := map[string]float64{
		"2*3+4": 10, "2+3*4": 14, "(2+3)*4": 20,
		"2^3^2": 512, "-3^2": -9, "10%3": 1,
	}
	for src, want := range cases {
		got, err := EvalExpression(src)
		if err != nil {
			t.Fatalf("%s: %v", src, err)
		}
		if diff := got - want; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("%s = %v, want %v", src, got, want)
		}
	}
}

func TestEvaluatorFunctions(t *testing.T) {
	if got, _ := EvalExpression("sqrt(16)"); got != 4 {
		t.Fatalf("sqrt(16)=%v", got)
	}
	if got, _ := EvalExpression("min(3,5)"); got != 3 {
		t.Fatalf("min=%v", got)
	}
	for _, bad := range []string{"fmt.Println(1)", "1+2)", "foo(1)", ""} {
		if _, err := EvalExpression(bad); err == nil {
			t.Fatalf("expected error for %q", bad)
		}
	}
}

func TestSolveVerifiedFirstTry(t *testing.T) {
	calls := 0
	var gotURL, gotKey string
	tr := func(url string, body []byte, key string) (string, error) {
		calls++
		gotURL, gotKey = url, key
		return wrap(goodReply), nil
	}
	r, err := Solve("2x + 3 = 11, solve for x", Options{APIKey: "sk-test", Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 0 || r.Evaluated == nil || *r.Evaluated != 4 {
		t.Fatalf("bad result: %+v", r)
	}
	if calls != 1 {
		t.Fatalf("calls=%d", calls)
	}
	if gotURL[len(gotURL)-17:] != "/chat/completions" || gotKey != "sk-test" {
		t.Fatalf("url=%s key=%s", gotURL, gotKey)
	}
}

func TestSolveRetryRecovers(t *testing.T) {
	n := 0
	tr := func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return wrap(wrongReply), nil
		}
		return wrap(goodReply), nil
	}
	r, err := Solve("2x+3=11", Options{APIKey: "sk", Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 1 {
		t.Fatalf("bad result: %+v", r)
	}
}

func TestSolveInvalidJSONThenOK(t *testing.T) {
	n := 0
	tr := func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return "no json", nil
		}
		return wrap(goodReply), nil
	}
	r, err := Solve("1+1", Options{APIKey: "sk", Transport: tr})
	if err != nil || !r.Verified {
		t.Fatalf("err=%v r=%+v", err, r)
	}
}

func TestSolveInvalidJSONTwice(t *testing.T) {
	tr := func(string, []byte, string) (string, error) { return "nothing", nil }
	_, err := Solve("1+1", Options{APIKey: "sk", Transport: tr})
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "INVALID_JSON" {
		t.Fatalf("err=%v", err)
	}
}

func TestSolveNoAPIKey(t *testing.T) {
	_, err := Solve("1+1", Options{})
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "NO_API_KEY" {
		t.Fatalf("err=%v", err)
	}
}

func TestSolveHTTPErrorNoRetry(t *testing.T) {
	calls := 0
	tr := func(string, []byte, string) (string, error) {
		calls++
		return "", errf("HTTP_ERROR", "401")
	}
	_, err := Solve("1+1", Options{APIKey: "sk", Transport: tr})
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "HTTP_ERROR" || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestSolveStillWrongUnverified(t *testing.T) {
	tr := func(string, []byte, string) (string, error) { return wrap(wrongReply), nil }
	r, err := Solve("2x+3=11", Options{APIKey: "sk", Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verified || r.Retries != 1 {
		t.Fatalf("r=%+v", r)
	}
}
