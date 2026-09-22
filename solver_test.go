package mathsolver

import (
	"errors"
	"testing"
)

const goodReply = `{"answer": 4, "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "verification": {"expression": "(11-3)/2"}}`
const wrongReply = `{"answer": 4, "steps": ["..."], "verification": {"expression": "(11-3)/3"}}`

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

func TestNewValidatesCredentials(t *testing.T) {
	if _, err := New("", "https://api.x"); err == nil || err.(*SolverError).Code != "NO_API_KEY" {
		t.Fatalf("want NO_API_KEY, got %v", err)
	}
	if _, err := New("sk", "not-a-url"); err == nil || err.(*SolverError).Code != "BAD_BASE_URL" {
		t.Fatalf("want BAD_BASE_URL, got %v", err)
	}
}

func TestSolveVerifiedFirstTry(t *testing.T) {
	calls := 0
	var gotURL, gotKey string
	solver, err := NewWithTransport("sk-test", "https://api.deepseek.com/v1", func(url string, body []byte, key string) (string, error) {
		calls++
		gotURL, gotKey = url, key
		return goodReply, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	solver.Model = "deepseek-chat"
	r, err := solver.Solve("2x + 3 = 11, solve for x")
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 0 || r.Evaluated == nil || *r.Evaluated != 4 {
		t.Fatalf("bad result: %+v", r)
	}
	if calls != 1 {
		t.Fatalf("calls=%d", calls)
	}
	if gotURL != "https://api.deepseek.com/v1/chat/completions" || gotKey != "sk-test" {
		t.Fatalf("url=%s key=%s", gotURL, gotKey)
	}
}

func TestSolveRetryRecovers(t *testing.T) {
	n := 0
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return wrongReply, nil
		}
		return goodReply, nil
	})
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 1 {
		t.Fatalf("bad result: %+v", r)
	}
}

func TestSolveInvalidJSONThenOK(t *testing.T) {
	n := 0
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return "no json", nil
		}
		return goodReply, nil
	})
	r, err := solver.Solve("1+1")
	if err != nil || !r.Verified {
		t.Fatalf("err=%v r=%+v", err, r)
	}
}

func TestSolveInvalidJSONTwice(t *testing.T) {
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) { return "nothing", nil })
	_, err := solver.Solve("1+1")
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "INVALID_JSON" {
		t.Fatalf("err=%v", err)
	}
}

func TestSolveHTTPErrorNoRetry(t *testing.T) {
	calls := 0
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		calls++
		return "", errf("HTTP_ERROR", "401")
	})
	_, err := solver.Solve("1+1")
	var se *SolverError
	if !errors.As(err, &se) || se.Code != "HTTP_ERROR" || calls != 1 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestSolveStillWrongUnverified(t *testing.T) {
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) { return wrongReply, nil })
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if r.Verified || r.Retries != 1 {
		t.Fatalf("r=%+v", r)
	}
}
