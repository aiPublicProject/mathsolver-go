package mathsolver

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

// v0.2 protocol fixtures: the model returns program/steps/check — never an answer.
const goodReply = `{"program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x", "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"], "check": "2*{x} + 3 - 11"}`
const noCheckReply = `{"program": "result = 0.15 * 80", "steps": ["Compute 15% of 80"]}`
const wrongCheckReply = `{"program": "let d = 11 - 3;\nresult = d / 2", "steps": ["..."], "check": "2*{x} + 3 - 12"}`
const brokenProgramReply = `{"program": "result = undefinedvar + 1", "steps": []}`

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

func TestEvalExpressionEnv(t *testing.T) {
	if got, err := EvalExpressionWith("d / 2", map[string]float64{"d": 8}); err != nil || got != 4 {
		t.Fatalf("d/2=%v err=%v", got, err)
	}
	if got, err := EvalExpressionWith("x + y", map[string]float64{"x": 1.5, "y": 2.5}); err != nil || got != 4 {
		t.Fatalf("x+y=%v err=%v", got, err)
	}
	if _, err := EvalExpression("d"); err == nil { // no env: undefined var still errors
		t.Fatal("expected error for undefined d")
	}
	if got, err := EvalExpressionWith("pi", map[string]float64{"pi": 3}); err != nil || got != 3 {
		t.Fatalf("env should shadow constant, got %v err=%v", got, err)
	}
}

func TestRunProgram(t *testing.T) {
	if got, err := RunProgram("let d = 11 - 3;\nlet x = d / 2;\nresult = x"); err != nil || got != 4 {
		t.Fatalf("let+result=%v err=%v", got, err)
	}
	if got, err := RunProgram("let a = 3; let b = 4; a * b"); err != nil || got != 12 {
		t.Fatalf("semicolons+bare=%v err=%v", got, err)
	}
	if got, err := RunProgram("0.15 * 80"); err != nil || got != 12 {
		t.Fatalf("bare=%v err=%v", got, err)
	}
	for _, bad := range []string{"result = undefinedvar + 1", "", "let a = 1; let b = 2"} {
		if _, err := RunProgram(bad); err == nil {
			t.Fatalf("expected error for %q", bad)
		}
	}
}

func TestRunCheck(t *testing.T) {
	v, passed, err := RunCheck("2*{x} + 3 - 11", 4)
	if err != nil || !passed || v != 0 {
		t.Fatalf("pass case: v=%v passed=%v err=%v", v, passed, err)
	}
	v, passed, err = RunCheck("2*{x} + 3 - 12", 4)
	if err != nil || passed || v != -1 {
		t.Fatalf("fail case: v=%v passed=%v err=%v", v, passed, err)
	}
	_, passed, err = RunCheck("80*15/100 - {x}", 12)
	if err != nil || !passed {
		t.Fatalf("recompute case: passed=%v err=%v", passed, err)
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

func TestSolveAnswerFromExecutionFirstTry(t *testing.T) {
	calls := 0
	var gotURL, gotKey string
	var gotBody map[string]any
	solver, err := NewWithTransport("sk-test", "https://api.deepseek.com/v1", func(url string, body []byte, key string) (string, error) {
		calls++
		gotURL, gotKey = url, key
		if json.Unmarshal(body, &gotBody) != nil {
			t.Fatal("bad request body")
		}
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
	// 答案=执行产物(4), 代回检验=0; 模型 JSON 里没有 answer 字段
	if !r.Verified || r.Retries != 0 || r.CheckValue == nil || *r.CheckValue != 0 || r.Answer != 4 {
		t.Fatalf("bad result: %+v", r)
	}
	if calls != 1 {
		t.Fatalf("calls=%d", calls)
	}
	if gotURL != "https://api.deepseek.com/v1/chat/completions" || gotKey != "sk-test" {
		t.Fatalf("url=%s key=%s", gotURL, gotKey)
	}
	if gotBody["model"] != "deepseek-chat" || gotBody["temperature"] != float64(0) {
		t.Fatalf("body=%v", gotBody)
	}
	var raw map[string]any
	if json.Unmarshal([]byte(goodReply), &raw) != nil {
		t.Fatal("fixture not JSON")
	}
	if _, has := raw["answer"]; has {
		t.Fatal("protocol violation: model JSON contains an answer field")
	}
}

func TestSolveNoCheckUnverified(t *testing.T) {
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		return noCheckReply, nil
	})
	r, err := solver.Solve("15% of 80")
	if err != nil {
		t.Fatal(err)
	}
	if r.Answer != 12 || r.Verified || r.Check != "" || r.CheckValue != nil {
		t.Fatalf("r=%+v", r)
	}
}

func TestSolveCheckFailRetryRecovers(t *testing.T) {
	n := 0
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return wrongCheckReply, nil
		}
		return goodReply, nil
	})
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Retries != 1 || r.Answer != 4 {
		t.Fatalf("r=%+v", r)
	}
}

func TestSolveProgramErrorRetryRecovers(t *testing.T) {
	n := 0
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		n++
		if n == 1 {
			return brokenProgramReply, nil
		}
		return goodReply, nil
	})
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if !r.Verified || r.Answer != 4 {
		t.Fatalf("r=%+v", r)
	}
}

func TestSolveProgramErrorPersists(t *testing.T) {
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) {
		return brokenProgramReply, nil
	})
	_, err := solver.Solve("2x+3=11")
	var se *SolverError
	if !errors.As(err, &se) {
		t.Fatalf("want SolverError, got %v", err)
	}
	if len(se.Code) < 6 || (se.Code[:8] != "PROGRAM_" && se.Code[:5] != "EXPR_") {
		t.Fatalf("want PROGRAM_*/EXPR_* code, got %s", se.Code)
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

func TestSolveCheckStillFailsUnverified(t *testing.T) {
	solver, _ := NewWithTransport("sk", "https://api.x", func(string, []byte, string) (string, error) { return wrongCheckReply, nil })
	r, err := solver.Solve("2x+3=11")
	if err != nil {
		t.Fatal(err)
	}
	if r.Answer != 4 || r.Verified || r.Retries != 1 { // 程序执行结果仍在, 只是检验不过
		t.Fatalf("r=%+v", r)
	}
}

func TestSmokeRealAPI(t *testing.T) {
	key := os.Getenv("SMOKE_API_KEY")
	if key == "" {
		t.Skip("smoke: set SMOKE_API_KEY to run")
	}
	base := os.Getenv("SMOKE_BASE_URL")
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	solver, err := New(key, base)
	if err != nil {
		t.Fatal(err)
	}
	if m := os.Getenv("SMOKE_MODEL"); m != "" {
		solver.Model = m
	}
	r, err := solver.Solve("2x + 3 = 11, solve for x")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("smoke: answer=%v verified=%v retries=%d", r.Answer, r.Verified, r.Retries)
	if !r.Verified || r.Answer != 4 {
		t.Fatalf("bad result: %+v", r)
	}
}
