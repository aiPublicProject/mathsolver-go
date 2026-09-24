// Package mathsolver is a BYOK AI math solver with execution-based verification (v0.2).
//
// Correctness model (PAL-style): the model never states the answer.
// It returns a small JavaScript-like PROGRAM; this package executes the
// program deterministically and the execution output IS the answer.
// For equations, a CHECK expression ({x} placeholder) must evaluate to 0
// when the computed answer is substituted back into the original equation.
package mathsolver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

const SystemPrompt = "You are a precise math solver.\n" +
	"Reply with STRICT JSON only, no markdown fences, in this exact shape:\n" +
	`{"program": "<string>", "steps": [<string>, ...], "check": "<string>"}` + "\n" +
	"Rules:\n" +
	"- \"program\" is a small JavaScript-like program that computes the final answer.\n" +
	"  One statement per line (or ; separated). Allowed statements:\n" +
	"      let NAME = EXPRESSION\n" +
	"      result = EXPRESSION\n" +
	"  EXPRESSIONs may use numbers, + - * / % ^ ( ), the functions\n" +
	"  abs sqrt sin cos tan ln log exp floor ceil round min max\n" +
	"  (log is base 10, ln is natural), the constants pi and e, and any\n" +
	"  variable defined by an earlier let. The value assigned to \"result\"\n" +
	"  is the answer. Never state the answer as a number in text.\n" +
	"- \"steps\" is an array of short plain-language explanation strings.\n" +
	"- \"check\" is a verification expression containing the placeholder {x}.\n" +
	"  After solving, {x} is replaced by the computed answer and the whole\n" +
	"  expression must evaluate to 0.\n" +
	"  For equations, substitute the answer back into the original equation\n" +
	"  (e.g. 2x+3=11 -> \"2*{x}+3-11\").\n" +
	"  For arithmetic, recompute via a different path and subtract the answer\n" +
	"  (e.g. 15% of 80 -> \"80*15/100-{x}\"). Provide \"check\" whenever possible."

func correctionPrompt(reason string) string {
	return "Your submission failed verification: " + reason +
		". Re-derive the problem carefully and reply again with the same strict JSON shape."
}

// SolverError carries a machine-readable code.
type SolverError struct {
	Code    string
	Message string
}

func (e *SolverError) Error() string { return e.Code + ": " + e.Message }

func errf(code, format string, args ...any) *SolverError {
	return &SolverError{Code: code, Message: fmt.Sprintf(format, args...)}
}

/* ---------------------- expression evaluator ---------------------- */

type token struct {
	kind string // "num", "id", or single-char op
	num  float64
	id   string
}

func tokenize(src string) ([]token, error) {
	var tokens []token
	i := 0
	for i < len(src) {
		c := src[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			i++
			continue
		}
		if c >= '0' && c <= '9' || c == '.' {
			j := i
			for j < len(src) && (src[j] >= '0' && src[j] <= '9' || src[j] == '.') {
				j++
			}
			if j < len(src) && (src[j] == 'e' || src[j] == 'E') {
				k := j + 1
				if k < len(src) && (src[k] == '+' || src[k] == '-') {
					k++
				}
				if k < len(src) && src[k] >= '0' && src[k] <= '9' {
					for k < len(src) && src[k] >= '0' && src[k] <= '9' {
						k++
					}
					j = k
				}
			}
			var v float64
			if _, err := fmt.Sscanf(src[i:j], "%g", &v); err != nil {
				return nil, errf("EXPR_BAD_NUMBER", "bad number %q", src[i:j])
			}
			tokens = append(tokens, token{kind: "num", num: v})
			i = j
			continue
		}
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_' {
			j := i
			for j < len(src) && (src[j] >= 'a' && src[j] <= 'z' || src[j] >= 'A' && src[j] <= 'Z' || src[j] == '_' || src[j] >= '0' && src[j] <= '9') {
				j++
			}
			tokens = append(tokens, token{kind: "id", id: src[i:j]})
			i = j
			continue
		}
		if strings.ContainsRune("+-*/%^(),", rune(c)) {
			tokens = append(tokens, token{kind: string(c)})
			i++
			continue
		}
		return nil, errf("EXPR_BAD_CHAR", "unexpected character %q", string(c))
	}
	return tokens, nil
}

type parser struct {
	tokens []token
	pos    int
	env    map[string]float64 // variable bindings from let-statements
}

func (p *parser) peek() *token {
	if p.pos < len(p.tokens) {
		return &p.tokens[p.pos]
	}
	return nil
}

func (p *parser) eat() (token, error) {
	if p.pos >= len(p.tokens) {
		return token{}, errf("EXPR_SYNTAX", "expected more tokens")
	}
	t := p.tokens[p.pos]
	p.pos++
	return t, nil
}

// EvalExpression evaluates a pure arithmetic expression string (no variables).
func EvalExpression(src string) (float64, error) {
	return EvalExpressionWith(src, nil)
}

// EvalExpressionWith evaluates an arithmetic expression with variable bindings.
// env names are case-sensitive and shadow the pi/e constants.
func EvalExpressionWith(src string, env map[string]float64) (float64, error) {
	if strings.TrimSpace(src) == "" {
		return 0, errf("EXPR_EMPTY", "empty expression")
	}
	tokens, err := tokenize(src)
	if err != nil {
		return 0, err
	}
	p := &parser{tokens: tokens, env: env}
	v, err := p.expr()
	if err != nil {
		return 0, err
	}
	if p.pos != len(tokens) {
		return 0, errf("EXPR_TRAILING", "trailing tokens")
	}
	if math.IsInf(v, 0) || math.IsNaN(v) {
		return 0, errf("EXPR_NON_FINITE", "non-finite result")
	}
	return v, nil
}

func (p *parser) expr() (float64, error) {
	v, err := p.term()
	if err != nil {
		return 0, err
	}
	for {
		t := p.peek()
		if t == nil || (t.kind != "+" && t.kind != "-") {
			return v, nil
		}
		op := t.kind
		if _, err := p.eat(); err != nil {
			return 0, err
		}
		r, err := p.term()
		if err != nil {
			return 0, err
		}
		if op == "+" {
			v += r
		} else {
			v -= r
		}
	}
}

func (p *parser) term() (float64, error) {
	v, err := p.unary()
	if err != nil {
		return 0, err
	}
	for {
		t := p.peek()
		if t == nil || (t.kind != "*" && t.kind != "/" && t.kind != "%") {
			return v, nil
		}
		op := t.kind
		if _, err := p.eat(); err != nil {
			return 0, err
		}
		r, err := p.unary()
		if err != nil {
			return 0, err
		}
		switch op {
		case "*":
			v *= r
		case "/":
			v /= r
		default:
			v = math.Mod(v, r)
		}
	}
}

func (p *parser) unary() (float64, error) {
	t := p.peek()
	if t != nil && t.kind == "-" {
		if _, err := p.eat(); err != nil {
			return 0, err
		}
		v, err := p.unary()
		return -v, err
	}
	if t != nil && t.kind == "+" {
		if _, err := p.eat(); err != nil {
			return 0, err
		}
		return p.unary()
	}
	return p.power()
}

func (p *parser) power() (float64, error) {
	base, err := p.atom()
	if err != nil {
		return 0, err
	}
	t := p.peek()
	if t != nil && t.kind == "^" {
		if _, err := p.eat(); err != nil {
			return 0, err
		}
		exp, err := p.unary() // right associative
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

func (p *parser) atom() (float64, error) {
	t, err := p.eat()
	if err != nil {
		return 0, err
	}
	switch t.kind {
	case "num":
		return t.num, nil
	case "id":
		if p.env != nil {
			if v, ok := p.env[t.id]; ok {
				return v, nil
			}
		}
		name := strings.ToLower(t.id)
		if nxt := p.peek(); nxt != nil && nxt.kind == "(" {
			if _, err := p.eat(); err != nil {
				return 0, err
			}
			var args []float64
			first, err := p.expr()
			if err != nil {
				return 0, err
			}
			args = append(args, first)
			for nxt := p.peek(); nxt != nil && nxt.kind == ","; {
				if _, err := p.eat(); err != nil {
					return 0, err
				}
				a, err := p.expr()
				if err != nil {
					return 0, err
				}
				args = append(args, a)
				nxt = p.peek()
			}
			if end, err := p.eat(); err != nil || end.kind != ")" {
				return 0, errf("EXPR_SYNTAX", "expected )")
			}
			return applyFn(name, args)
		}
		switch name {
		case "pi":
			return math.Pi, nil
		case "e":
			return math.E, nil
		}
		return 0, errf("EXPR_UNKNOWN_ID", "unknown identifier %q", name)
	case "(":
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if end, err := p.eat(); err != nil || end.kind != ")" {
			return 0, errf("EXPR_SYNTAX", "expected )")
		}
		return v, nil
	}
	return 0, errf("EXPR_SYNTAX", "unexpected token %q", t.kind)
}

func applyFn(name string, args []float64) (float64, error) {
	arg := func(i int) float64 {
		if i < len(args) {
			return args[i]
		}
		return math.NaN()
	}
	switch name {
	case "abs":
		return math.Abs(arg(0)), nil
	case "sqrt":
		return math.Sqrt(arg(0)), nil
	case "sin":
		return math.Sin(arg(0)), nil
	case "cos":
		return math.Cos(arg(0)), nil
	case "tan":
		return math.Tan(arg(0)), nil
	case "ln":
		return math.Log(arg(0)), nil
	case "log":
		return math.Log10(arg(0)), nil
	case "exp":
		return math.Exp(arg(0)), nil
	case "floor":
		return math.Floor(arg(0)), nil
	case "ceil":
		return math.Ceil(arg(0)), nil
	case "round":
		return math.Round(arg(0)), nil
	case "min":
		m := math.Inf(1)
		for _, a := range args {
			m = math.Min(m, a)
		}
		return m, nil
	case "max":
		m := math.Inf(-1)
		for _, a := range args {
			m = math.Max(m, a)
		}
		return m, nil
	}
	return 0, errf("EXPR_UNKNOWN_FUNC", "unknown function %q", name)
}

/* ---------------------- program interpreter ---------------------- */

var (
	letStmt     = regexp.MustCompile(`^let\s+([a-zA-Z_]\w*)\s*=\s*(.+)$`)
	assignStmt  = regexp.MustCompile(`^([a-zA-Z_]\w*)\s*=\s*(.+)$`)
	checkHolder = regexp.MustCompile(`(?i)\{\s*x\s*\}`)
)

// RunProgram executes a model-generated program. Statements (one per line or
// ; separated): let NAME = EXPR | NAME = EXPR | bare EXPR. The answer is the
// value of `result`, else the last bare expression. The model never states
// the answer as a number — execution output IS the answer.
func RunProgram(src string) (float64, error) {
	if strings.TrimSpace(src) == "" {
		return 0, errf("PROGRAM_EMPTY", "empty program")
	}
	env := map[string]float64{}
	resultDefined := false
	lastDefined := false
	var lastValue float64
	for _, raw := range strings.FieldsFunc(src, func(r rune) bool { return r == '\n' || r == ';' }) {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if m := letStmt.FindStringSubmatch(line); m != nil {
			v, err := EvalExpressionWith(m[2], env)
			if err != nil {
				return 0, err
			}
			env[m[1]] = v
			if m[1] == "result" {
				resultDefined = true
			}
			continue
		}
		if m := assignStmt.FindStringSubmatch(line); m != nil {
			v, err := EvalExpressionWith(m[2], env)
			if err != nil {
				return 0, err
			}
			env[m[1]] = v
			if m[1] == "result" {
				resultDefined = true
			}
			continue
		}
		v, err := EvalExpressionWith(line, env)
		if err != nil {
			return 0, err
		}
		lastValue, lastDefined = v, true
	}
	if resultDefined {
		return env["result"], nil
	}
	if lastDefined {
		return lastValue, nil
	}
	return 0, errf("PROGRAM_NO_RESULT", "program produced no result")
}

// RunCheck substitutes the computed answer into a check expression ({x}
// placeholder) and evaluates it. Passes when the value is ~0 (scaled
// tolerance). Returns (value, passed, err).
func RunCheck(checkSrc string, answer float64) (float64, bool, error) {
	substituted := checkHolder.ReplaceAllString(checkSrc, "("+strconv.FormatFloat(answer, 'g', -1, 64)+")")
	value, err := EvalExpression(substituted)
	if err != nil {
		return 0, false, err
	}
	return value, math.Abs(value) <= 1e-6*math.Max(1, math.Abs(answer)), nil
}

/* ---------------------- solve ---------------------- */

type Parsed struct {
	Program string
	Steps   []string
	Check   string // "" = none provided
}

type Result struct {
	// Answer is the output of executing the model's program locally.
	Answer float64
	Steps  []string
	// Program is the executed program (the answer's provenance).
	Program string
	// Check is the verification expression ("" = none provided).
	Check string
	// CheckValue is the evaluated check expression (nil when no check).
	CheckValue *float64
	// Verified is true only when the check expression evaluated to ~0.
	Verified bool
	Retries  int
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type replyShape struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

type modelReply struct {
	Program string   `json:"program"`
	Steps   []string `json:"steps"`
	Check   string   `json:"check"`
}

// Transport fetches a model reply: given (url, body, apiKey) returns message content.
type Transport func(url string, body []byte, apiKey string) (string, error)

func DefaultTransport(url string, body []byte, apiKey string) (string, error) {
	return transportWith(http.DefaultClient)(url, body, apiKey)
}

// transportWith builds a Transport on top of a custom *http.Client — inject
// timeouts, proxies, or (in tests) a mock RoundTripper so the default
// transport's real code path runs without sockets.
func transportWith(hc *http.Client) Transport {
	return func(url string, body []byte, apiKey string) (string, error) {
		req, err := http.NewRequest("POST", url, bytes.NewReader(body))
		if err != nil {
			return "", errf("HTTP_ERROR", "%v", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)
		res, err := hc.Do(req)
		if err != nil {
			return "", errf("HTTP_ERROR", "%v", err)
		}
		defer res.Body.Close()
		if res.StatusCode >= 300 {
			return "", errf("HTTP_ERROR", "API responded %d", res.StatusCode)
		}
		var shaped replyShape
		if err := json.NewDecoder(res.Body).Decode(&shaped); err != nil {
			return "", errf("HTTP_ERROR", "invalid JSON from API")
		}
		if len(shaped.Choices) == 0 {
			return "", errf("HTTP_ERROR", "API response missing choices")
		}
		if shaped.Choices[0].Message.Content == "" {
			return "", errf("HTTP_ERROR", "API response missing message content")
		}
		return shaped.Choices[0].Message.Content, nil
	}
}

func parseModelReply(text string) (Parsed, error) {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end <= start {
		return Parsed{}, errf("INVALID_JSON", "no JSON object in reply")
	}
	var m modelReply
	if err := json.Unmarshal([]byte(text[start:end+1]), &m); err != nil {
		return Parsed{}, errf("INVALID_JSON", "reply was not valid JSON")
	}
	if m.Program == "" {
		return Parsed{}, errf("INVALID_JSON", "missing program")
	}
	return Parsed{Program: m.Program, Steps: m.Steps, Check: strings.TrimSpace(m.Check)}, nil
}

// Client is a BYOK client for an OpenAI-compatible endpoint.
// Instantiate once with New/NewWithTransport, then call Solve for each problem.
type Client struct {
	// APIKey is the user's own key (BYOK).
	APIKey string
	// BaseURL is any OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1
	BaseURL string
	// Model defaults to gpt-4o-mini; set after New if needed.
	Model string

	transport Transport
}

// New creates a client with the built-in HTTP transport.
func New(apiKey, baseURL string) (*Client, error) {
	return NewWithTransport(apiKey, baseURL, DefaultTransport)
}

// NewWithTransport creates a client with an injected transport (url, body, apiKey) -> reply.
func NewWithTransport(apiKey, baseURL string, transport Transport) (*Client, error) {
	if apiKey == "" {
		return nil, errf("NO_API_KEY", "apiKey is required (BYOK)")
	}
	base := strings.TrimRight(baseURL, "/")
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		return nil, errf("BAD_BASE_URL", "baseURL must be an http(s) URL, e.g. https://api.deepseek.com/v1")
	}
	return &Client{APIKey: apiKey, BaseURL: base, Model: "gpt-4o-mini", transport: transport}, nil
}

// NewWithHTTPClient creates a client whose built-in transport uses hc.
// Inject a custom *http.Client for timeouts/proxies, or a mock RoundTripper in tests.
func NewWithHTTPClient(apiKey, baseURL string, hc *http.Client) (*Client, error) {
	if hc == nil {
		hc = http.DefaultClient
	}
	return NewWithTransport(apiKey, baseURL, transportWith(hc))
}

// attempt executes a parsed reply. Never fails; reports ok=false with the error.
type attemptOut struct {
	ok         bool
	answer     float64
	checkValue *float64
	verified   bool
	err        *SolverError
}

// Solve solves a math problem. Answer is the output of executing the model's
// program; Verified is true only when the check expression ({x} substituted
// with the answer) evaluated to ~0.
func (c *Client) Solve(problem string) (Result, error) {
	if strings.TrimSpace(problem) == "" {
		return Result{}, errf("NO_PROBLEM", "problem must be non-empty")
	}
	url := c.BaseURL + "/chat/completions"
	messages := []message{
		{Role: "system", Content: SystemPrompt},
		{Role: "user", Content: problem},
	}
	type payload struct {
		Model       string    `json:"model"`
		Messages    []message `json:"messages"`
		Temperature float64   `json:"temperature"`
	}
	call := func() (string, error) {
		body, _ := json.Marshal(payload{Model: c.Model, Messages: messages, Temperature: 0})
		return c.transport(url, body, c.APIKey)
	}

	content, err := call()
	if err != nil {
		return Result{}, err
	}
	parsed, perr := parseModelReply(content)
	if perr != nil {
		if se, ok := perr.(*SolverError); !ok || se.Code != "INVALID_JSON" {
			return Result{}, perr
		}
		messages = append(messages, message{"assistant", "invalid JSON"},
			message{"user", "Your reply was not valid JSON. Reply again with the exact strict JSON shape."})
		content, err = call()
		if err != nil {
			return Result{}, err
		}
		parsed, perr = parseModelReply(content)
		if perr != nil {
			return Result{}, perr
		}
	}

	attempt := func(p Parsed) attemptOut {
		answer, err := RunProgram(p.Program)
		if err != nil {
			if se, ok := err.(*SolverError); ok {
				return attemptOut{err: se}
			}
			return attemptOut{err: errf("PROGRAM_ERROR", "%v", err)}
		}
		out := attemptOut{ok: true, answer: answer}
		if p.Check != "" {
			v, passed, cerr := RunCheck(p.Check, answer)
			if cerr != nil {
				if se, ok := cerr.(*SolverError); ok {
					return attemptOut{err: se}
				}
				return attemptOut{err: errf("EXPR_ERROR", "%v", cerr)}
			}
			cv := v
			out.checkValue = &cv
			out.verified = passed
		}
		return out
	}

	outcome := attempt(parsed)
	retries := 0
	if !outcome.ok || !outcome.verified {
		retries = 1
		var reason string
		if !outcome.ok {
			reason = fmt.Sprintf("program failed to execute (%s: %s)", outcome.err.Code, outcome.err.Message)
		} else {
			reason = fmt.Sprintf("check evaluated to %v instead of 0", *outcome.checkValue)
		}
		raw, _ := json.Marshal(parsed)
		messages = append(messages, message{"assistant", string(raw)},
			message{"user", correctionPrompt(reason)})
		content, err = call()
		if err != nil {
			return Result{}, err
		}
		secondParsed, serr := parseModelReply(content)
		if serr != nil {
			return Result{}, serr
		}
		second := attempt(secondParsed)
		if !second.ok {
			return Result{}, second.err // PROGRAM_* error persisted after retry
		}
		parsed, outcome = secondParsed, second
	}

	return Result{Answer: outcome.answer, Steps: parsed.Steps, Program: parsed.Program,
		Check: parsed.Check, CheckValue: outcome.checkValue, Verified: outcome.verified,
		Retries: retries}, nil
}
