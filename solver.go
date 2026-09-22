// Package mathsolver is a BYOK AI math solver with independent verification.
// An answer is only Verified=true when the model's verification expression
// (pure arithmetic) is evaluated locally and matches the answer.
package mathsolver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
)

const SystemPrompt = "You are a precise math solver.\n" +
	"Reply with STRICT JSON only, no markdown fences, in this exact shape:\n" +
	`{"answer": <number>, "steps": [<string>, ...], "verification": {"expression": "<string>"}}` + "\n" +
	"Rules:\n" +
	"- \"answer\" must be a single number (the final result).\n" +
	"- \"steps\" must be an array of short plain-language explanation strings.\n" +
	"- \"verification.expression\" must be a pure arithmetic expression that\n" +
	"  evaluates to the answer. Allowed: numbers, + - * / % ^ ( ), and the\n" +
	"  functions abs sqrt sin cos tan ln log exp floor ceil round min max\n" +
	"  (log is base 10, ln is natural), and the constants pi and e.\n" +
	"- The expression must recompute the answer independently."

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

// EvalExpression evaluates a pure arithmetic expression string.
func EvalExpression(src string) (float64, error) {
	if strings.TrimSpace(src) == "" {
		return 0, errf("EXPR_EMPTY", "empty expression")
	}
	tokens, err := tokenize(src)
	if err != nil {
		return 0, err
	}
	p := &parser{tokens: tokens}
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

func numericallyEqual(a, b float64) bool {
	return math.Abs(a-b) <= 1e-6*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

/* ---------------------- solve ---------------------- */

type Parsed struct {
	Answer     float64
	Steps      []string
	Expression string
}

type Result struct {
	Answer     float64
	Steps      []string
	Expression string
	Evaluated  *float64
	Verified   bool
	Retries    int
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
	Answer       float64  `json:"answer"`
	Steps        []string `json:"steps"`
	Verification struct {
		Expression string `json:"expression"`
	} `json:"verification"`
}

// Transport fetches a model reply: given (url, body, apiKey) returns message content.
type Transport func(url string, body []byte, apiKey string) (string, error)

func DefaultTransport(url string, body []byte, apiKey string) (string, error) {
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return "", errf("HTTP_ERROR", "%v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	res, err := http.DefaultClient.Do(req)
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
	return shaped.Choices[0].Message.Content, nil
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
	if m.Verification.Expression == "" {
		return Parsed{}, errf("INVALID_JSON", "missing verification.expression")
	}
	return Parsed{Answer: m.Answer, Steps: m.Steps, Expression: m.Verification.Expression}, nil
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

// Solve solves a math problem. Verified is true only when the model's
// verification expression independently re-evaluates to the answer.
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

	var parsed Parsed
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

	evaluate := func(p Parsed) (*float64, bool) {
		ev, err := EvalExpression(p.Expression)
		if err != nil {
			return nil, false
		}
		return &ev, numericallyEqual(ev, p.Answer)
	}

	evaluated, verified := evaluate(parsed)
	retries := 0
	if !verified {
		retries = 1
		raw, _ := json.Marshal(parsed)
		messages = append(messages, message{"assistant", string(raw)},
			message{"user", fmt.Sprintf("Your verification expression evaluated to %v, which does not match your answer %v. Re-derive the problem carefully and reply again with the same strict JSON shape.", deref(evaluated), parsed.Answer)})
		if content, err = call(); err == nil {
			if second, serr := parseModelReply(content); serr == nil {
				ev2, ok2 := evaluate(second)
				if ev2 != nil {
					evaluated = ev2
				}
				if ok2 {
					parsed, verified = second, true
				}
			}
		}
	}

	return Result{Answer: parsed.Answer, Steps: parsed.Steps, Expression: parsed.Expression,
		Evaluated: evaluated, Verified: verified, Retries: retries}, nil
}

func deref(p *float64) any {
	if p == nil {
		return "an error"
	}
	return *p
}
