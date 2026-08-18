// chain.go - context chaining: when a session approaches the context cap,
// summarize the old context with an SLM and continue on a fresh context
// seeded with [summary + recent turns].
//
// Why orchestration, not C: llama.cpp's native answer is a KV-cache SHIFT
// (drop the middle, keep the head+tail) - it loses the conversation silently.
// Chaining keeps the full archive on disk (storage is cheap) and compresses
// the LIVE context with a summary, so nothing is lost and the model always
// works on a fresh, complete context.
//
// State: per-session live-token position (in-memory; the ledger records chain
// events; sessions/<id>.jsonl archives every turn's content).
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

const defaultChainCap int = 3072 // 75% of the 4096 backend context

var posMu sync.Mutex
var sessionPos map[string]int = make(map[string]int)

type chainTurn struct {
	Ts      int64  `json:"ts"`
	Role    string `json:"role"`
	Content string `json:"content"`
}

func estTokens(s string) int {
	return len(s)/4 + 8
}

func sessionContentPath(session string) string {
	return filepath.Join(fleetDir, "sessions", session+".jsonl")
}

// appendContent: archive one turn (storage is cheap; this is the archive).
func appendContent(session string, role string, content string) {
	dir := filepath.Dir(sessionContentPath(session))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return
	}
	f, err := os.OpenFile(sessionContentPath(session), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	data, _ := json.Marshal(chainTurn{Ts: nowMs(), Role: role, Content: content})
	_, _ = f.Write(append(data, '\n'))
}

// readContent: last N archived turns for a session.
func readContent(session string, lastN int) []chainTurn {
	f, err := os.Open(sessionContentPath(session))
	if err != nil {
		return nil
	}
	defer f.Close()
	var all []chainTurn
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var t chainTurn
		if json.Unmarshal([]byte(sc.Text()), &t) == nil {
			all = append(all, t)
		}
	}
	if lastN > 0 && len(all) > lastN {
		all = all[len(all)-lastN:]
	}
	return all
}

// livePosition: sum of estimated tokens in the archived session (the live
// context is roughly the whole archive until a chain resets it).
func livePosition(session string) int {
	posMu.Lock()
	defer posMu.Unlock()
	if p, ok := sessionPos[session]; ok {
		return p
	}
	var p int = 0
	for _, t := range readContent(session, 0) {
		p += estTokens(t.Content)
	}
	sessionPos[session] = p
	return p
}

func addPosition(session string, n int) {
	posMu.Lock()
	sessionPos[session] = livePositionLocked(session) + n
	posMu.Unlock()
}

func livePositionLocked(session string) int {
	if p, ok := sessionPos[session]; ok {
		return p
	}
	var p int = 0
	for _, t := range readContent(session, 0) {
		p += estTokens(t.Content)
	}
	sessionPos[session] = p
	return p
}

func resetPosition(session string, n int) {
	posMu.Lock()
	sessionPos[session] = n
	posMu.Unlock()
}

// summarizeVia: ask the generalist SLM to compress the history.
func summarizeVia(cfg *serveConfig, history string) string {
	prompt := "Summarize this conversation so a fresh model can continue without " +
		"losing anything. Keep all concrete requirements, code decisions, and the " +
		"user's goal. Output only the summary:\n\n" + history
	body, _ := json.Marshal(map[string]any{"prompt": prompt, "n_predict": 250,
		"temperature": 0.2})
	resp, err := httpPostJSON(cfg.backends["general"]+"/completion", body)
	if err != nil {
		return ""
	}
	var cr completionResp
	if json.Unmarshal(resp, &cr) != nil {
		return ""
	}
	return strings.TrimSpace(cr.Content)
}

// chainContext: if the session is about to exceed the cap, summarize the
// archive and return the new live prefix [summary + recent turns].
// Returns (prefix, true) when a chain happened.
func chainContext(cfg *serveConfig, session string, newPrompt string, cap int) (string, bool) {
	pos := livePosition(session)
	if pos+estTokens(newPrompt) <= cap {
		fmt.Fprintf(os.Stderr, "[chain] %s: pos %d + %d <= cap %d (no chain)\n",
			session, pos, estTokens(newPrompt), cap)
		return "", false
	}
	turns := readContent(session, 0)
	if len(turns) < 4 {
		fmt.Fprintf(os.Stderr, "[chain] %s: pos %d > cap %d but only %d turns\n",
			session, pos, cap, len(turns))
		return "", false // not enough history to summarize meaningfully
	}
	fmt.Fprintf(os.Stderr, "[chain] %s: CHAINING pos %d cap %d turns %d\n",
		session, pos, cap, len(turns))
	// history = all but the last 2 turns (which stay verbatim in the tail)
	var hist strings.Builder
	for i := 0; i < len(turns)-2; i++ {
		t := turns[i]
		hist.WriteString(t.Role + ": " + t.Content + "\n")
		if hist.Len() > 12000 {
			break
		}
	}
	summary := summarizeVia(cfg, hist.String())
	if summary == "" {
		return "", false
	}
	var tail strings.Builder
	for i := len(turns) - 2; i < len(turns); i++ {
		t := turns[i]
		tail.WriteString(t.Role + ": " + t.Content + "\n")
	}
	prefix := "CONTEXT SUMMARY (prior conversation):\n" + summary + "\n\n" +
		"RECENT EXCHANGE:\n" + tail.String()
	resetPosition(session, estTokens(prefix))
	// archive the chain event so the ledger and future reads see it
	appendContent(session, "system", "CONTEXT CHAINED at pos "+fmt.Sprint(pos)+": "+summary)
	return prefix, true
}

func httpPostJSON(url string, body []byte) ([]byte, error) {
	req, err := http.NewRequest("POST", url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// jsonString: JSON-encode a string (used when rebuilding forwarded bodies).
func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
