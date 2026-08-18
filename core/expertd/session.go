// session.go - session ledger: which SLM served which turn, scope events,
// and the out-of-scope escalation protocol.
//
// Escalation protocol:
//   A session is owned by one SLM (per language/region). When the router
//   detects a "pocket of knowledge" the active SLM does not own - via the
//   tier-2 sub-index on the input (deterministic) or on the SLM's own output
//   (drift detection) - it returns:
//
//     destination hit out of scope - shall we delegate an SLM for <LABEL> (<LANG>)?
//
//   The ledger records every turn: session id, SLM, language, latency, and
//   any scope event, so "which SLMs are currently being used for such
//   session/prompt" is a query, not an OpenMP fork.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

type turn struct {
	Session    string  `json:"session"`
	Ts         int64   `json:"ts"`
	SLM        string  `json:"slm"`
	Lang       string  `json:"lang"`
	Tokens     int     `json:"tokens"`
	WallMs     int64   `json:"wall_ms"`
	ScopeEvent string  `json:"scope_event,omitempty"`
	Escalation string  `json:"escalation,omitempty"`
	Margin     float64 `json:"margin,omitempty"`
	Pending    string  `json:"pending,omitempty"` // lang awaiting user consent
}

var sessionsPath string = fleetDir + "/sessions.jsonl" // overridden in applyEnv()

// ---- in-memory hot state ------------------------------------------------
// lastTurns: per-session latest turn (the router reads it on EVERY request;
// the ledger file grows forever, so it must never be re-read per request).
// useCounts: per-backend-URL turn counts for the /slms roster (polled
// every 500ms by the OMP status line).
var stateMu sync.Mutex
var lastTurns map[string]turn = map[string]turn{}
var useCounts map[string]int = map[string]int{}

func appendTurn(t turn) {
	data, err := json.Marshal(t)
	if err != nil {
		return
	}
	f, err := os.OpenFile(sessionsPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return
	}
	_, _ = f.Write(append(data, '\n'))
	_ = f.Close()
	// hot state
	stateMu.Lock()
	lastTurns[t.Session] = t
	for _, prefix := range []string{"backend(", "autostart("} {
		if strings.HasPrefix(t.SLM, prefix) && strings.HasSuffix(t.SLM, ")") {
			useCounts[strings.TrimSuffix(strings.TrimPrefix(t.SLM, prefix), ")")]++
		}
	}
	stateMu.Unlock()
}

// warmState: one cold read of the ledger at startup so the in-memory
// counters/last-turns survive restarts. After this, appendTurn keeps them
// hot and the file is never re-read per request.
func warmState() {
	f, err := os.Open(sessionsPath)
	if err != nil {
		return
	}
	defer f.Close()
	buf := getScanBuf()
	defer putScanBuf(buf)
	sc := bufio.NewScanner(f)
	sc.Buffer(buf, 1<<20)
	stateMu.Lock()
	for sc.Scan() {
		var t turn
		if json.Unmarshal(sc.Bytes(), &t) != nil {
			continue
		}
		if t.Session != "" {
			lastTurns[t.Session] = t
		}
		for _, prefix := range []string{"backend(", "autostart("} {
			if strings.HasPrefix(t.SLM, prefix) && strings.HasSuffix(t.SLM, ")") {
				useCounts[strings.TrimSuffix(strings.TrimPrefix(t.SLM, prefix), ")")]++
			}
		}
	}
	stateMu.Unlock()
}

// ledgerUsesByTarget: per-backend-URL turn counts for the roster.
func ledgerUsesByTarget() map[string]int {
	stateMu.Lock()
	defer stateMu.Unlock()
	out := make(map[string]int, len(useCounts))
	for k, v := range useCounts {
		out[k] = v
	}
	return out
}

// sessionsCmd: show the last N turns grouped by session, with the active SLM.
func sessionsCmd(args []string) {
	var n int = 20
	if len(args) > 0 {
		n = atoi(args[0])
		if n <= 0 {
			n = 20
		}
	}
	f, err := os.Open(sessionsPath)
	if err != nil {
		fmt.Println("no sessions yet:", err)
		return
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	buf := getScanBuf()
	defer putScanBuf(buf)
	sc.Buffer(buf, 1<<20)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	start := len(lines) - n
	if start < 0 {
		start = 0
	}
	var active map[string]turn = make(map[string]turn)
	for i := start; i < len(lines); i++ {
		var t turn
		if json.Unmarshal([]byte(lines[i]), &t) == nil {
			active[t.Session] = t
		}
	}
	fmt.Printf("%-12s %-22s %-8s %-8s %-9s %s\n", "SESSION", "LAST TURN (UTC)", "SLM", "LANG", "WALL", "SCOPE")
	var order []string
	for s := range active {
		order = append(order, s)
	}
	sortStrings(order)
	for _, s := range order {
		t := active[s]
		when := time.UnixMilli(t.Ts).UTC().Format("15:04:05")
		scope := t.ScopeEvent
		if scope == "" {
			scope = "-"
		}
		fmt.Printf("%-12s %-22s %-8s %-8s %-9s %s\n",
			truncate(s, 12), when, truncate(t.SLM, 22), t.Lang,
			fmt.Sprintf("%dms", t.WallMs), scope)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// lastSessionTurn: the most recent turn for a session (its "owner" SLM/lang).
// lastSessionTurn: the latest turn of a session, from RAM. Falls back to a
// tail scan of the ledger for sessions seen before this process started
// (the fallback result is cached in RAM).
func lastSessionTurn(session string) (turn, bool) {
	stateMu.Lock()
	if t, ok := lastTurns[session]; ok {
		stateMu.Unlock()
		return t, true
	}
	stateMu.Unlock()
	f, err := os.Open(sessionsPath)
	if err != nil {
		return turn{}, false
	}
	defer f.Close()
	buf := getScanBuf()
	defer putScanBuf(buf)
	var last turn
	var found bool = false
	sc := bufio.NewScanner(f)
	sc.Buffer(buf, 1<<20)
	for sc.Scan() {
		var t turn
		if json.Unmarshal([]byte(sc.Text()), &t) == nil && t.Session == session {
			last = t
			found = true
		}
	}
	if found {
		stateMu.Lock()
		lastTurns[session] = last
		stateMu.Unlock()
	}
	return last, found
}

// scopeCheck: resolve text through the sub-index; if the top hit's language
// differs from the session's active language, build the escalation message.
func scopeCheck(idxPath string, text string, activeLang string) (string, string) {
	idx, err := loadSubIndex(idxPath)
	if err != nil || len(idx.Sections) == 0 {
		return "", ""
	}
	hits := idx.resolve(text, 2)
	if len(hits) == 0 {
		return "", ""
	}
	top := hits[0]
	if top.Lang != activeLang && top.Score > 0 {
		msg := fmt.Sprintf(
			"destination hit out of scope - shall we delegate an SLM for %s (%s)?",
			top.Label, top.Lang)
		ev := fmt.Sprintf("out-of-scope->%s(%s)", top.Label, top.Lang)
		return ev, msg
	}
	return "", ""
}

// stripTiming: drop llama-cli's timing/chat lines from captured output.
func stripTiming(out string) string {
	var keep []string
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "t/s") || strings.HasPrefix(l, "> ") ||
			strings.HasPrefix(l, "[") && strings.Contains(l, "Exiting") {
			continue
		}
		keep = append(keep, l)
	}
	return strings.Join(keep, "\n")
}
