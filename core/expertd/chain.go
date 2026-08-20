// chain.go - context policy for sessions.
//
// Chaining was REMOVED (Aug 2025): it cost minutes per turn on the potato,
// and for chat sessions it rebuilt the prompt from the client transcript
// anyway, so the summarizer's work never shrank the live context. The
// contract now: the backend window is fixed (4096), the gateway signals
// exhaustion (GOTATO_CONTEXT_EXHAUSTED) instead of truncating, and the
// client restarts the context from its summary cache. This file keeps the
// append-only session archive (the ledger every turn is written to) and
// the deterministic token estimate. The summary SEGMENT store (segstore.go)
// is the cache the client-side restart rehydrates from.
//
// State: per-session archive files (sessions/<id>.jsonl); no in-RAM
// position bookkeeping - the live window is the client's responsibility.
package main

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
)

const defaultChainCap int = 3072 // legacy: unused since chaining removal

const backendWindow int = 7000 // est tokens safe under the -c 8192 window (q8_0 KV)

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
	var dir = filepath.Dir(sessionContentPath(session))
	{
		var err = os.MkdirAll(dir, 0755)
		if err != nil {
			return
		}
	}
	var f, err = os.OpenFile(sessionContentPath(session), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	var data []byte

	data, _ = json.Marshal(chainTurn{Ts: nowMs(), Role: role, Content: content})
	_, _ = f.Write(append(data, '\n'))
}

// readContent: last N archived turns for a session.
func readContent(session string, lastN int) []chainTurn {
	var f, err = os.Open(sessionContentPath(session))
	if err != nil {
		return nil
	}
	defer f.Close()
	var all []chainTurn
	var sc = bufio.NewScanner(f)
	var buf = getScanBuf()
	defer putScanBuf(buf)
	sc.Buffer(buf, 1<<20)
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
// jsonString: JSON-encode a string (used when rebuilding forwarded bodies).
func jsonString(s string) string {
	var b []byte

	b, _ = json.Marshal(s)
	return string(b)
}
