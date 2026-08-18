// serve.go - the llama harness: an HTTP gateway in front of the SLM servers.
//
// The raw llama-server web UI has no scope awareness: a python expert asked
// for Go code will happily produce mediocre Go. This gateway runs the
// out-of-scope protocol on EVERY request, then either:
//
//   - forwards to the owning SLM's llama-server (in-scope), or
//   - returns the escalation question to the user (out-of-scope), or
//   - asks the user to add the missing stack element ("get a slice from an
//     element of a stack") when no index section/backend exists for the hit.
//
// API (llama-server-compatible on purpose):
//   POST /completion   {"prompt": ..., "n_predict": ..., "session": "s1"}
//                      -> forwarded completion, or escalation JSON
//   GET  /health       {"status":"ok"}
//
// Escalation responses carry "content" (what the user sees), "stop": true,
// and the X-Gotato-Escalation header for machines.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type serveConfig struct {
	addr     string
	backends map[string]string // lang -> llama-server port
	mu       *sync.Mutex       // guards backends (autostart mutates it)
}

func defaultServeConfig() *serveConfig {
	return &serveConfig{
		addr: ":8090",
		backends: map[string]string{
			"python":     "http://127.0.0.1:8081", // python expert (0.6B + LoRA)
			"general":    "http://127.0.0.1:8082", // 2B generalist
			"hard":       "http://127.0.0.1:8083", // 4B generalist
			"translator": "http://127.0.0.1:8086", // 1.7B instruct (the bridge)
		},
		mu: &sync.Mutex{},
	}
}

type completionReq struct {
	Prompt   string `json:"prompt"`
	NPredict int    `json:"n_predict"`
	Session  string `json:"session"`
	ChainCap int    `json:"chain_cap"` // optional per-request context cap
}

type completionResp struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
}

func escalationJSON(msg string) []byte {
	out, _ := json.Marshal(completionResp{Content: msg, Stop: true})
	return out
}

// hasBackend: is a llama-server actually reachable for this language?
func hasBackend(cfg *serveConfig, lang string) bool {
	port, ok := cfg.backends[lang]
	if !ok {
		return false
	}
	resp, err := http.Get(port + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return strings.Contains(string(body), "ok")
}

// autostartLang: spawn a llama-server for a ready expert of lang (the
// "say yes and I will start one" promise). Returns the backend URL or "".
func autostartLang(cfg *serveConfig, lang string) string {
	idx := loadIndex()
	e, ok := idx[lang]
	if !ok || e.Status != "ready" || e.Lora == "" {
		return ""
	}
	base := fleetDir + "/" + e.Base
	lora := fleetDir + "/" + e.Lora
	if _, err := os.Stat(base); err != nil {
		return ""
	}
	if _, err := os.Stat(lora); err != nil {
		return ""
	}
	var port int = 8084
	for {
		url := fmt.Sprintf("http://127.0.0.1:%d", port)
		if resp, err := http.Get(url + "/health"); err != nil || resp.StatusCode != 200 {
			break
		}
		port++
		if port > 8099 {
			return ""
		}
	}
	serverBin := strings.TrimSuffix(llamaCli, "llama-cli") + "llama-server"
	cmd := exec.Command(serverBin, "-m", base, "--lora", lora,
		"-t", "4", "-c", "4096", "--port", strconv.Itoa(port))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return ""
	}
	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	for i := 0; i < 60; i++ {
		resp, err := http.Get(url + "/health")
		if err == nil {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if strings.Contains(string(body), "ok") {
				cfg.mu.Lock()
				cfg.backends[lang] = url
				cfg.mu.Unlock()
				return url
			}
		}
		time.Sleep(2 * time.Second)
	}
	return ""
}

func serveHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":"ok"}`))
			return
		}
		if r.Method != http.MethodPost || r.URL.Path != "/completion" {
			http.NotFound(w, r)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body", 400)
			return
		}
		var req completionReq
		if json.Unmarshal(body, &req) != nil || req.Prompt == "" {
			http.Error(w, `{"error":"bad request"}`, 400)
			return
		}
		if req.Session == "" {
			req.Session = "default"
		}

		// ---- consent flow: "yes" to a pending escalation ----------------
		var pendingLang string = ""
		if prev, ok := lastSessionTurn(req.Session); ok && prev.Pending != "" {
			pendingLang = prev.Pending
		}
		if pendingLang != "" {
			lower := strings.ToLower(req.Prompt)
			var consented bool = strings.Contains(lower, "yes") ||
				strings.Contains(lower, "sure") || strings.Contains(lower, "go ahead") ||
				strings.Contains(lower, "ok") || strings.Contains(lower, "delegate") ||
				strings.Contains(lower, "do it")
			if consented {
				if url := autostartLang(cfg, pendingLang); url != "" {
					fmt.Fprintf(os.Stderr, "[gateway] autostarted %s backend at %s\n", pendingLang, url)
					// adopt the owner so the next task reaches the new slice
					appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
						SLM: "autostart(" + url + ")", Lang: pendingLang, Tokens: 0,
						WallMs: 0, ScopeEvent: "delegated->" + pendingLang})
					msg := fmt.Sprintf("delegated - the %s slice is running (%s). send the task.",
						pendingLang, url)
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("X-Gotato-Escalation", "delegated->"+pendingLang)
					_, _ = w.Write(escalationJSON(msg))
					return
				}
			}
		}

		// ---- the scope protocol ------------------------------------------
		var idxPath string = fleetDir + "/subindex.json"
		var semanticLang string = ""
		var semanticMargin float64 = 0
		var semanticLabel string = ""
		if hits := resolveIndex(idxPath, req.Prompt, 2); len(hits) > 0 {
			top := hits[0]
			if len(hits) > 1 && hits[1].Score > 0 {
				semanticMargin = top.Score / hits[1].Score
			} else {
				semanticMargin = 2.0
			}
			if semanticMargin >= 2.0 {
				semanticLang = top.Lang
				semanticLabel = top.Label
			}
		}
		// lexical scope signal: when the index has no opinion but the prompt
		// is clearly code in a language that isn't the session owner's, treat
		// it as out of scope too (the "pasted a Go element" case).
		var lexLang string = ""
		var lexHits int = 0
		if semanticLang == "" && looksLikeCode(req.Prompt) {
			var lexConf float64 = 0
			lexLang, lexConf, lexHits = detectLanguageN(req.Prompt, "")
			_ = lexConf
			// trust the lexical signal only on >= 2 distinct pattern hits
			if lexHits < 2 || lexLang == "default" {
				lexLang = ""
			}
		}
		var scopeLang string = semanticLang
		var scopeLabel string = semanticLabel
		if scopeLang == "" {
			scopeLang = lexLang
			scopeLabel = lexLang
		}
		var ownerLang string = ""
		if prev, ok := lastSessionTurn(req.Session); ok {
			ownerLang = prev.Lang
		}
		fmt.Fprintf(os.Stderr, "[gateway] session=%s owner=%q semantic=%q/%q lex=%q/hits=%d scope=%q codeish=%v\n",
			req.Session, ownerLang, semanticLang, semanticLabel, lexLang, lexHits, scopeLang,
			looksLikeCode(req.Prompt))

		// branch 1: out-of-scope vs the session owner -> ask the user
		if ownerLang != "" && scopeLang != "" && scopeLang != ownerLang {
			msg := fmt.Sprintf(
				"destination hit out of scope - shall we delegate an SLM for %s (%s)?",
				scopeLabel, scopeLang)
			appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
				SLM: "none", Lang: ownerLang, Tokens: 0, WallMs: 0,
				ScopeEvent: "out-of-scope->" + semanticLang, Escalation: msg,
				Pending: semanticLang})
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Gotato-Escalation", "out-of-scope->"+semanticLang)
			_, _ = w.Write(escalationJSON(msg))
			return
		}

		// branch 2: the hit's language has no backend -> ask for a slice
		var target string = ""
		if scopeLang != "" {
			if hasBackend(cfg, scopeLang) {
				target = cfg.backends[scopeLang]
			} else {
				msg := fmt.Sprintf(
					"no %s slice is running yet - add a %s element to the stack "+
						"(or say yes and I will start one) so the watcher can slice it.",
					scopeLang, scopeLang)
				appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
					SLM: "none", Lang: ownerLang, Tokens: 0, WallMs: 0,
					ScopeEvent: "missing-slice->" + scopeLang, Escalation: msg,
					Pending: scopeLang})
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Gotato-Escalation", "missing-slice->"+scopeLang)
				_, _ = w.Write(escalationJSON(msg))
				return
			}
		}
		// no semantic hit: fall back to the owner, then the generalist
		if target == "" {
			if ownerLang != "" && hasBackend(cfg, ownerLang) {
				target = cfg.backends[ownerLang]
			} else {
				target = cfg.backends["general"]
			}
		}

		// ---- context chaining: approach the cap -> summarize + fresh context
		var chainCap int = defaultChainCap
		if req.ChainCap > 0 {
			chainCap = req.ChainCap
		}
		prefix, chained := chainContext(cfg, req.Session, req.Prompt, chainCap)
		var fwdBody string = string(body)
		if chained {
			fwdBody = fmt.Sprintf(`{"prompt":%s,"n_predict":%d,"cache_prompt":false}`,
				jsonString(prefix+"\n\n"+req.Prompt), req.NPredict)
			appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
				SLM: "summarizer", Lang: "", Tokens: 0, WallMs: 0,
				ScopeEvent: "context-chain", Escalation: "chained at cap"})
			fmt.Fprintf(os.Stderr, "[gateway] session %s: context chained (cap %d)\n",
				req.Session, chainCap)
		}

		// ---- language bridge: translate the prompt for the SLM -----------
		var origPrompt string = req.Prompt
		if bridgeZH {
			req.Prompt = translateToZH(cfg, req.Prompt)
			// the SLM must actually SEE the translation: rebuild the body
			var streamFlag string = "false"
			if strings.Contains(fwdBody, "\"stream\":true") ||
				strings.Contains(fwdBody, "\"stream\": true") {
				streamFlag = "true"
			}
			fwdBody = fmt.Sprintf(`{"prompt":%s,"n_predict":%d,"session":%s,"stream":%s,"temperature":0}`,
				jsonString(req.Prompt), req.NPredict, jsonString(req.Session), streamFlag)
		}

		fmt.Fprintf(os.Stderr, "[bridge] session=%s orig=%s zh=%s target=%s\n",
			req.Session, contentHash(origPrompt)[:8], contentHash(req.Prompt)[:8], target)

		// ---- forward to the chosen SLM ------------------------------------
		// cache_prompt is always false: warm slots carry the PREVIOUS reply
		// into the next request, which breaks determinism. Context continuity
		// is the chain feature's job - it rebuilds the prefix explicitly.
		if !strings.Contains(fwdBody, "cache_prompt") {
			fwdBody = strings.TrimSuffix(fwdBody, "}") + `,"cache_prompt":false}`
		}
		// determinism contract: the gateway forces greedy unless the client
		// explicitly asked for sampling.
		if !strings.Contains(fwdBody, "\"temperature\"") {
			fwdBody = strings.TrimSuffix(fwdBody, "}") + `,"temperature":0}`
		}
		// SLM display tag: base name + topic suffix (the TUI shows this next
		// to its "Thinking" indicator).
		var servedLang string = scopeLang
		if servedLang == "" {
			servedLang = ownerLang
		}
		var slmTag string = slmDisplayTag(target, scopeLang, scopeLabel, ownerLang)

		// ---- streaming: relay SSE so the TUI can render live --------------
		var isStream bool = strings.Contains(fwdBody, "\"stream\":true") ||
			strings.Contains(fwdBody, "\"stream\": true")
		t0 := time.Now()
		resp, err := http.Post(target+"/completion", "application/json", strings.NewReader(fwdBody))
		if err != nil {
			http.Error(w, `{"error":"backend unreachable: `+target+`"}`, 502)
			return
		}
		defer resp.Body.Close()
		if isStream {
			streamRelay(w, resp, &req, target, slmTag, chained, t0, servedLang)
			return
		}
		out, _ := io.ReadAll(resp.Body)

		// ledger + archive + chain marker BEFORE writing, so headers and the
		// chained content prefix actually reach the client.
		var cr completionResp
		var tokens int = 0
		var promptTokens int = 0
		if json.Unmarshal(out, &cr) == nil && cr.Content != "" {
			_ = json.Unmarshal(out, &struct{ Content *string }{})
		}
		var respTimings struct {
			Timings struct {
				PredictedN int `json:"predicted_n"`
				PromptN   int `json:"prompt_n"`
			} `json:"timings"`
		}
		_ = json.Unmarshal(out, &respTimings)
		tokens = respTimings.Timings.PredictedN
		promptTokens = respTimings.Timings.PromptN
		appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
			SLM: "backend(" + target + ")", Lang: servedLang,
			Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
		var enReply string = cr.Content
		if bridgeZH {
			enReply = translateToEN(cfg, cr.Content)
		}
		fmt.Fprintf(os.Stderr, "[bridge] raw=%s en=%s\n",
			contentHash(cr.Content)[:8], contentHash(enReply)[:8])
		// archive content (storage is cheap) and track the live position
		appendContent(req.Session, "user", origPrompt)
		var cr2 completionResp
		if json.Unmarshal(out, &cr2) == nil && cr2.Content != "" {
			if bridgeZH {
				cr2.Content = translateToEN(cfg, cr2.Content)
				chainedOut, _ := json.Marshal(cr2)
				out = chainedOut
				w.Header().Set("X-Gotato-Bridge", "zh")
			}
			appendContent(req.Session, "assistant", cr2.Content)
			if chained {
				cr2.Content = "[context chained - summary prepended to next turn]\n" + cr2.Content
				chainedOut, _ := json.Marshal(cr2)
				out = chainedOut
				w.Header().Set("X-Gotato-Chain", "true")
			}
		}
		if promptTokens > 0 {
			addPosition(req.Session, promptTokens+tokens)
		} else {
			addPosition(req.Session, estTokens(req.Prompt)+tokens)
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Gotato-Backend", target)
		w.Header().Set("X-Gotato-SLM", slmTag)
		if chained {
			w.Header().Set("X-Gotato-Chain", "true")
		}
		_, _ = w.Write(out)
	}
}

func serveCmd(args []string) {
	cfg := defaultServeConfig()
	for i := 0; i < len(args); i++ {
		if args[i] == "--addr" && i+1 < len(args) {
			cfg.addr = args[i+1]
			i++
		} else if args[i] == "--bridge" && i+1 < len(args) {
			bridgeZH = args[i+1] == "zh"
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/completion", serveHandler(cfg))
	mux.HandleFunc("/health", serveHandler(cfg))
	fmt.Printf("[gateway] listening on %s | backends: %v\n", cfg.addr, cfg.backends)
	if err := http.ListenAndServe(cfg.addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// slmDisplayTag: the human-facing SLM name with a topic suffix, e.g.
// "python-expert · utils" or "rust-slice · rust". Shown by the TUI next to
// its Thinking indicator and sent as the X-Gotato-SLM header.
func slmDisplayTag(target string, lang string, label string, owner string) string {
	var base string = "slm"
	switch {
	case strings.Contains(target, ":8081"):
		base = "python-expert"
	case strings.Contains(target, ":8082"):
		base = "2b-general"
	case strings.Contains(target, ":8083"):
		base = "4b-general"
	default:
		if lang != "" {
			base = lang + "-slice"
		}
	}
	var topic string = label
	if topic == "" {
		topic = lang
	}
	if topic == "" {
		topic = owner
	}
	if topic == "" {
		topic = "general"
	}
	return base + " \u00b7 " + topic
}

// streamRelay: passthrough SSE from the backend, archiving the accumulated
// content and recording the turn. Headers are set before the first flush so
// the client sees the SLM tag before any token arrives.
func streamRelay(w http.ResponseWriter, resp *http.Response, req *completionReq,
	target string, slmTag string, chained bool, t0 time.Time, servedLang string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Gotato-Backend", target)
	w.Header().Set("X-Gotato-SLM", slmTag)
	if chained {
		w.Header().Set("X-Gotato-Chain", "true")
	}
	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}
	var acc strings.Builder
	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			acc.Write(buf[:n])
			_, _ = w.Write(buf[:n])
			if ok {
				flusher.Flush()
			}
		}
		if err != nil {
			break
		}
	}
	// parse the accumulated SSE for content + final timings
	var content strings.Builder
	var tokens int = 0
	var promptTokens int = 0
	sc := bufio.NewScanner(strings.NewReader(acc.String()))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var frame struct {
			Content  string `json:"content"`
			Stop     bool   `json:"stop"`
			Timings  struct {
				PredictedN int `json:"predicted_n"`
				PromptN   int `json:"prompt_n"`
			} `json:"timings"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(line[5:])), &frame) != nil {
			continue
		}
		if frame.Content != "" {
			content.WriteString(frame.Content)
		}
		if frame.Stop {
			tokens = frame.Timings.PredictedN
			promptTokens = frame.Timings.PromptN
		}
	}
	appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
		SLM: "backend(" + target + ")", Lang: servedLang,
		Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
	appendContent(req.Session, "user", req.Prompt)
	if content.Len() > 0 {
		appendContent(req.Session, "assistant", content.String())
	}
	if promptTokens > 0 {
		addPosition(req.Session, promptTokens+tokens)
	} else {
		addPosition(req.Session, estTokens(req.Prompt)+tokens)
	}
}
