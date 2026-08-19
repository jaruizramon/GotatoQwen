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
	"runtime"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
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
	// in-flight SLM tracking: which slice is serving RIGHT NOW. Read by
	// GET /slms so the TUI can haptically highlight the active SLM.
	activeName   string // slm base name ("python-expert", "2b-general", ...)
	activeLang   string
	activeSince  int64 // ms epoch; 0 = idle
}

// setActive/clearActive: mark the currently-serving SLM. Called around the
// actual backend forward in both /completion and /v1/chat/completions.
func (cfg *serveConfig) setActive(name string, lang string) {
	cfg.mu.Lock()
	cfg.activeName = name
	cfg.activeLang = lang
	cfg.activeSince = time.Now().UnixMilli()
	cfg.mu.Unlock()
}

func (cfg *serveConfig) clearActive() {
	cfg.mu.Lock()
	cfg.activeName = ""
	cfg.activeLang = ""
	cfg.activeSince = 0
	cfg.mu.Unlock()
}

func defaultServeConfig() *serveConfig {
	return &serveConfig{
		addr: ":8090",
		backends: map[string]string{
			"python":     "http://127.0.0.1:8081", // python expert (0.6B + LoRA)
			"general":    "http://127.0.0.1:8082", // 2B generalist
			"hard":       "http://127.0.0.1:8083", // 4B generalist
			"rust":       "http://127.0.0.1:8084", // rust expert (0.6B + LoRA)
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
	Stream   bool   `json:"stream,omitempty"`
	// RoutePrompt: the text the router should scope on. When empty, the
	// router uses Prompt. The chat endpoint sets it to the LAST user
	// message so the index/lexical detectors never see transcript markers
	// like "[user] ".
	RoutePrompt string `json:"route_prompt,omitempty"`
	// SkipBridge: bypass the zh prompt/response translation for this
	// request. OpenAI-chat clients (omp) get English in -> English out;
	// the bridge doubles latency and leaks ZH into streamed replies.
	SkipBridge bool `json:"skip_bridge,omitempty"`
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
	resp, err := httpClient.Get(port + "/health")
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
	if !ok || e.Status != "ready" || (e.Lora == "" && e.Mask == "") {
		return ""
	}
	base := fleetDir + "/" + e.Base
	var servePath string = base
	var lora string = ""
	if e.Mask != "" {
		// a mask slice IS the model: zeroed weights, no adapter
		servePath = fleetDir + "/" + e.Mask
		if _, err := os.Stat(servePath); err != nil {
			return ""
		}
	} else {
		lora = fleetDir + "/" + e.Lora
		if _, err := os.Stat(base); err != nil {
			return ""
		}
		if _, err := os.Stat(lora); err != nil {
			return ""
		}
	}
	var port int = 8084
	for {
		url := fmt.Sprintf("http://127.0.0.1:%d", port)
		if resp, err := httpClient.Get(url + "/health"); err != nil || resp.StatusCode != 200 {
			break
		}
		port++
		if port > 8099 {
			return ""
		}
	}
	serverBin := strings.TrimSuffix(llamaCli, "llama-cli") + "llama-server"
	cmdArgs := []string{"-m", servePath}
	if lora != "" {
		cmdArgs = append(cmdArgs, "--lora", lora)
	}
	// -np 1: this llama.cpp build's auto default splits the 4096 ctx into
	// 4 slots of ~1024 each, and a chained omp prompt (3400+ tokens) cannot
	// fit one slot -> "Context size has been exceeded" -> empty replies.
	cmdArgs = append(cmdArgs, "-t", "4", "-c", "4096", "-np", "1", "--port", strconv.Itoa(port))
	cmd := exec.Command(serverBin, cmdArgs...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		return ""
	}
	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	for i := 0; i < 60; i++ {
		resp, err := httpClient.Get(url + "/health")
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

// route: the forwarding plan produced by the scope protocol. Either the
// backend to call (target/fwdBody/...) or a direct-answer override
// (escalation question, delegation confirmation, missing-slice prompt).
type routePlan struct {
	target         string
	slmTag         string
	servedLang     string
	fwdBody        string
	origPrompt     string
	override       string // non-empty: answer directly, do not forward
	overrideHeader string // X-Gotato-Escalation value for the override
}

// routeCompletion: the whole pre-forward pipeline - consent flow, scope
// protocol (index + lexical), context chaining, language bridge, and body
// normalization. Shared by /completion and /v1/chat/completions so both
// entry points run identical routing.
func routeCompletion(cfg *serveConfig, req *completionReq) *routePlan {
	r := &routePlan{}

	// ---- consent flow: "yes" to a pending escalation ----------------
	// Scoped on the routing text (chat endpoint: the LAST user message) so
	// an agent system prompt full of "yes/ok/do it" can never accidentally
	// consent to a pending delegation.
	routeText := req.RoutePrompt
	if routeText == "" {
		routeText = req.Prompt
	}
	var pendingLang string = ""
	if prev, ok := lastSessionTurn(req.Session); ok && prev.Pending != "" {
		pendingLang = prev.Pending
	}
	if pendingLang != "" {
		lower := strings.ToLower(routeText)
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
				r.override = fmt.Sprintf("delegated - the %s slice is running (%s). send the task.",
					pendingLang, url)
				r.overrideHeader = "delegated->" + pendingLang
				return r
			}
			// no ready slice: promise to slice one (the oracle registered the
			// language, the watcher may be mid-build; racing is harmless).
			if langKnown(pendingLang) && spawnSliceBuild(pendingLang) {
				appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
					SLM: "slice-build", Lang: pendingLang, Tokens: 0, WallMs: 0,
					ScopeEvent: "slicing->" + pendingLang})
				r.override = fmt.Sprintf(
					"slicing an SLM for %s now (gglora, ~20 min on the potato) - the watcher will publish it to the index.",
					pendingLang)
				r.overrideHeader = "slicing->" + pendingLang
				return r
			}
		}
	}

	// ---- the scope protocol ------------------------------------------
	// The router scopes on RoutePrompt when the caller provides it (chat
	// transcripts carry <|im_start|> markers and a system prompt that
	// would pollute the semantic index and lexical detectors); otherwise
	// on the full prompt.
	if routeText == "" {
		routeText = req.Prompt
	}
	var idxPath string = fleetDir + "/subindex.json"
	var semanticLang string = ""
	var semanticMargin float64 = 0
	var semanticLabel string = ""
	// ---- the 2B decides the topic first --------------------------------
	// User contract: every prompt goes to the general 2B SLM first; the
	// inference then switches to the stack SLM the task is about. The 2B's
	// classification is authoritative (no consent round-trip); the Go
	// semantic/lexical protocol below is the fallback when the 2B says
	// "general" or an unknown word.
	var fromBrain bool = false
	if routeText != "" {
		if topic := routerBrain(cfg, routeText); topic != "" {
			semanticLang = topic
			semanticLabel = topic
			fromBrain = true
			fmt.Fprintf(os.Stderr, "[brain] session=%s topic=%s (2B decided)\n", req.Session, topic)
		}
	}
	if !fromBrain {
		if hits := resolveIndex(idxPath, routeText, 2); len(hits) > 0 {
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
	}
	// lexical scope signal: when the index has no opinion but the prompt
	// is clearly code in a language that isn't the session owner's, treat
	// it as out of scope too (the "pasted a Go element" case).
	var lexLang string = ""
	var lexHits int = 0
	if semanticLang == "" && looksLikeCode(routeText) {
		var lexConf float64 = 0
		lexLang, lexConf, lexHits = detectLanguageN(routeText, "")
		_ = lexConf
		// trust the lexical signal only on >= 2 distinct pattern hits AND
		// only when the stack actually owns that SLM (manifest-aware: the
		// fallback must not reach past the stack's own slices).
		if lexHits < 2 || lexLang == "default" || !manifestHas(lexLang) {
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
		looksLikeCode(routeText))

	// branch 1: out-of-scope vs the session owner -> switch (2B decision)
	// or ask the user (index decision)
	if ownerLang != "" && scopeLang != "" && scopeLang != ownerLang {
		if fromBrain {
			// the 2B decided: adopt the new owner and route (no consent)
			appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
				SLM: "brain-switch", Lang: scopeLang, Tokens: 0, WallMs: 0,
				ScopeEvent: "brain-switch->" + scopeLang})
		} else {
			msg := fmt.Sprintf(
				"destination hit out of scope - shall we delegate an SLM for %s (%s)?",
				scopeLabel, scopeLang)
			appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
				SLM: "none", Lang: ownerLang, Tokens: 0, WallMs: 0,
				ScopeEvent: "out-of-scope->" + semanticLang, Escalation: msg,
				Pending: semanticLang})
			r.override = msg
			r.overrideHeader = "out-of-scope->" + semanticLang
			return r
		}
	}

	// branch 2: the hit's language has no backend -> start it (2B
	// decision) or ask for a slice (index decision)
	var target string = ""
	if scopeLang != "" {
		if hasBackend(cfg, scopeLang) {
			target = cfg.backends[scopeLang]
		} else if fromBrain {
			// the 2B already decided: start the ready expert without asking
			target = autostartLang(cfg, scopeLang)
		}
		if target == "" {
			msg := fmt.Sprintf(
				"no %s slice is running yet - say yes and I will slice an SLM for %s "+
					"(or add a %s element to the stack so the watcher picks it up).",
				scopeLang, scopeLang, scopeLang)
			appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
				SLM: "none", Lang: ownerLang, Tokens: 0, WallMs: 0,
				ScopeEvent: "missing-slice->" + scopeLang, Escalation: msg,
				Pending: scopeLang})
			r.override = msg
			r.overrideHeader = "missing-slice->" + scopeLang
			return r
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

	// ---- context guard: no chaining (removed - it cost minutes per turn
	// and rebuilt from client transcripts anyway), no silent truncation.
	// When the session no longer fits the backend window, the client gets
	// a machine-detectable exhausted response and restarts the context
	// from its summary cache; the segstore rehydrates past summaries.
	if estTokens(req.Prompt) > backendWindow {
		r.override = "[context exhausted] the session no longer fits the SLM window - restart the context from the summary cache"
		r.overrideHeader = "context-exhausted"
		return r
	}
	var fwdBody string = ""
	{
		rawBody, _ := json.Marshal(req)
		fwdBody = string(rawBody)
	}

	// ---- language bridge: translate the prompt for the SLM -----------
	r.origPrompt = req.Prompt
	if bridgeZH && !req.SkipBridge {
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
		req.Session, contentHash(r.origPrompt)[:8], contentHash(req.Prompt)[:8], target)

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
	// transcript-style prompts (chat endpoint): stop at the next role
	// marker so the SLM finishes the assistant turn instead of rambling
	// to the n_predict cap.
	if req.Stream || strings.Contains(fwdBody, "<|im_start|>") {
		if !strings.Contains(fwdBody, "\"stop\"") {
			fwdBody = strings.TrimSuffix(fwdBody, "}") +
				`,"stop":["<|im_end|>","<|im_start|>"]}`
		}
	}
	// SLM display tag: base name + topic suffix (the TUI shows this next
	// to its "Thinking" indicator).
	r.servedLang = scopeLang
	if r.servedLang == "" {
		r.servedLang = ownerLang
	}
	r.target = target
	r.slmTag = slmDisplayTag(target, scopeLang, scopeLabel, ownerLang)
	r.fwdBody = fwdBody
	return r
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

		// ---- route: scope protocol + chain + bridge -> forwarding plan
		plan := routeCompletion(cfg, &req)
		if plan.override != "" {
			w.Header().Set("Content-Type", "application/json")
			if plan.overrideHeader != "" {
				w.Header().Set("X-Gotato-Escalation", plan.overrideHeader)
			}
			_, _ = w.Write(escalationJSON(plan.override))
			return
		}

		// ---- forward to the chosen SLM ------------------------------------
		cfg.setActive(slmBaseName(plan.target, plan.servedLang), plan.servedLang)
		defer cfg.clearActive()
		t0 := time.Now()
		resp, err := httpClient.Post(plan.target+"/completion", "application/json", strings.NewReader(plan.fwdBody))
		if err != nil {
			http.Error(w, `{"error":"backend unreachable: `+plan.target+`"}`, 502)
			return
		}
		defer resp.Body.Close()
		if strings.Contains(plan.fwdBody, "\"stream\":true") ||
			strings.Contains(plan.fwdBody, "\"stream\": true") {
			streamRelay(w, resp, &req, plan.target, plan.slmTag, t0, plan.servedLang)
			return
		}
		out, _ := io.ReadAll(resp.Body)

		// ledger + archive BEFORE writing, so headers and content reach
		// the client.
		var cr completionResp
		var tokens int = 0
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
		appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
			SLM: "backend(" + plan.target + ")", Lang: plan.servedLang,
			Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
		var enReply string = cr.Content
		if bridgeZH {
			enReply = translateToEN(cfg, cr.Content)
		}
		fmt.Fprintf(os.Stderr, "[bridge] raw=%s en=%s\n",
			contentHash(cr.Content)[:8], contentHash(enReply)[:8])
		// archive content (storage is cheap) and track the live position
		appendContent(req.Session, "user", plan.origPrompt)
		var cr2 completionResp
		if json.Unmarshal(out, &cr2) == nil && cr2.Content != "" {
			if bridgeZH {
				cr2.Content = translateToEN(cfg, cr2.Content)
				chainedOut, _ := json.Marshal(cr2)
				out = chainedOut
				w.Header().Set("X-Gotato-Bridge", "zh")
			}
			appendContent(req.Session, "assistant", cr2.Content)
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Gotato-Backend", plan.target)
		w.Header().Set("X-Gotato-SLM", plan.slmTag)
		_, _ = w.Write(out)
	}
}

func serveCmd(args []string) {
	cfg := defaultServeConfig()
	verifyURL = cfg.backends["general"] // the 2B verifies the tool-executor's writes
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
	mux.HandleFunc("/slms", slmsHandler(cfg))
	mux.HandleFunc("/v1/chat/completions", chatHandler(cfg))
	mux.HandleFunc("/v1/models", modelsHandler(cfg))
	mux.HandleFunc("/manifest", manifestHandler)
	mux.HandleFunc("/memstats", memstatsHandler(cfg))
	segCacheInit() // the reserved summary-RAM cap (GOTATO_SUMMARY_CACHE_MB)
	warmState() // one cold ledger read; the hot state is RAM-only afterwards
	fmt.Printf("[gateway] listening on %s | backends: %v\n", cfg.addr, cfg.backends)
	if err := http.ListenAndServe(cfg.addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func backendPort(url string) string {
	i := strings.LastIndex(url, ":")
	if i < 0 {
		return url
	}
	return url[i+1:]
}

// slmsHandler: GET /slms - the live SLM roster for the TUI.
//
//	{"roster":[{"name":"python-expert","lang":"python","port":8081,"uses":12,"used":true},...],
//	 "active":{"name":"python-expert","lang":"python","since":1787...}|"",
//	 "ts":...}
//
// Roster = every configured backend. "used" = served at least one turn in
// the ledger; "active" = the slice serving RIGHT NOW (in-flight request).
func slmsHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/slms" {
			http.NotFound(w, r)
			return
		}
		cfg.mu.Lock()
		backends := make(map[string]string, len(cfg.backends))
		for k, v := range cfg.backends {
			backends[k] = v
		}
		activeName, activeLang, activeSince := cfg.activeName, cfg.activeLang, cfg.activeSince
		cfg.mu.Unlock()

		// ledger-discovered slices: SLMs that served turns but are not in the
		// configured map (manually started servers) still belong in the roster
		// - "all of the used SLMs shown" is the contract.
		ledgerLangs := map[string]string{} // url -> lang
		if f, err := os.Open(sessionsPath); err == nil {
			sc := bufio.NewScanner(f)
			sc.Buffer(make([]byte, 1<<20), 1<<20)
			for sc.Scan() {
				var t turn
				if json.Unmarshal(sc.Bytes(), &t) != nil {
					continue
				}
				for _, prefix := range []string{"backend(", "autostart("} {
					if strings.HasPrefix(t.SLM, prefix) && strings.HasSuffix(t.SLM, ")") {
						url := strings.TrimSuffix(strings.TrimPrefix(t.SLM, prefix), ")")
						if _, ok := backends[t.Lang]; !ok && t.Lang != "" {
							backends[t.Lang] = url
							ledgerLangs[url] = t.Lang
						}
					}
				}
			}
			f.Close()
		}
		_ = activeLang // carried for symmetry; the TUI keys on the name
		uses := ledgerUsesByTarget()
		type slmEntry struct {
			Name   string `json:"name"`
			Lang   string `json:"lang"`
			Port   string `json:"port"`
			Target string `json:"target"`
			Uses   int    `json:"uses"`
			Used   bool   `json:"used"`
		}
		// ---- manifest-aware roster ----------------------------------
		// omp-potato's SLM bar shows ONLY this stack's own slices (the
		// per-stack manifest) plus the generalists every prompt consults
		// first (the 2B brain) and the 4B hard fallback. Slices from other
		// stacks (gdscript-slice, rust-slice, ...) never appear.
		var rosterLangs []string = stackManifestLangs()
		roster := make([]slmEntry, 0, len(rosterLangs)+2)
		if gen, ok := backends["general"]; ok {
			roster = append(roster, slmEntry{Name: "2b-general", Lang: "general",
				Port: backendPort(gen), Target: gen, Uses: uses[gen], Used: uses[gen] > 0})
		}
		if hard, ok := backends["hard"]; ok {
			roster = append(roster, slmEntry{Name: "4b-general", Lang: "hard",
				Port: backendPort(hard), Target: hard, Uses: uses[hard], Used: uses[hard] > 0})
		}
		for _, lang := range rosterLangs {
			var url string = ""
			if b, ok := backends[lang]; ok && hasBackend(cfg, lang) {
				url = b
			}
			roster = append(roster, slmEntry{
				Name:   slmBaseName(url, lang),
				Lang:   lang,
				Port:   backendPort(url),
				Target: url,
				Uses:   uses[url],
				Used:   uses[url] > 0,
			})
		}
		var active *slmEntry
		if activeName != "" {
			for i := range roster {
				if roster[i].Name == activeName {
					active = &roster[i]
					break
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		if active == nil {
			_, _ = w.Write([]byte(fmt.Sprintf(
				`{"roster":%s,"active":"","ts":%d}`,
				mustJSON(roster), time.Now().UnixMilli())))
			return
		}
		_, _ = w.Write([]byte(fmt.Sprintf(
			`{"roster":%s,"active":{"name":%s,"lang":%s,"since":%d},"ts":%d}`,
			mustJSON(roster), jsonString(active.Name), jsonString(active.Lang), activeSince,
			time.Now().UnixMilli())))
	}
}

func mustJSON(v any) string {
	out, _ := json.Marshal(v)
	return string(out)
}

// memstatsHandler: GET /memstats - runtime heap + goroutine state, so
// memory leaks in the gateway are observable (the OMP status line polls
// /slms; a growing HeapAlloc at constant load means a leak).
// toolNames: the builtin tool set the chat contract exposes.
var toolNames = []string{"list_dir", "read_file", "run_command"}

// braceClose: index just past the brace that closes the object opening at
// bi (nested-brace aware; -1 if unclosed).
func braceClose(tail string, bi int) int {
	var depth int = 0
	for i := bi; i < len(tail); i++ {
		switch tail[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// partialCallHold: the longest suffix of tail that is a proper prefix of a
// call pattern (wrapped or bare) - a chunk may have cut the pattern mid-way,
// and emitting it would leak the call JSON to the client.
func partialCallHold(tail string) int {
	patterns := []string{"<tool_call>"}
	for _, tn := range toolNames {
		patterns = append(patterns, `{"name":"`+tn+`","arguments":{`)
	}
	var maxLen int = 0
	for _, p := range patterns {
		if len(p) > maxLen {
			maxLen = len(p)
		}
	}
	limit := len(tail)
	if limit > maxLen-1 {
		limit = maxLen - 1
	}
	for l := limit; l >= 1; l-- {
		suffix := tail[len(tail)-l:]
		for _, p := range patterns {
			if strings.HasPrefix(p, suffix) {
				return len(tail) - l
			}
		}
	}
	return -1
}

// routerBrain: the general 2B decides the task's topic first (the user
// contract: every prompt consults the 2B, then the inference switches to
// the stack SLM the task is about). Deterministic: temperature 0, tiny
// budget, thinking disabled via chat_template_kwargs (the Qwen3.5 base
// otherwise burns its whole budget on <think>). The candidate list is the
// CURRENT STACK's SLMs (the per-stack manifest, falling back to the
// catalogue), so the 2B only ever picks a slice this stack owns. The
// answer is matched against the candidates word-boundary; "general" or
// an unknown word returns "" and the Go index/lexical protocol takes
// over (the 2B may still miss - the fallback is the safety net).
func routerBrain(cfg *serveConfig, text string) string {
	var url string = cfg.backends["general"]
	if url == "" {
		return ""
	}
	var known []string = stackManifestLangs()
	var head string = text
	if len(head) > 600 {
		head = head[:600]
	}
	var prompt string = fmt.Sprintf(
		"Classify the programming language of this task (%s, or general).\nTask: %s\nLanguage: ",
		strings.Join(known, ", "), head)
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": prompt}},
		"max_tokens": 24, "temperature": 0,
		"chat_template_kwargs": map[string]any{"enable_thinking": false}})
	resp, err := httpPostJSONSlow(url+"/v1/chat/completions", body)
	if err != nil {
		return ""
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
		return ""
	}
	var lower string = strings.ToLower(out.Choices[0].Message.Content)
	for _, name := range known {
		if regexp.MustCompile(`\b`+regexp.QuoteMeta(name)+`\b`).MatchString(lower) {
			return name
		}
	}
	return ""
}

// manifestHas: does the current stack's manifest own an SLM for lang?
// Without a manifest, the catalogue is the authority (any catalogued
// language is sliceable for the stack on demand).
func manifestHas(lang string) bool {
	var langs map[string]*stackLangEntry = stackManifest()
	if langs == nil {
		return langKnown(lang)
	}
	_, ok := langs[lang]
	return ok
}

// stackManifestLangs: the language keys of the CURRENT stack's manifest
// (the gateway's cwd IS the stack), falling back to the catalogue when no
// manifest exists yet. The 2B only picks from these.
func stackManifestLangs() []string {
	var names map[string]bool = make(map[string]bool)
	if langs := stackManifest(); langs != nil {
		for name := range langs {
			names[name] = true
		}
	}
	if len(names) == 0 {
		for name := range langCatalog {
			if name != "summarizer" {
				names[name] = true
			}
		}
	}
	var out []string = make([]string, 0, len(names))
	for name := range names {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// toolBrain: the instruct 1.7B slice that drives the tool loop (the proven
// cowork model); falls back to the generalist if it is not registered.
func toolBrain(cfg *serveConfig) string {
	cfg.mu.Lock()
	defer cfg.mu.Unlock()
	if b, ok := cfg.backends["translator"]; ok {
		return b
	}
	return cfg.backends["general"]
}

func memstatsHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/memstats" {
			http.NotFound(w, r)
			return
		}
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		stateMu.Lock()
		nsessions := len(lastTurns)
		nuses := len(useCounts)
		stateMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write([]byte(fmt.Sprintf(
			`{"heap_alloc":%d,"heap_objects":%d,"num_gc":%d,"goroutines":%d,"sessions_in_ram":%d,"use_count_keys":%d}`,
			ms.HeapAlloc, ms.HeapObjects, ms.NumGC, runtime.NumGoroutine(), nsessions, nuses)))
	}
}

// stripThink: remove <think>...</think> blocks from SLM output. Qwen3's
// thinking mode wraps its chain-of-thought in those tags; the chat endpoint
// strips them so omp never sees potato reasoning (the cowork /completion
// passthrough keeps them for its own Thinking UI).
func stripThink(s string) string {
	if !strings.Contains(s, "<think>") {
		return s
	}
	var sb strings.Builder
	rest := s
	for {
		i := strings.Index(rest, "<think>")
		if i < 0 {
			sb.WriteString(rest)
			break
		}
		sb.WriteString(rest[:i])
		j := strings.Index(rest[i:], "</think>")
		if j < 0 {
			break // unterminated: drop the rest (the tag IS the ramble)
		}
		rest = rest[i+j+len("</think>"):]
	}
	return sb.String()
}

// ---- OpenAI-compatible chat endpoint ------------------------------------
//
// POST /v1/chat/completions lets any OpenAI-shaped client (omp, llama-ui,
// curl) drive the fleet through the SAME scope protocol as /completion.
// The response model field carries the serving SLM tag and the
// X-Gotato-SLM header names it too; the /slms roster stays in sync via
// the in-flight tracker.

func chatHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, `{"error":{"message":"read body"}}`, 400)
			return
		}
		var creq struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"` // string OR [{type,text},...]
			} `json:"messages"`
			Stream               bool `json:"stream"`
			MaxTokens            int  `json:"max_tokens"`
			MaxCompletionTokens  int  `json:"max_completion_tokens"`
			Session              string `json:"session"`
		}
		if json.Unmarshal(body, &creq) != nil || len(creq.Messages) == 0 {
			http.Error(w, `{"error":{"message":"bad request"}}`, 400)
			return
		}
		// content may be a plain string or an OpenAI block array; fold both.
		contentOf := func(raw json.RawMessage) string {
			var s string
			if json.Unmarshal(raw, &s) == nil {
				return s
			}
			var blocks []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(raw, &blocks) != nil {
				return ""
			}
			var sb strings.Builder
			for _, b := range blocks {
				if b.Type == "text" && b.Text != "" {
					sb.WriteString(b.Text)
				}
			}
			return sb.String()
		}
		session := creq.Session
		if session == "" {
			session = r.Header.Get("X-Gotato-Session")
		}
		if session == "" {
			session = "omp"
		}
		// flatten the transcript into Qwen's NATIVE chat template so the SLM
		// stops at <|im_end|> (its own EOS) instead of rambling past the
		// n_predict cap on ad-hoc [user]/[assistant] markers.
		var sb strings.Builder
		var lastUser string = ""
		for _, m := range creq.Messages {
			content := contentOf(m.Content)
			if strings.TrimSpace(content) == "" {
				continue
			}
			switch m.Role {
			case "system":
				sb.WriteString("<|im_start|>system\n" + content + "<|im_end|>\n")
			case "user":
				sb.WriteString("<|im_start|>user\n" + content + "<|im_end|>\n")
				lastUser = content
			case "assistant":
				sb.WriteString("<|im_start|>assistant\n" + content + "<|im_end|>\n")
			default:
				sb.WriteString("<|im_start|" + m.Role + "\n" + content + "<|im_end|>\n")
			}
		}
		// open the assistant turn so the SLM continues it; the tool contract
		// rides in the context so the routed SLM can ask for tools (the
		// gateway executes them and loops with the instruct brain).
		sb.WriteString(chatToolBlock())
		sb.WriteString("<|im_start|>assistant\n")
		nPredict := creq.MaxTokens
		if nPredict <= 0 {
			nPredict = creq.MaxCompletionTokens
		}
		if nPredict <= 0 {
			nPredict = 256
		}
		// potato cap: 0.6B-4B SLMs ramble; 640 gives the baked-in think block
		// room to close (~250-400 tokens) with ~250 left for the answer.
		// Every extra hundred tokens is ~5-10s of wall time at these speeds.
		if nPredict > 640 {
			nPredict = 640
		}
		// potato budget: the fleet backends run -c 4096, and omp ships a
		// ~14k-token system prompt. Truncate from the FRONT (the tail holds
		// the recent turns + the assistant opener the SLM must continue);
		// the router still scopes on the full last user message.
		prompt := strings.TrimSpace(sb.String())
		// no chaining, no silent truncation: when the transcript no longer
		// fits the backend window, signal the client (omp restarts the
		// context from its summary cache; retrieveSegments below rehydrates
		// past summaries on demand).
		if estTokens(prompt) > backendWindow {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Gotato-Context-Exhausted", "1")
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			_, _ = w.Write([]byte(`{"error":{"message":"context exhausted","code":"GOTATO_CONTEXT_EXHAUSTED","hint":"restart the context with a summary cache and resend"}}`))
			return
		}
		// summary-store retrieval: if the current request matches a past
		// segment's preview, rehydrate its full summary (RAM or gz) so a
		// "what did we decide about X" lands on the right past point.
		for _, seg := range retrieveSegments(session, lastUser, 2) {
			full := loadSegment(session, seg)
			if full != "" {
				prompt = fmt.Sprintf("[PAST SEGMENT seg-%d: %s]\n%s\n\n", seg.ID, seg.Preview, full) + prompt
			}
		}
		req := completionReq{Prompt: prompt, NPredict: nPredict,
			Session: session, RoutePrompt: lastUser, Stream: creq.Stream, SkipBridge: true}
		plan := routeCompletion(cfg, &req)
		if plan.override != "" {
			// escalation / delegation text arrives as a normal assistant reply
			writeChatCompletion(w, plan.override, "router", 0, 0)
			return
		}

		cfg.setActive(slmBaseName(plan.target, plan.servedLang), plan.servedLang)
		defer cfg.clearActive()
		t0 := time.Now()
		resp, err := httpClient.Post(plan.target+"/completion", "application/json", strings.NewReader(plan.fwdBody))
		if err != nil {
			http.Error(w, `{"error":{"message":"backend unreachable: `+plan.target+`"}}`, 502)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("X-Gotato-Backend", plan.target)
		w.Header().Set("X-Gotato-SLM", plan.slmTag)
		if strings.Contains(plan.fwdBody, "\"stream\":true") ||
			strings.Contains(plan.fwdBody, "\"stream\": true") {
			streamRelayChat(cfg, w, resp, &req, plan.target, plan.slmTag, t0, plan.servedLang)
			return
		}
		out, _ := io.ReadAll(resp.Body)
		var cr completionResp
		_ = json.Unmarshal(out, &cr)
		var respTimings struct {
			Timings struct {
				PredictedN int `json:"predicted_n"`
				PromptN   int `json:"prompt_n"`
			} `json:"timings"`
		}
		_ = json.Unmarshal(out, &respTimings)
		tokens := respTimings.Timings.PredictedN
		promptTokens := respTimings.Timings.PromptN
		appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
			SLM: "backend(" + plan.target + ")", Lang: plan.servedLang,
			Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
		appendContent(req.Session, "user", plan.origPrompt)
		content := cr.Content
		if bridgeZH && content != "" {
			content = translateToEN(cfg, content)
		}
		// tool loop (non-stream): the routed SLM asked for tools -> run the
		// instruct-brain loop and answer with the final result. The tool
		// transcript rides in the reply so the user SEES what was executed.
		if m := toolCallRe.FindStringSubmatch(cr.Content); m != nil {
			var final string
			var transcript string
			final, transcript = coworkTurn(toolBrain(cfg), session, lastUser, false)
			var tb strings.Builder
			tb.WriteString("\n[executed tool calls]\n")
			tb.WriteString(transcript)
			tb.WriteString(final)
			content = tb.String()
		}
		if stripped := stripThink(content); stripped != "" {
			content = stripped // a fully-think reply stays visible: an empty
			// message reads as empty-stop to omp and triggers its retry loop
		}
		if content != "" {
			appendContent(req.Session, "assistant", content)
		}
		writeChatCompletion(w, content, plan.slmTag, promptTokens, tokens)
	}
}

// writeChatCompletion: OpenAI chat-completion JSON, model = the SLM tag.
func writeChatCompletion(w http.ResponseWriter, content string, model string, promptTokens int, tokens int) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      "chatcmpl-gotato",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": "stop",
		}},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": tokens,
			"total_tokens":      promptTokens + tokens,
		},
	})
}

// streamRelayChat: llama /completion SSE -> OpenAI chat.completion.chunk
// SSE, with the same ledger/archive/position bookkeeping as streamRelay.
// Frames are transformed and flushed INCREMENTALLY - buffering the whole
// backend stream would stall OpenAI clients (omp aborts after ~15s with
// zero bytes).
func streamRelayChat(cfg *serveConfig, w http.ResponseWriter, resp *http.Response, req *completionReq,
	target string, slmTag string, t0 time.Time, servedLang string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Gotato-Backend", target)
	w.Header().Set("X-Gotato-SLM", slmTag)
	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}
	id := "chatcmpl-gotato"
	created := time.Now().Unix()
	var content strings.Builder
	var reasoningText strings.Builder
	var emittedContent bool = false
	var tokens int = 0
	var promptTokens int = 0
	inThink := false // <think> spans chunks; content inside routes to reasoning
	var toolTail strings.Builder
	var toolCalls []string
	var toolSeen bool = false
	// emitContentDelta / emitReasoningDelta: OpenAI chunk writers.
	emitContentDelta := func(text string) {
		if text == "" {
			return
		}
		emittedContent = true
		chunk := fmt.Sprintf(`{"id":%s,"object":"chat.completion.chunk","created":%d,"model":%s,"choices":[{"index":0,"delta":{"content":%s},"finish_reason":null}]}`,
			jsonString(id), created, jsonString(slmTag), jsonString(text))
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		if ok {
			flusher.Flush()
		}
	}
	emitReasoningDelta := func(text string) {
		if text == "" {
			return
		}
		reasoningText.WriteString(text)
		chunk := fmt.Sprintf(`{"id":%s,"object":"chat.completion.chunk","created":%d,"model":%s,"choices":[{"index":0,"delta":{"reasoning_content":%s},"finish_reason":null}]}`,
			jsonString(id), created, jsonString(slmTag), jsonString(text))
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		if ok {
			flusher.Flush()
		}
	}
	sc := bufio.NewScanner(resp.Body)
	sbuf := getScanBuf()
	defer putScanBuf(sbuf)
	sc.Buffer(sbuf, 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		var frame struct {
			Content string `json:"content"`
			Stop    bool   `json:"stop"`
			Timings struct {
				PredictedN int `json:"predicted_n"`
				PromptN   int `json:"prompt_n"`
			} `json:"timings"`
		}
		if json.Unmarshal([]byte(strings.TrimSpace(line[5:])), &frame) != nil {
			continue
		}
		if frame.Content != "" {
			// route <think>…</think> to delta.reasoning_content instead of
			// dropping it: OMP renders reasoning as its Thinking block, and a
			// fully-stripped reply would read as empty-stop and trigger
			// omp's retry loop (the empty-stop bug).
			var text string = frame.Content
			for {
				var i int = strings.Index(text, "<think>")
				if i < 0 {
					break
				}
				if i > 0 {
					content.WriteString(text[:i])
					emitContentDelta(text[:i])
				}
				var j int = strings.Index(text[i:], "</think>")
				if j < 0 {
					text = text[i+len("<think>"):]
					inThink = true
					break
				}
				text = text[i+len("<think>") : i+j]
				emitReasoningDelta(text)
				text = frame.Content[i+j+len("</think>"):]
			}
			if inThink {
				// remainder of a chunk inside an open think block
				if j := strings.Index(text, "</think>"); j >= 0 {
					emitReasoningDelta(text[:j])
					text = text[j+len("</think>"):]
					inThink = false
				} else {
					emitReasoningDelta(text)
					continue
				}
			}
						// suppress <tool_call> spans (stateful across chunks): the call
			// JSON never reaches the client; at stream end the gateway runs
			// the tool loop with the instruct brain and streams the result.
			// Tolerates BOTH the contract form (<tool_call>…</tool_call>) and
			// the bare JSON the tiny models actually emit.
			toolTail.WriteString(text)
			for {
				tail := toolTail.String()
				wi := strings.Index(tail, "<tool_call>")
				bi := -1
				for _, tn := range toolNames {
					if k := strings.Index(tail, `{"name":"`+tn+`","arguments":{`); k >= 0 && (bi < 0 || k < bi) {
						bi = k
					}
				}
				var hold int = -1
				var wend int = -1
				if wi >= 0 {
					if j := strings.Index(tail[wi:], "</tool_call>"); j >= 0 {
						wend = wi + j + len("</tool_call>")
					} else {
						hold = wi
					}
				}
				if bi >= 0 && (wi < 0 || bi < wi) {
					if j := braceClose(tail, bi); j >= 0 {
						wend = j
					} else {
						hold = bi
					}
				}
				if wend >= 0 {
					start := wi
					if wi < 0 || (bi >= 0 && bi < wi) {
						start = bi
					}
					if start > 0 {
						content.WriteString(tail[:start])
						emitContentDelta(tail[:start])
					}
					toolSeen = true
					toolCalls = append(toolCalls, tail[start:wend])
					toolTail.Reset()
					toolTail.WriteString(tail[wend:])
					continue
				}
				// no complete call: hold a partial-call suffix (a chunk may
				// have cut the pattern mid-way) before emitting anything
				if hold < 0 {
					hold = partialCallHold(tail)
				}
				if hold >= 0 {
					if hold > 0 {
						content.WriteString(tail[:hold])
						emitContentDelta(tail[:hold])
					}
					toolTail.Reset()
					toolTail.WriteString(tail[hold:])
					break
				}
				if tail != "" {
					content.WriteString(tail)
					emitContentDelta(tail)
				}
				toolTail.Reset()
				break
			}
		}
		if frame.Stop {
			tokens = frame.Timings.PredictedN
			promptTokens = frame.Timings.PromptN
			final := fmt.Sprintf(`{"id":%s,"object":"chat.completion.chunk","created":%d,"model":%s,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`,
				jsonString(id), created, jsonString(slmTag), promptTokens, tokens, promptTokens+tokens)
			fmt.Fprintf(w, "data: %s\n\n", final)
		}
	}
	// tool loop: the routed SLM asked for tools -> execute with the
	// instruct brain (the proven cowork model) and stream the final answer.
	if toolSeen || (toolTail.Len() > 0 && strings.Contains(toolTail.String(), `{"name":"`)) {
		var transcript string
		var final string
		final, transcript = coworkTurn(toolBrain(cfg), req.Session, req.RoutePrompt, false)
		var n int = len(toolCalls)
		if strings.Count(toolTail.String(), `{"name":"`) > 0 {
			n++
		}
		var marker string = fmt.Sprintf("\n[executed %d tool call(s)]\n", n)
		content.WriteString(marker)
		emitContentDelta(marker)
		// the transcript is streamed too: omp sees what the tools did
		// (list_dir/read_file/run_command + the result sizes) before the
		// final answer.
		if transcript != "" {
			emitContentDelta(transcript)
			content.WriteString(transcript)
		}
		content.WriteString(final)
		emitContentDelta(final)
		if transcript != "" {
			appendContent(req.Session, "system", "TOOLS: "+transcript)
		}
	}
	// empty-stop guard: if the ENTIRE reply was thinking (unclosed <think>
	// burned the whole budget), the reasoning was already streamed to the
	// client's Thinking block; end with a continuation hint instead of an
	// empty stop (which triggers omp's retry loop) or a ramble dump.
	if !emittedContent && reasoningText.Len() > 0 {
		var fallback string = "\n[the model spent its whole budget thinking - say 'continue' to get the answer]"
		content.WriteString(fallback)
		emitContentDelta(fallback)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if ok {
		flusher.Flush()
	}
	appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
		SLM: "backend(" + target + ")", Lang: servedLang,
		Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
	appendContent(req.Session, "user", req.Prompt)
	if content.Len() > 0 {
		appendContent(req.Session, "assistant", content.String())
	}
}

// modelsHandler: GET /v1/models - advertise the fleet to OpenAI-shaped
// clients. Every roster member is listed plus the gateway alias.
// stackManifest: read the CURRENT stack's per-stack SLM manifest (the
// gateway's cwd IS the stack). Returns nil when missing. omp-potato
// reads it via GET /manifest to know which SLMs to launch.
func stackManifest() map[string]*stackLangEntry {
	var wd string
	var err error
	wd, err = os.Getwd()
	if err != nil {
		return nil
	}
	var base string = filepath.Base(strings.TrimRight(wd, "/"))
	var path string = fleetDir + "/stacks/" + base + ".json"
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var st struct {
		Stack     string                     `json:"stack"`
		Languages map[string]*stackLangEntry `json:"languages"`
	}
	if json.Unmarshal(data, &st) != nil {
		return nil
	}
	return st.Languages
}

func modelsHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		// omp-potato sees ONLY this stack's SLMs (manifest-aware): the
		// model list IS the per-stack manifest's languages.
		var langs []string = stackManifestLangs()
		data := make([]map[string]any, 0, len(langs)+1)
		for _, lang := range langs {
			data = append(data, map[string]any{
				"id":       "slm-" + lang,
				"object":   "model",
				"owned_by": "gotatoqwen",
			})
		}
		data = append(data, map[string]any{"id": "gotato-gateway", "object": "model", "owned_by": "gotatoqwen"})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}
}

// manifestHandler: GET /manifest - the stack's SLM manifest with the full
// launch recipes (base, lora, ctx, threads, window) so omp-potato can
// start the stack's SLMs itself.
func manifestHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet || r.URL.Path != "/manifest" {
		http.NotFound(w, r)
		return
	}
	var wd string
	var err error
	wd, err = os.Getwd()
	if err != nil {
		http.Error(w, `{"error":"no stack"}`, 500)
		return
	}
	var base string = filepath.Base(strings.TrimRight(wd, "/"))
	var path string = fleetDir + "/stacks/" + base + ".json"
	data, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, `{"error":"no manifest for `+base+` - run expertd stacks `+wd+`"}`, 404)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

// slmBaseName: the stable SLM identity for a backend target (no topic
// suffix). Used by the roster endpoint and the display tag alike so the
// TUI can key chips on one canonical name.
func slmBaseName(target string, lang string) string {
	var base string = "slm"
	switch {
	case strings.Contains(target, ":8081"):
		base = "python-expert"
	case strings.Contains(target, ":8082"):
		base = "2b-general"
	case strings.Contains(target, ":8083"):
		base = "4b-general"
	case strings.Contains(target, ":8086"):
		base = "instruct"
	default:
		if lang != "" {
			base = lang + "-slice"
		}
	}
	return base
}

// slmDisplayTag: the human-facing SLM name with a topic suffix, e.g.
// "python-expert · utils" or "rust-slice · rust". Shown by the TUI next to
// its Thinking indicator and sent as the X-Gotato-SLM header.
func slmDisplayTag(target string, lang string, label string, owner string) string {
	base := slmBaseName(target, lang)
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
	target string, slmTag string, t0 time.Time, servedLang string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Gotato-Backend", target)
	w.Header().Set("X-Gotato-SLM", slmTag)
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
	sc := bufio.NewScanner(strings.NewReader(acc.String()))
	sbuf := getScanBuf()
	defer putScanBuf(sbuf)
	sc.Buffer(sbuf, 1<<20)
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
		}
	}
	appendTurn(turn{Session: req.Session, Ts: time.Now().UnixMilli(),
		SLM: "backend(" + target + ")", Lang: servedLang,
		Tokens: tokens, WallMs: time.Since(t0).Milliseconds()})
	appendContent(req.Session, "user", req.Prompt)
	if content.Len() > 0 {
		appendContent(req.Session, "assistant", content.String())
	}
}
