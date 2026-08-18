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

// route: the forwarding plan produced by the scope protocol. Either the
// backend to call (target/fwdBody/...) or a direct-answer override
// (escalation question, delegation confirmation, missing-slice prompt).
type routePlan struct {
	target         string
	slmTag         string
	servedLang     string
	chained        bool
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
	// lexical scope signal: when the index has no opinion but the prompt
	// is clearly code in a language that isn't the session owner's, treat
	// it as out of scope too (the "pasted a Go element" case).
	var lexLang string = ""
	var lexHits int = 0
	if semanticLang == "" && looksLikeCode(routeText) {
		var lexConf float64 = 0
		lexLang, lexConf, lexHits = detectLanguageN(routeText, "")
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
		looksLikeCode(routeText))

	// branch 1: out-of-scope vs the session owner -> ask the user
	if ownerLang != "" && scopeLang != "" && scopeLang != ownerLang {
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

	// ---- context chaining: approach the cap -> summarize + fresh context
	var chainCap int = defaultChainCap
	if req.ChainCap > 0 {
		chainCap = req.ChainCap
	}
	prefix, chained := chainContext(cfg, req.Session, req.Prompt, chainCap)
	var fwdBody string = ""
	{
		rawBody, _ := json.Marshal(req)
		fwdBody = string(rawBody)
	}
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
		resp, err := http.Post(plan.target+"/completion", "application/json", strings.NewReader(plan.fwdBody))
		if err != nil {
			http.Error(w, `{"error":"backend unreachable: `+plan.target+`"}`, 502)
			return
		}
		defer resp.Body.Close()
		if strings.Contains(plan.fwdBody, "\"stream\":true") ||
			strings.Contains(plan.fwdBody, "\"stream\": true") {
			streamRelay(w, resp, &req, plan.target, plan.slmTag, plan.chained, t0, plan.servedLang)
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
			if plan.chained {
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
		w.Header().Set("X-Gotato-Backend", plan.target)
		w.Header().Set("X-Gotato-SLM", plan.slmTag)
		if plan.chained {
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
	mux.HandleFunc("/slms", slmsHandler(cfg))
	mux.HandleFunc("/v1/chat/completions", chatHandler(cfg))
	mux.HandleFunc("/v1/models", modelsHandler(cfg))
	fmt.Printf("[gateway] listening on %s | backends: %v\n", cfg.addr, cfg.backends)
	if err := http.ListenAndServe(cfg.addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// ledgerUsesByTarget: how many turns each backend served, from the session
// ledger. Keys are backend URLs ("http://127.0.0.1:8081"). Both the
// "backend(...)" and "autostart(...)" turn shapes contribute.
func ledgerUsesByTarget() map[string]int {
	uses := map[string]int{}
	f, err := os.Open(sessionsPath)
	if err != nil {
		return uses
	}
	defer f.Close()
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
				uses[url]++
			}
		}
	}
	return uses
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
		langs := make([]string, 0, len(backends))
		for lang := range backends {
			langs = append(langs, lang)
		}
		sort.Strings(langs)

		type slmEntry struct {
			Name   string `json:"name"`
			Lang   string `json:"lang"`
			Port   string `json:"port"`
			Target string `json:"target"`
			Uses   int    `json:"uses"`
			Used   bool   `json:"used"`
		}
		roster := make([]slmEntry, 0, len(langs))
		for _, lang := range langs {
			url := backends[lang]
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
		// open the assistant turn so the SLM continues it
		sb.WriteString("<|im_start|>assistant\n")
		nPredict := creq.MaxTokens
		if nPredict <= 0 {
			nPredict = creq.MaxCompletionTokens
		}
		if nPredict <= 0 {
			nPredict = 256
		}
		// potato cap: a 0.6B-4B SLM rambles past 512 tokens; every extra
		// hundred tokens is ~10s of wall time at these speeds.
		if nPredict > 512 {
			nPredict = 512
		}
		// potato budget: the fleet backends run -c 4096, and omp ships a
		// ~14k-token system prompt. Truncate from the FRONT (the tail holds
		// the recent turns + the assistant opener the SLM must continue);
		// the router still scopes on the full last user message.
		prompt := strings.TrimSpace(sb.String())
		const maxPromptChars = 3600 * 4 // ~3600 tokens at the len/4 estimate
		if len(prompt) > maxPromptChars {
			prompt = "[gateway: earlier context truncated to fit the 4k window]\n" +
				prompt[len(prompt)-maxPromptChars:]
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
		resp, err := http.Post(plan.target+"/completion", "application/json", strings.NewReader(plan.fwdBody))
		if err != nil {
			http.Error(w, `{"error":{"message":"backend unreachable: `+plan.target+`"}}`, 502)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("X-Gotato-Backend", plan.target)
		w.Header().Set("X-Gotato-SLM", plan.slmTag)
		if strings.Contains(plan.fwdBody, "\"stream\":true") ||
			strings.Contains(plan.fwdBody, "\"stream\": true") {
			streamRelayChat(w, resp, &req, plan.target, plan.slmTag, plan.chained, t0, plan.servedLang)
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
		content = stripThink(content)
		if content != "" {
			appendContent(req.Session, "assistant", content)
		}
		if promptTokens > 0 {
			addPosition(req.Session, promptTokens+tokens)
		} else {
			addPosition(req.Session, estTokens(req.Prompt)+tokens)
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
func streamRelayChat(w http.ResponseWriter, resp *http.Response, req *completionReq,
	target string, slmTag string, chained bool, t0 time.Time, servedLang string) {
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
	var tokens int = 0
	var promptTokens int = 0
	inThink := false // <think> spans chunks; drop everything until </think>
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
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
			if inThink {
				if j := strings.Index(frame.Content, "</think>"); j >= 0 {
					frame.Content = frame.Content[j+len("</think>"):]
					inThink = false
				} else {
					frame.Content = ""
				}
			}
			if i := strings.Index(frame.Content, "<think>"); i >= 0 && !inThink {
				if j := strings.Index(frame.Content[i:], "</think>"); j >= 0 {
					frame.Content = frame.Content[:i] + frame.Content[i+j+len("</think>"):]
				} else {
					frame.Content = frame.Content[:i]
					inThink = true
				}
			}
			if frame.Content == "" {
				continue
			}
			content.WriteString(frame.Content)
			chunk := fmt.Sprintf(`{"id":%s,"object":"chat.completion.chunk","created":%d,"model":%s,"choices":[{"index":0,"delta":{"content":%s},"finish_reason":null}]}`,
				jsonString(id), created, jsonString(slmTag), jsonString(frame.Content))
			fmt.Fprintf(w, "data: %s\n\n", chunk)
			if ok {
				flusher.Flush()
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
	if promptTokens > 0 {
		addPosition(req.Session, promptTokens+tokens)
	} else {
		addPosition(req.Session, estTokens(req.Prompt)+tokens)
	}
}

// modelsHandler: GET /v1/models - advertise the fleet to OpenAI-shaped
// clients. Every roster member is listed plus the gateway alias.
func modelsHandler(cfg *serveConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		cfg.mu.Lock()
		backends := make(map[string]string, len(cfg.backends))
		for k, v := range cfg.backends {
			backends[k] = v
		}
		cfg.mu.Unlock()
		langs := make([]string, 0, len(backends))
		for lang := range backends {
			langs = append(langs, lang)
		}
		sort.Strings(langs)
		data := make([]map[string]any, 0, len(langs)+1)
		for _, lang := range langs {
			url := backends[lang]
			data = append(data, map[string]any{
				"id":      slmBaseName(url, lang),
				"object":  "model",
				"owned_by": "gotatoqwen",
			})
		}
		data = append(data, map[string]any{"id": "gotato-gateway", "object": "model", "owned_by": "gotatoqwen"})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}
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
