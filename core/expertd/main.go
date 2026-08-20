// expertd - deterministic core of the SLM-fleet (Go, no GC, no :=).
//
// Replaces the Python orchestration layer: stack scanning, language
// detection, capability index, watcher daemon, and router dispatch.
// Compute stays in C (llama.cpp, torch) - this binary only moves bytes
// and decisions, which is exactly what belongs in a no-GC static binary.
//
// Disciplines:
//   - runtime/debug.SetGCPercent(-1): no garbage collector, ever
//   - zero short-variable declarations (no :=): every binding is typed
//   - stdlib only: no fsnotify, no cobra, no dependencies to drift
//   - index.json schema is byte-compatible with the Python prototype
//
// Usage:
//
//	expertd scan <dir>                 print {lang: files, signature} as JSON
//	expertd detect <file|text>         print detected language + confidence
//	expertd watch <dir>                daemon: scan, index, spawn builder
//	expertd route <file|text> [-n N]   detect -> index -> exec llama-cli
//	expertd bench <dir>                repeated scan timing
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var fleetDir string = "/home/pipo/slm-fleet"
var indexPath string = fleetDir + "/index.json"
var llamaCli string = "/home/pipo/llama.cpp/build/bin/llama-cli"
var ggloraBin string = "" // resolved in applyEnv: repo core/gglora/gglora

// applyEnv: the agent-setup contract. Every machine-specific path can be
// overridden by environment variables, so a fresh checkout needs no edits:
//
//	GOTATO_FLEET     fleet dir (models, adapters, index, sessions, sub-index)
//	GOTATO_LLAMA_BIN llama.cpp build/bin dir (llama-cli, llama-server, ...)
//	GOTATO_GGLORA    the Go LoRA trainer binary (default: <repo>/core/gglora/gglora)
//
// No python anywhere: corpus collect, LoRA training and adapter writing are
// all Go (expertd build + gglora train).
func applyEnv() {
	var v = os.Getenv("GOTATO_FLEET")
	if v != "" {
		fleetDir = v
		indexPath = fleetDir + "/index.json"
		sessionsPath = fleetDir + "/sessions.jsonl"
	}
	{
		var v = os.Getenv("GOTATO_LLAMA_BIN")
		if v != "" {
			llamaCli = v + "/llama-cli"
		}
	}
	{
		var v = os.Getenv("GOTATO_GGLORA")
		if v != "" {
			ggloraBin = v
		}
	}
	if ggloraBin == "" {
		var self, err = os.Executable()
		if err == nil {
			ggloraBin = filepath.Join(filepath.Dir(self), "..", "gglora", "gglora")
		}
	}
}

// ---- language map: extension -> language --------------------------------
var extToLang map[string]string = map[string]string{
	".py": "python", ".pyw": "python", ".pyi": "python",
	".ts": "typescript", ".tsx": "typescript",
	".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
	".rs": "rust", ".go": "go", ".rb": "ruby", ".java": "java",
	".kt": "kotlin", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp",
	".hpp": "cpp", ".cs": "csharp", ".php": "php", ".swift": "swift",
}

var langSignals map[string][]*regexp.Regexp = map[string][]*regexp.Regexp{
	"python": {regexp.MustCompile(`(?m)^\s*(def |class |import |from |async def )`),
		regexp.MustCompile(`print\(.*\)`), regexp.MustCompile(`self\.`),
		regexp.MustCompile(`if __name__`)},
	"typescript": {regexp.MustCompile(`(interface |type |const .*: |: string|: number)`),
		regexp.MustCompile(`import .* from ['"]`), regexp.MustCompile(`export (default )?(function|class|const)`)},
	"javascript": {regexp.MustCompile(`(function |const .* = \(.*\) =>|require\(|module\.exports)`),
		regexp.MustCompile(`console\.log\(|=>\s*\{`)},
	"rust": {regexp.MustCompile(`(?m)^\s*(fn |let mut |impl |pub |use )`),
		regexp.MustCompile(`println!\(|unwrap\(\)|\.await`)},
	"go": {regexp.MustCompile(`(?m)^\s*(package \w+|func |import |var |const |type )`),
		regexp.MustCompile(`:=`), regexp.MustCompile(`fmt\.|\w+\.\w+\(`),
		regexp.MustCompile(`\*\w+|\[\]\w+`)},
}

// detectLanguageN: detectLanguage plus the number of pattern hits (a
// language is trusted when >= 2 distinct patterns fire; single weak hits
// like "type hints" in English text are rejected).
func detectLanguageN(text string, path string) (string, float64, int) {
	var lang, conf = detectLanguage(text, path)
	var head = text
	if len(head) > 4000 {
		head = head[:4000]
	}
	var hits = 0
	var p *regexp.Regexp

	for _, p = range langSignals[lang] {
		if p.MatchString(head) {
			hits++
		}
	}
	return lang, conf, hits
}

// looksLikeCode: crude gate - does this text contain code markers? English
// task descriptions fail this, pasted code snippets pass it.
func looksLikeCode(text string) bool {
	var markers = []string{"package ", "func ", "def ", "fn ", "import ",
		"const ", "var ", "let ", "pub ", "use ", "impl ", "class ",
		"#include", "using ", ":=", "->", "::"}
	var head string = text
	if len(head) > 6000 {
		head = head[:6000]
	}
	var m string

	for _, m = range markers {
		if strings.Contains(head, m) {
			return true
		}
	}
	return false
}

// ---- index schema (byte-compatible with the Python prototype) ----------
type expertEntry struct {
	Status    string  `json:"status"`
	Signature string  `json:"signature"`
	Files     int     `json:"files"`
	PID       int     `json:"pid"`
	ErrorAt   float64 `json:"error_at,omitempty"`
	Error     string  `json:"error,omitempty"`
	Lora      string  `json:"lora,omitempty"`
	// Mask: a sliced GGUF (ggslice output) - a full model file with the
	// off-domain heads/neurons zeroed; served with -m, no adapter needed.
	Mask         string  `json:"mask,omitempty"`
	Kind         string  `json:"kind,omitempty"` // "mask" | "adapter"
	Base         string  `json:"base,omitempty"`
	TrainedAt    float64 `json:"trained_at,omitempty"`
	TrainSeconds float64 `json:"train_seconds,omitempty"`
}

type indexFile map[string]expertEntry

type scanResult struct {
	Lang      string   `json:"lang"`
	Files     []string `json:"files"`
	Signature string   `json:"signature"`
}

// ---- scan ---------------------------------------------------------------
func scanDir(stack string) []scanResult {
	var out []scanResult = make([]scanResult, 0)
	var byLang map[string]*scanResult = make(map[string]*scanResult)
	var skipDirs map[string]bool = map[string]bool{".git": true, "node_modules": true, "__pycache__": true}

	filepath.WalkDir(stack, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		var ext = strings.ToLower(filepath.Ext(path))
		var lang, ok = extToLang[ext]
		if !ok {
			return nil
		}
		var info fs.FileInfo

		info, err = d.Info()
		if err != nil {
			return nil
		}
		if info.Size() < 80 || info.Size() > 400*1024 {
			return nil
		}
		var entry *scanResult

		entry, ok = byLang[lang]
		if !ok {
			entry = &scanResult{Lang: lang, Files: make([]string, 0, 8)}
			byLang[lang] = entry
		}
		entry.Files = append(entry.Files, path)
		return nil
	})

	var langs []string = make([]string, 0, len(byLang))
	var lang string

	for lang = range byLang {
		langs = append(langs, lang)
	}
	sort.Strings(langs)
	{
		var lang string

		for _, lang = range langs {
			var entry = byLang[lang]
			sort.Strings(entry.Files)
			var h hash.Hash = sha256.New()
			var p string

			for _, p = range entry.Files {
				h.Write([]byte(p))
				var st syscall.Stat_t
				if syscall.Stat(p, &st) == nil {
					var secs float64 = float64(st.Mtim.Sec) + float64(st.Mtim.Nsec)/1e9
					h.Write([]byte(strconv.FormatFloat(secs, 'f', -1, 64)))
				}
			}
			entry.Signature = hex.EncodeToString(h.Sum(nil))[:16]
			out = append(out, *entry)
		}
	}
	return out
}

// ---- detect -------------------------------------------------------------
func detectLanguage(text string, path string) (string, float64) {
	if path != "" {
		var ext = strings.ToLower(filepath.Ext(path))
		var lang, ok = extToLang[ext]
		if ok {
			return lang, 0.9
		}
	}
	var head string = text
	if len(head) > 4000 {
		head = head[:4000]
	}
	var scores map[string]int = make(map[string]int)
	var lang string

	var pats []*regexp.Regexp

	for lang, pats = range langSignals {
		var n int = 0
		var p *regexp.Regexp

		for _, p = range pats {
			if p.MatchString(head) {
				n++
			}
		}
		scores[lang] = n
	}
	var best string = "default"
	var bestN int = 0
	var total int = 0
	{
		var lang string

		var n int

		for lang, n = range scores {
			total += n
			if n > bestN {
				bestN = n
				best = lang
			}
		}
	}
	if total == 0 {
		return "default", 0.1
	}
	return best, float64(bestN) / float64(total)
}

func resolveIndex(idxPath string, text string, topN int) []hit {
	var idx, err = loadSubIndex(idxPath)
	if err != nil {
		return nil
	}
	return idx.resolve(text, topN)
}

// usedPlaceholder: no model was invoked yet (escalation turns).
func usedPlaceholder() string {
	return "none"
}

// ---- index --------------------------------------------------------------
func loadIndex() indexFile {
	var idx indexFile = make(indexFile)
	var data, err = os.ReadFile(indexPath)
	if err != nil {
		return idx
	}
	_ = json.Unmarshal(data, &idx)
	return idx
}

func saveIndex(idx indexFile) {
	var data, err = json.MarshalIndent(idx, "", " ")
	if err != nil {
		return
	}
	var tmp string = indexPath + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		_ = os.Rename(tmp, indexPath)
	}
}

func isAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	var err = syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}

// ---- watch --------------------------------------------------------------
func watch(stack string, interval time.Duration, once bool) {
	fmt.Printf("[expertd] watching %s | index %s | GC off\n", stack, indexPath)
	for {
		var idx = loadIndex()
		var scans = scanDir(stack)
		var sc scanResult

		for _, sc = range scans {
			var entry, ok = idx[sc.Lang]
			if !ok {
				entry = expertEntry{}
			}
			if entry.Status == "building" && isAlive(entry.PID) {
				continue
			}
			if entry.Status == "error" && entry.ErrorAt > float64(time.Now().Unix())-300 {
				continue
			}
			if entry.Status == "ready" && entry.Signature == sc.Signature {
				continue
			}
			entry.Status = "building"
			entry.Signature = sc.Signature
			entry.Files = len(sc.Files)
			entry.ErrorAt = 0
			idx[sc.Lang] = entry
			saveIndex(idx)

			var logPath = fleetDir + "/build_" + sc.Lang + ".log"
			var logFile, err = os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
			if err != nil {
				continue
			}
			// the builder is this same binary: `expertd build` collects the
			// corpus, trains with gglora (Go, no python) and publishes index.json
			var self string

			self, _ = os.Executable()
			var cmd = exec.Command("nice", "-n", "10", self, "build", sc.Lang, stack)
			cmd.Stdout = logFile
			cmd.Stderr = logFile
			cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
			{
				var err = cmd.Start()
				if err == nil {
					entry.PID = cmd.Process.Pid
					idx[sc.Lang] = entry
					saveIndex(idx)
					_ = logFile.Close()
					fmt.Printf("[expertd] %s: %d files changed -> builder pid %d (2 threads, nice 10)\n",
						sc.Lang, len(sc.Files), cmd.Process.Pid)
				}
			}
		}
		if once {
			return
		}
		time.Sleep(interval)
	}
}

// ---- route --------------------------------------------------------------
func route(target string, gen int, session string, scopeCheckOn bool) int {
	var text string = ""
	var path string = ""
	{
		var st, err = os.Stat(target)
		if err == nil && !st.IsDir() {
			path = target
			var data, err = os.ReadFile(target)
			if err == nil {
				text = string(data)
			}
		} else {
			text = target
		}
	}
	var lang, conf = detectLanguage(text, path)

	// tier-2 outranks tier-1 ALWAYS (when the index exists): semantics beat
	// lexical detection on strong margins. --scope-check additionally gates
	// the session-owner escalation protocol.
	var idxPath string = fleetDir + "/subindex.json"
	var semanticLang string = ""
	var semanticMargin float64 = 0
	var hits = resolveIndex(idxPath, text, 2)
	if len(hits) > 0 {
		var top = hits[0]
		if len(hits) > 1 && hits[1].Score > 0 {
			semanticMargin = top.Score / hits[1].Score
		} else {
			semanticMargin = 2.0
		}
		if semanticMargin >= 2.0 {
			semanticLang = top.Lang
		}
	}

	// session ownership: the SLM that served the previous turn owns the session
	var ownerLang string = ""
	{
		var prev, ok = lastSessionTurn(session)
		if ok {
			ownerLang = prev.Lang
		}
	}
	if scopeCheckOn && semanticLang != "" && ownerLang != "" && semanticLang != ownerLang {
		// ask before switching - the out-of-scope protocol
		var scopeEv string = "out-of-scope->" + semanticLang
		var escalation string = fmt.Sprintf(
			"destination hit out of scope - shall we delegate an SLM for %s?",
			semanticLang)
		var t turn = turn{Session: session, Ts: time.Now().UnixMilli(),
			SLM: usedPlaceholder(), Lang: ownerLang, Tokens: 0, WallMs: 0,
			ScopeEvent: scopeEv, Escalation: escalation}
		appendTurn(t)
		fmt.Printf("[router] session owner=%s | semantic hit=%s (margin %.1fx)\n",
			ownerLang, semanticLang, semanticMargin)
		fmt.Printf("[router] >>> %s\n", escalation)
		return 0
	}
	if semanticLang != "" {
		lang = semanticLang
		conf = semanticMargin
	}

	var idx = loadIndex()
	var entry, ok = idx[lang]
	var model string = ""
	var lora string = ""
	var used string = "generalist"
	if ok && entry.Status == "ready" && entry.Lora != "" {
		model = fleetDir + "/" + entry.Base
		lora = fleetDir + "/" + entry.Lora
		used = "expert(" + entry.Lora + ")"
		fmt.Printf("[router] lang=%s conf=%.2f -> expert (%s + %s)\n", lang, conf, entry.Base, entry.Lora)
	} else {
		model = fleetDir + "/Qwen3.5-4B-Q4_K_M.gguf"
		fmt.Printf("[router] lang=%s conf=%.2f -> generalist\n", lang, conf)
	}
	{
		var err error

		_, err = os.Stat(model)
		if err != nil {
			fmt.Printf("[router] model missing: %s\n", model)
			return 1
		}
	}

	var args = []string{"-m", model, "-p", text, "-n", strconv.Itoa(gen), "-t", "4",
		"-c", "4096", "--temp", "0.3", "--log-disable", "-st"}
	if lora != "" {
		args = append(args, "--lora", lora)
	}
	var t0 = time.Now()
	var buf bytes.Buffer
	var cmd = exec.Command(llamaCli, args...)
	cmd.Stdout = &buf
	cmd.Stderr = os.Stderr
	var err = cmd.Run()
	var dt = time.Since(t0)

	// out-of-scope protocol: input-side (deterministic) then output-side (drift)
	var scopeEv string = ""
	var escalation string = ""
	if scopeCheckOn {
		var idxPath string = fleetDir + "/subindex.json"
		// the session's owner is the SLM that served the previous turn
		var ownerLang string = lang
		var prev, ok = lastSessionTurn(session)
		if ok {
			ownerLang = prev.Lang
		}
		scopeEv, escalation = scopeCheck(idxPath, text, ownerLang)
		if scopeEv == "" {
			scopeEv, escalation = scopeCheck(idxPath, stripTiming(buf.String()), lang)
		}
	}
	var t = turn{Session: session, Ts: time.Now().UnixMilli(), SLM: used,
		Lang: lang, Tokens: gen, WallMs: dt.Milliseconds(),
		ScopeEvent: scopeEv, Escalation: escalation}
	appendTurn(t)

	var out = buf.String()
	fmt.Print(out)
	fmt.Printf("\n[router] [%s] %d tokens in %.1fs\n", used, gen, dt.Seconds())
	if escalation != "" {
		fmt.Printf("\n[router] >>> %s\n", escalation)
	}
	if err != nil {
		fmt.Printf("[router] llama-cli failed: %v\n", err)
		return 1
	}
	return 0
}

// ---- bench --------------------------------------------------------------
func bench(stack string, rounds int) {
	var total time.Duration = 0
	var i int = 0
	for i = 0; i < rounds; i++ {
		var t0 time.Time = time.Now()
		scanDir(stack)
		total += time.Since(t0)
	}
	fmt.Printf("[expertd] scan(%s) x%d: avg %.2f ms\n", stack, rounds,
		float64(total.Milliseconds())/float64(rounds))
}

func main() {
	// Deterministic no-GC core: fine for the short-lived scan/route CLI
	// paths (3 MB RSS). The long-running servers (serve = the gateway,
	// mcp = the tool server) allocate per request and would grow without
	// bound - they NEED the GC, or the potato OOMs the gateway after a
	// few hundred requests.
	var cmd string = ""
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}
	if cmd != "serve" && cmd != "mcp" {
		debug.SetGCPercent(-1)
	}
	applyEnv()
	langCatalogInit()
	if len(os.Args) < 2 {
		fmt.Println("usage: expertd scan|detect|watch|route|bench|index|resolve|langs|oracle ...")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "build":
		buildCmd(os.Args[2:])
	case "langs":
		langsCmd(os.Args[2:])
	case "stacks":
		stacksCmd(os.Args[2:])
	case "oracle":
		oracleCmd(os.Args[2:])
	case "scan":
		var sc scanResult

		for _, sc = range scanDir(os.Args[2]) {
			fmt.Printf("%s: %d files, sig=%s\n", sc.Lang, len(sc.Files), sc.Signature)
		}
	case "detect":
		var text string = ""
		var path string = ""
		if len(os.Args) > 3 {
			text = os.Args[2]
		} else {
			path = os.Args[2]
		}
		var lang, conf = detectLanguage(text, path)
		fmt.Printf("lang=%s conf=%.2f\n", lang, conf)
	case "watch":
		var stack string = "/home/pipo/stack"
		var once bool = false
		var i = 2
		for i = 2; i < len(os.Args); i++ {
			if os.Args[i] == "--once" {
				once = true
			} else {
				stack = os.Args[i]
			}
		}
		watch(stack, 15*time.Second, once)
	case "route":
		var gen int = 120
		var session string = "default"
		var scopeCheckOn bool = false
		var target string = ""
		var i = 2
		for i = 2; i < len(os.Args); i++ {
			switch os.Args[i] {
			case "-n":
				if i+1 < len(os.Args) {
					gen, _ = strconv.Atoi(os.Args[i+1])
					i++
				}
			case "--session":
				if i+1 < len(os.Args) {
					session = os.Args[i+1]
					i++
				}
			case "--scope-check":
				scopeCheckOn = true
			default:
				target = os.Args[i]
			}
		}
		if target == "" {
			fmt.Println("usage: expertd route <file|text> [-n N] [--session ID] [--scope-check]")
			os.Exit(2)
		}
		os.Exit(route(target, gen, session, scopeCheckOn))
	case "sessions":
		sessionsCmd(os.Args[2:])
	case "serve":
		serveCmd(os.Args[2:])
	case "chat":
		chatCmd(os.Args[2:])
	case "mcp":
		// bundled MCP server: spawn and speak JSON-RPC on stdio.
		// run_command approval is env-gated: GOTATO_APPROVE=1 permits it.
		mcpServerLoop(os.Getenv("GOTATO_APPROVE") == "1")
	case "bench":
		bench(os.Args[2], 20)
	case "index":
		indexCmd(os.Args[2:])
	case "resolve":
		resolveCmd(os.Args[2:])
	default:
		io.WriteString(os.Stderr, "unknown command\n")
		os.Exit(2)
	}
}
