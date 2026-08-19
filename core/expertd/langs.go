// langs.go - the expanding language catalog + the 27B oracle.
//
// The problem: extToLang/langSignals are hardcoded, so a new programming
// language dropped into the stack is invisible to scan/index/watch/routing.
//
// The design: a language CATALOG (languages.json in the fleet) seeded with
// the builtin map. When `expertd oracle <stack>` finds files no catalog
// entry covers, it asks the Qwen3.8-27B oracle to classify them and return
// a strict JSON contract (name, extensions, regex signals, keywords). The
// oracle's analysis is registered in the catalog, so the very next watch
// cycle scans the new language, builds its LoRA expert (gglora train), and
// the router's lexical path (detectLanguageN over the catalog signals)
// starts dispatching to it. One loop: file appears -> 27B analyzes ->
// index grows -> sub-expert found-or-created.
//
// The oracle is an OpenAI-shaped endpoint (llama-server on the 40GB box
// or an API tier): GOTATO_ORACLE_URL. `--mock` runs a builtin classifier
// with the SAME JSON contract so the loop is testable on the potato
// without the 27B.
//
// Note on "analyze the 27B's weights": measured dead end (AGENT.md) - the
// weights contain no separable experts (redundancy ~0 across 64 layers),
// so the 27B's contribution is KNOWLEDGE (code analysis), not weight
// carving.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"
)

// ---- the catalog ----------------------------------------------------------

type langEntry struct {
	Exts     []string `json:"exts"`
	Signals  []string `json:"signals"`  // regex sources, compiled at load
	Keywords []string `json:"keywords"`
	Source   string   `json:"source"` // "builtin" | "oracle"
	AddedAt  float64  `json:"added_at,omitempty"`
}

type langCatalogFile struct {
	Languages map[string]*langEntry `json:"languages"`
}

var catalogPath string = fleetDir + "/languages.json"
var langCatalog map[string]*langEntry = make(map[string]*langEntry)

// seedCatalog: the builtin languages (the old hardcoded maps, now data).
func seedCatalog() {
	langCatalog["python"] = &langEntry{
		Exts: []string{".py", ".pyw", ".pyi"},
		Signals: []string{`(?m)^\s*(def |class |import |from |async def )`,
			`print\(.*\)`, `self\.`, `if __name__`},
		Keywords: []string{"def", "class", "import", "self"},
		Source:   "builtin"}
	langCatalog["typescript"] = &langEntry{
		Exts: []string{".ts", ".tsx"},
		Signals: []string{`(interface |type |const .*: |: string|: number)`,
			`import .* from ['"]`, `export (default )?(function|class|const)`},
		Keywords: []string{"interface", "type", "export"},
		Source:   "builtin"}
	langCatalog["javascript"] = &langEntry{
		Exts: []string{".js", ".jsx", ".mjs"},
		Signals: []string{`(function |const .* = \(.*\) =>|require\(|module\.exports)`,
			`console\.log\(|=>\s*\{`},
		Keywords: []string{"function", "const", "require"},
		Source:   "builtin"}
	langCatalog["rust"] = &langEntry{
		Exts: []string{".rs"},
		Signals: []string{`(?m)^\s*(fn |let mut |impl |pub |use )`,
			`println!\(|unwrap\(\)|\.await`},
		Keywords: []string{"fn", "let", "impl", "pub"},
		Source:   "builtin"}
	langCatalog["go"] = &langEntry{
		Exts: []string{".go"},
		Signals: []string{`(?m)^\s*(package \w+|func |import |var |const |type )`,
			`:=`, `fmt\.|\w+\.\w+\(`, `\*\w+|\[\]\w+`},
		Keywords: []string{"package", "func", "import"},
		Source:   "builtin"}
	langCatalog["ruby"] = &langEntry{
		Exts:     []string{".rb"},
		Signals:  []string{`(?m)^\s*(def |class |require |end)`, `puts `, `@\w+`},
		Keywords: []string{"def", "class", "require"},
		Source:   "builtin"}
	langCatalog["java"] = &langEntry{
		Exts:     []string{".java"},
		Signals:  []string{`(?m)^\s*(public |private |protected |class \w+|import )`, `System\.out`},
		Keywords: []string{"public", "class", "void"},
		Source:   "builtin"}
	langCatalog["kotlin"] = &langEntry{
		Exts:     []string{".kt"},
		Signals:  []string{`(?m)^\s*(fun |val |var |class |package )`, `println\(`},
		Keywords: []string{"fun", "val", "class"},
		Source:   "builtin"}
	langCatalog["c"] = &langEntry{
		Exts:     []string{".c", ".h"},
		Signals:  []string{`#include <`, `#define `, `(?m)^\s*(int |void |char |struct |typedef )`},
		Keywords: []string{"include", "struct", "void"},
		Source:   "builtin"}
	langCatalog["cpp"] = &langEntry{
		Exts:     []string{".cpp", ".cc", ".hpp"},
		Signals:  []string{`#include <`, `std::`, `(?m)^\s*(int |void |template |namespace )`},
		Keywords: []string{"std", "template", "namespace"},
		Source:   "builtin"}
	langCatalog["csharp"] = &langEntry{
		Exts:     []string{".cs"},
		Signals:  []string{`(?m)^\s*(using |namespace |public class |static void )`, `Console\.Write`},
		Keywords: []string{"using", "namespace", "class"},
		Source:   "builtin"}
	langCatalog["php"] = &langEntry{
		Exts:     []string{".php"},
		Signals:  []string{`<\?php`, `function \w+\(`, `\$this->`},
		Keywords: []string{"function", "echo", "php"},
		Source:   "builtin"}
	langCatalog["swift"] = &langEntry{
		Exts:     []string{".swift"},
		Signals:  []string{`(?m)^\s*(import |func |struct |class |let |var )`, `print\(`},
		Keywords: []string{"func", "struct", "import"},
		Source:   "builtin"}
}

// langCatalogInit: seed, merge the fleet file, rebuild the runtime maps
// (extToLang, langSignals) that scan/index/detect/watch all read.
func langCatalogInit() {
	seedCatalog()
	catalogPath = fleetDir + "/languages.json"
	if data, err := os.ReadFile(catalogPath); err == nil {
		var cf langCatalogFile
		if json.Unmarshal(data, &cf) == nil {
			for name, e := range cf.Languages {
				langCatalog[name] = e
			}
		}
	}
	rebuildLangMaps()
}

// rebuildLangMaps: extToLang + langSignals from the catalog (the two
// package globals everything else reads - one source of truth).
func rebuildLangMaps() {
	extToLang = make(map[string]string)
	langSignals = make(map[string][]*regexp.Regexp)
	for name, e := range langCatalog {
		for _, ext := range e.Exts {
			extToLang[ext] = name
		}
		var pats []*regexp.Regexp = make([]*regexp.Regexp, 0, len(e.Signals))
		for _, s := range e.Signals {
			if re, err := regexp.Compile(s); err == nil {
				pats = append(pats, re)
			}
		}
		if len(pats) > 0 {
			langSignals[name] = pats
		}
	}
}

func saveCatalog() {
	var cf langCatalogFile = langCatalogFile{Languages: langCatalog}
	data, _ := json.MarshalIndent(cf, "", " ")
	var tmp string = catalogPath + ".tmp"
	if os.WriteFile(tmp, data, 0644) == nil {
		_ = os.Rename(tmp, catalogPath)
	}
}

func langKnown(lang string) bool {
	_, ok := langCatalog[lang]
	return ok
}

func langsCmd(args []string) {
	langCatalogInit()
	var names []string = make([]string, 0, len(langCatalog))
	for name := range langCatalog {
		names = append(names, name)
	}
	sort.Strings(names)
	fmt.Printf("%-14s %-10s %-12s %-8s %s\n", "language", "source", "exts", "expert", "signals")
	for _, name := range names {
		e := langCatalog[name]
		var expert string = "-"
		if idx := loadIndex(); idx != nil {
			if en, ok := idx[name]; ok {
				expert = en.Status
			}
		}
		fmt.Printf("%-14s %-10s %-12s %-8s %v\n",
			name, e.Source, strings.Join(e.Exts, ","), expert, len(e.Signals))
	}
}

// ---- the oracle -----------------------------------------------------------

const oracleAnalyzePrompt = `You are a programming language analyst for an
expert-slicing system. Below are code samples from files whose language is
NOT in our catalog. Identify the programming language of each sample and
return ONLY a JSON object, no prose:

{"languages":[{"name":"lua","exts":[".lua"],"signals":["regex1","regex2"],
"keywords":["kw1","kw2"]}]}

Rules:
- "name": lowercase identifier (e.g. "lua", "perl", "r", "sql").
- "exts": the file extensions for the language, each starting with ".".
- "signals": 2-4 Go-compatible regexes that match distinctive constructs
  (e.g. "(?m)^\\s*(local |function )" for lua). They must match ONLY this
  language, never common English.
- "keywords": 2-4 distinctive keywords.
- One entry per distinct language in the samples; if a sample is not code,
  skip it. Do not invent languages already covered by the catalog.`

type oracleProposal struct {
	Name     string   `json:"name"`
	Exts     []string `json:"exts"`
	Signals  []string `json:"signals"`
	Keywords []string `json:"keywords"`
}

type oracleResponse struct {
	Languages []oracleProposal `json:"languages"`
}

// collectUnknownFiles: stack files no catalog extension covers (the
// candidates for oracle analysis). Skips binaries (null-byte sniff) and
// non-code blobs by size heuristic.
func collectUnknownFiles(stack string, cap int) []string {
	var out []string
	filepath.Walk(stack, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if info.Name() == ".git" || info.Name() == "node_modules" || info.Name() == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}
		ext := strings.ToLower(filepath.Ext(p))
		if _, ok := extToLang[ext]; ok {
			return nil
		}
		if info.Size() < 80 || info.Size() > 300*1024 {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		var probe int = len(data)
		if probe > 512 {
			probe = 512
		}
		if bytes.IndexByte(data[:probe], 0) >= 0 {
			return nil // binary (png/stl/obj/...): not code, not analyzable
		}
		out = append(out, p)
		if len(out) >= cap {
			return errStopWalk
		}
		return nil
	})
	return out
}

var errStopWalk = fmt.Errorf("stop")

func sampleFiles(files []string) string {
	var sb strings.Builder
	for _, p := range files {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		sb.WriteString("# ==== " + p + " ====\n")
		var s string = string(data)
		if len(s) > 1500 {
			s = s[:1500]
		}
		sb.WriteString(s)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

// oracleClassifyMock: the potato stand-in for the 27B. Same JSON contract;
// recognizes a few languages the catalog lacks via distinctive regexes.
func oracleClassifyMock(samples string) oracleResponse {
	var resp oracleResponse
	var rules = []struct {
		name     string
		exts     []string
		signals  []string
		keywords []string
	}{
		{"lua", []string{".lua"}, []string{`(?m)^\s*(local |function |end\b)`, `require\(['"]\w+['"]\)`, `:upper\(|:rep\(`}, []string{"local", "function", "end"}},
		{"perl", []string{".pl", ".pm"}, []string{`use strict`, `my \$`, `sub \w+ \{`}, []string{"my", "sub", "use"}},
		{"r", []string{".r", ".R"}, []string{`(?m)^\s*\w+ <- `, `library\(|require\(`}, []string{"library", "data.frame"}},
		{"sql", []string{".sql"}, []string{`(?m)^\s*(SELECT |INSERT INTO |CREATE TABLE |UPDATE )`, `FROM \w+`}, []string{"SELECT", "FROM", "CREATE"}},
		{"dart", []string{".dart"}, []string{`import 'package:`, `(?m)^\s*(void main|class \w+ extends)`}, []string{"class", "main", "import"}},
		{"gdscript", []string{".gd"}, []string{`(?m)^\s*extends \w+`, `func _ready\(|func _process\(|@export|@onready`, `signal \w+`, `move_and_slide\(|queue_free\(`}, []string{"extends", "func", "signal", "export"}},
		{"html-css", []string{".html", ".htm", ".css"}, []string{`(?i)<!doctype html|<html`, `(?i)</(div|span|body|head|section|header|footer)>`, `(?i)class=\s*["']|id=\s*["']|href=`}, []string{"html", "div", "class", "css"}},
	}
	for _, r := range rules {
		var hit bool = false
		for _, s := range r.signals {
			if re, err := regexp.Compile(s); err == nil && re.MatchString(samples) {
				hit = true
				break
			}
		}
		if hit {
			resp.Languages = append(resp.Languages, oracleProposal{
				Name: r.name, Exts: r.exts, Signals: r.signals, Keywords: r.keywords})
		}
	}
	return resp
}

// oracleClassifyRemote: ask the 27B (OpenAI-shaped endpoint) to classify.
func oracleClassifyRemote(samples string) (oracleResponse, error) {
	var url string = os.Getenv("GOTATO_ORACLE_URL")
	if url == "" {
		return oracleResponse{}, fmt.Errorf("GOTATO_ORACLE_URL is not set (the 27B oracle endpoint); use --mock on the potato")
	}
	body, _ := json.Marshal(map[string]any{
		"model":      "qwen3.8-27b",
		"messages":   []map[string]string{{"role": "system", "content": oracleAnalyzePrompt}, {"role": "user", "content": samples}},
		"max_tokens": 800,
		"temperature": 0,
	})
	resp, err := httpPostJSONSlow(strings.TrimSuffix(url, "/")+"/v1/chat/completions", body)
	if err != nil {
		return oracleResponse{}, err
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(resp, &out) != nil || len(out.Choices) == 0 {
		return oracleResponse{}, fmt.Errorf("bad oracle response")
	}
	// the 27B may wrap the JSON in fences; extract the first { ... } block
	var content string = out.Choices[0].Message.Content
	var start int = strings.Index(content, "{")
	var end int = strings.LastIndex(content, "}")
	if start < 0 || end <= start {
		return oracleResponse{}, fmt.Errorf("oracle returned no JSON: %s", content[:minInt(len(content), 200)])
	}
	var oc oracleResponse
	if json.Unmarshal([]byte(content[start:end+1]), &oc) != nil {
		return oracleResponse{}, fmt.Errorf("oracle JSON invalid: %s", content[start:minInt(end+1, start+300)])
	}
	return oc, nil
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

// registerProposals: validate + add oracle-discovered languages to the
// catalog, rebuild the runtime maps, persist. Returns the added names.
func registerProposals(oc oracleResponse) []string {
	var added []string
	var nameRe *regexp.Regexp = regexp.MustCompile(`^[a-z][a-z0-9_-]*$`)
	var extRe *regexp.Regexp = regexp.MustCompile(`^\.[a-zA-Z0-9_]+$`)
	for _, p := range oc.Languages {
		p.Name = strings.ToLower(strings.TrimSpace(p.Name))
		if !nameRe.MatchString(p.Name) || langKnown(p.Name) {
			continue
		}
		var e langEntry
		for _, ext := range p.Exts {
			ext = strings.ToLower(strings.TrimSpace(ext))
			if extRe.MatchString(ext) {
				e.Exts = append(e.Exts, ext)
			}
		}
		for _, s := range p.Signals {
			if _, err := regexp.Compile(s); err == nil {
				e.Signals = append(e.Signals, s)
			}
		}
		for _, kw := range p.Keywords {
			e.Keywords = append(e.Keywords, strings.ToLower(kw))
		}
		if len(e.Exts) == 0 || len(e.Signals) == 0 {
			continue
		}
		e.Source = "oracle"
		e.AddedAt = float64(time.Now().Unix())
		langCatalog[p.Name] = &e
		added = append(added, p.Name)
	}
	if len(added) > 0 {
		rebuildLangMaps()
		saveCatalog()
	}
	return added
}

// oracleCmd: expertd oracle <stack> [--mock]
func oracleCmd(args []string) {
	var stack string = ""
	var mock bool = false
	for _, a := range args {
		if a == "--mock" {
			mock = true
		} else {
			stack = a
		}
	}
	if stack == "" {
		fmt.Println("usage: expertd oracle <stack> [--mock]")
		os.Exit(2)
	}
	langCatalogInit()
	var files []string = collectUnknownFiles(stack, 32)
	if len(files) == 0 {
		fmt.Println("[oracle] no files outside the catalog - nothing to analyze")
		return
	}
	fmt.Printf("[oracle] %d unknown files -> analyzing with %s\n",
		len(files), map[bool]string{true: "mock classifier", false: "Qwen3.8-27B"}[mock])
	var samples string = sampleFiles(files)
	var oc oracleResponse
	var err error
	if mock {
		oc = oracleClassifyMock(samples)
	} else {
		oc, err = oracleClassifyRemote(samples)
		if err != nil {
			fmt.Println("[oracle]", err)
			os.Exit(1)
		}
	}
	if len(oc.Languages) == 0 {
		fmt.Println("[oracle] no language identified - the samples may not be code")
		return
	}
	var added []string = registerProposals(oc)
	if len(added) == 0 {
		fmt.Println("[oracle] nothing new (languages already catalogued)")
		return
	}
	fmt.Printf("[oracle] catalogued: %v -> languages.json (the watcher slices them on its next cycle)\n", added)
}

// spawnSliceBuild: detach `expertd build <lang> <stack>` (the gateway's
// "say yes and I will slice one" promise for languages with no ready
// expert). The builder re-checks index.json before publishing, so racing
// the watcher is harmless.
func spawnSliceBuild(lang string) bool {
	if e, ok := loadIndex()[lang]; ok && e.Status == "ready" {
		return false
	}
	self, err := os.Executable()
	if err != nil {
		return false
	}
	stack, err := os.Getwd()
	if err != nil {
		return false
	}
	cmd := exec.Command("nice", "-n", "10", self, "build", lang, stack)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	return cmd.Start() == nil
}
