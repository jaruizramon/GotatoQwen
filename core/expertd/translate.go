// translate.go - the language bridge (--bridge zh).
//
// Determinism contract:
//   - fixed layer (templates, escalation messages): table-driven, provably
//     deterministic, zero cost
//   - free-form layer (user prompts, replies): greedy decoding (temp=0) for
//     run-to-run reproducibility, PLUS a disk cache keyed by content hash,
//     so identical input ALWAYS yields the identical translation - even
//     across restarts. The cache is the determinism guarantee.
//   - code blocks (``` fences) are NEVER translated.
//
// Caches live in <fleet>/translations/zh/<sha256>.txt (storage is cheap).
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var bridgeZH bool = false

// ---- fixed layer: deterministic table -----------------------------------
var zhTemplates map[string]string = map[string]string{
	"out-of-scope":  "超出当前范围 - 是否委派一个 %s 的SLM？",
	"missing-slice": "还没有 %s 切片在运行 - 向栈中添加 %s 元素（或说“是”，我将启动它），观察器将为其切片。",
	"delegated":     "已委派 - %s 切片正在运行（%s）。请发送任务。",
	"chained":       "[上下文已链接 - 摘要已前置到下一轮]",
}

func zhTemplate(key string, args ...string) string {
	t, ok := zhTemplates[key]
	if !ok || !bridgeZH {
		return ""
	}
	if len(args) > 0 {
		var a []any = make([]any, len(args))
		var i int = 0
		for i = 0; i < len(args); i++ {
			a[i] = args[i]
		}
		return fmt.Sprintf(t, a...)
	}
	return t
}

// ---- code-block splitter --------------------------------------------------
type seg struct {
	code  bool
	text  string
}

// splitBlocks: split text on ``` fences, marking code segments.
func splitBlocks(text string) []seg {
	var out []seg
	var cur strings.Builder
	var inCode bool = false
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			if cur.Len() > 0 {
				out = append(out, seg{code: inCode, text: cur.String()})
				cur.Reset()
			}
			inCode = !inCode
			continue
		}
		cur.WriteString(line + "\n")
	}
	if cur.Len() > 0 {
		out = append(out, seg{code: inCode, text: cur.String()})
	}
	return out
}

// ---- cache ----------------------------------------------------------------
func cachePath(hash string) string {
	return filepath.Join(fleetDir, "translations", "zh", hash+".txt")
}

func cacheGet(hash string) (string, bool) {
	data, err := os.ReadFile(cachePath(hash))
	if err != nil {
		return "", false
	}
	return string(data), true
}

func cachePut(hash string, text string) {
	dir := filepath.Dir(cachePath(hash))
	if os.MkdirAll(dir, 0755) != nil {
		return
	}
	_ = os.WriteFile(cachePath(hash), []byte(text), 0644)
}

func contentHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])[:24]
}

// ---- free-form layer: greedy translation with cache ----------------------
// translateVia: one greedy translation call against the translator backend
// (an INSTRUCT model - base models narrate instead of following orders).
// Uses /v1/chat/completions with a hard system directive and enough budget
// for the model to finish any thinking before the content field.
func translateVia(cfg *serveConfig, direction string, text string) string {
	backend := cfg.backends["translator"]
	if backend == "" {
		return ""
	}
	var directive string
	if direction == "zh" {
		directive = "You are a professional translator. Translate the user text " +
			"into Chinese. Respond with ONLY the Chinese translation. " +
			"No explanation. No thinking."
	} else {
		directive = "You are a professional translator. Translate the Chinese " +
			"text into natural English. Respond with ONLY the English translation. " +
			"No explanation. No thinking."
	}
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": directive},
			{"role": "user", "content": text},
		},
		"temperature": 0, "max_tokens": 400})
	resp, err := httpPostJSON(backend+"/v1/chat/completions", body)
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
	return strings.TrimSpace(out.Choices[0].Message.Content)
}

// translateProseCached: deterministic (cache-first) prose translation.
func translateProseCached(cfg *serveConfig, direction string, text string) string {
	hash := contentHash(direction + "|" + text)
	if cached, ok := cacheGet(hash); ok {
		return cached
	}
	out := translateVia(cfg, direction, text)
	if out == "" {
		return text // never degrade: fall back to the original
	}
	// refuse to cache unclean output (thinking narration, bloat): a polluted
	// cache entry would poison every future identical prompt.
	if len(out) < 4 || len(out) > len(text)*6 {
		return out
	}
	cachePut(hash, out)
	return out
}

// translateBlocks: translate prose segments, keep code verbatim.
func translateBlocks(cfg *serveConfig, direction string, text string) string {
	segs := splitBlocks(text)
	if len(segs) <= 1 && !segs[0].code {
		return translateProseCached(cfg, direction, text)
	}
	var out strings.Builder
	for _, s := range segs {
		if s.code {
			out.WriteString(s.text)
		} else if strings.TrimSpace(s.text) != "" {
			out.WriteString(translateProseCached(cfg, direction, s.text))
		}
	}
	return out.String()
}

func translateToZH(cfg *serveConfig, text string) string {
	return translateBlocks(cfg, "zh", text)
}

func translateToEN(cfg *serveConfig, text string) string {
	return translateBlocks(cfg, "en", text)
}
