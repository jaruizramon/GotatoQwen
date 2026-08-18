// index.go - the "large but fast sub-index": token n-grams -> labeled
// sections -> delegation target. Deterministic, RAM-resident, Go.
//
// Idea: certain input tokens must accurately hit the semantics of a region,
// so the router can delegate to the SLM that owns that region without paying
// LLM tokens. Built from the project's labeled sections (per language), it
// is the tier-2 index below the lexical tier-1 (extensions/signatures) and
// above the tier-3 fallback (streamed LLM).
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ---- tokenization (code-aware, deterministic) ---------------------------
func indexTokens(text string) []string {
	var out []string = make([]string, 0, 64)
	var cur strings.Builder
	var flush = func() {
		if cur.Len() >= 2 {
			out = append(out, strings.ToLower(cur.String()))
		}
		cur.Reset()
	}
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '_' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// ---- the index -----------------------------------------------------------
type section struct {
	Path     string            `json:"path"`
	Lang     string            `json:"lang"`
	Label    string            `json:"label"` // semantic label (module name)
	Tokens   int               `json:"tokens"`
	NGrams   map[string]int    `json:"-"` // ngram -> count within section
	TopTerms []string          `json:"top_terms"`
}

type subIndex struct {
	Sections []*section         `json:"sections"`
	Postings map[string][]int   `json:"postings"` // ngram -> section ids
	DocFreq  map[string]int     `json:"docfreq"`
	NGramN   int                `json:"ngram_n"`
}

func newSubIndex(n int) *subIndex {
	return &subIndex{NGramN: n, Postings: make(map[string][]int), DocFreq: make(map[string]int)}
}

// build: tokenize each section file, count n-grams, build postings.
func (idx *subIndex) build(root string, exts map[string]string, labelOf func(path string) string) int {
	var files []string
	filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(p))
		if _, ok := exts[ext]; ok {
			files = append(files, p)
		}
		return nil
	})
	sort.Strings(files)
	for _, p := range files {
		data, err := os.ReadFile(p)
		if err != nil || len(data) < 80 {
			continue
		}
		toks := indexTokens(string(data))
		if len(toks) < 16 {
			continue
		}
		sec := &section{Path: p, Lang: exts[strings.ToLower(filepath.Ext(p))],
			Label: labelOf(p), Tokens: len(toks), NGrams: make(map[string]int)}
		var n int = idx.NGramN
		if n < 3 {
			n = 3
		}
		for k := 1; k <= n; k++ {
			for i := 0; i+k <= len(toks); i++ {
				key := fmt.Sprintf("%d:%s", k, strings.Join(toks[i:i+k], " "))
				sec.NGrams[key]++
			}
		}
		// top terms by frequency for the label
		type tf struct {
			t string
			n int
		}
		var tfs []tf
		for t, n := range sec.NGrams {
			tfs = append(tfs, tf{t, n})
		}
		sort.Slice(tfs, func(a, b int) bool { return tfs[a].n > tfs[b].n })
		for i := 0; i < len(tfs) && i < 8; i++ {
			sec.TopTerms = append(sec.TopTerms, tfs[i].t)
		}
		id := len(idx.Sections)
		idx.Sections = append(idx.Sections, sec)
		seen := make(map[string]bool)
		for ng := range sec.NGrams {
			idx.Postings[ng] = append(idx.Postings[ng], id)
			if !seen[ng] {
				idx.DocFreq[ng]++
				seen[ng] = true
			}
		}
	}
	return len(idx.Sections)
}

// resolve: score sections by IDF-weighted n-gram hits; return top matches.
type hit struct {
	Section  int     `json:"section"`
	Path     string  `json:"path"`
	Lang     string  `json:"lang"`
	Label    string  `json:"label"`
	Score    float64 `json:"score"`
	Hits     int     `json:"hits"`
	TopTerms []string `json:"top_terms"`
}

func (idx *subIndex) resolve(text string, topN int) []hit {
	toks := indexTokens(text)
	scores := make([]float64, len(idx.Sections))
	nhits := make([]int, len(idx.Sections))
	var n float64 = float64(len(idx.Sections))
	for k := 1; k <= 3; k++ {
		for i := 0; i+k <= len(toks); i++ {
			ng := strings.Join(toks[i:i+k], " ")
			key := fmt.Sprintf("%d:%s", k, ng)
			ids, ok := idx.Postings[key]
			if !ok {
				// prefix-tolerant fallback: retries ~ retry, chunked ~ chunk
				for cand := range idx.Postings {
					if cand[0] == byte('0'+k) && len(cand) > 2 &&
						len(ng) >= 4 && (strings.HasPrefix(cand[2:], ng) ||
							strings.HasPrefix(ng, cand[2:])) {
						ids = idx.Postings[cand]
						break
					}
				}
			}
			if len(ids) == 0 {
				continue
			}
			df := float64(idx.DocFreq[key])
			if df == 0 {
				df = 1
			}
			w := (1.0 + n/(1.0+df)) * float64(k) // longer n-grams weigh more
			for _, id := range ids {
				scores[id] += w
				nhits[id]++
			}
		}
	}
	var out []hit
	for id := range idx.Sections {
		if nhits[id] > 0 {
			out = append(out, hit{Section: id, Path: idx.Sections[id].Path,
				Lang: idx.Sections[id].Lang, Label: idx.Sections[id].Label,
				Score: scores[id], Hits: nhits[id], TopTerms: idx.Sections[id].TopTerms})
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Score > out[b].Score })
	if len(out) > topN {
		out = out[:topN]
	}
	return out
}

// ---- persistence ----------------------------------------------------------
func (idx *subIndex) save(path string) error {
	data, err := json.Marshal(idx)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func loadSubIndex(path string) (*subIndex, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx subIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, err
	}
	if idx.Postings == nil {
		idx.Postings = make(map[string][]int)
		idx.DocFreq = make(map[string]int)
	}
	return &idx, nil
}

// ---- CLI helpers ----------------------------------------------------------
func indexCmd(args []string) {
	var root string = "/home/pipo/stack"
	var out string = fleetDir + "/subindex.json"
	var n int = 3
	for i := 0; i < len(args); i++ {
		if args[i] == "--out" && i+1 < len(args) {
			out = args[i+1]
			i++
		} else if args[i] == "--n" && i+1 < len(args) {
			n = atoi(args[i+1])
			i++
		} else {
			root = args[i]
		}
	}
	idx := newSubIndex(n)
	labelOf := func(p string) string {
		return strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
	}
	t0 := nowMs()
	count := idx.build(root, extToLang, labelOf)
	buildMs := nowMs() - t0
	_ = idx.save(out)
	fmt.Printf("[index] %d sections, %d ngrams (%d-grams) in %d ms -> %s\n",
		count, len(idx.Postings), n, buildMs, out)
}

func resolveCmd(args []string) {
	if len(args) < 1 {
		fmt.Println("usage: expertd resolve <file|text> [--index path]")
		os.Exit(2)
	}
	var idxPath string = fleetDir + "/subindex.json"
	var text string
	for i := 0; i < len(args); i++ {
		if args[i] == "--index" && i+1 < len(args) {
			idxPath = args[i+1]
			i++
		} else if _, err := os.Stat(args[i]); err == nil {
			data, _ := os.ReadFile(args[i])
			text = string(data)
		} else {
			text = args[i]
		}
	}
	idx, err := loadSubIndex(idxPath)
	if err != nil {
		fmt.Println("index:", err)
		os.Exit(1)
	}
	t0 := nowMs()
	hits := idx.resolve(text, 3)
	dt := nowMs() - t0
	if len(hits) == 0 {
		fmt.Printf("[resolve] no hits (%d ms) -> tier-3: streamed LLM\n", dt)
		return
	}
	top := hits[0]
	fmt.Printf("[resolve] %d ms | top: %s (%s) score=%.1f hits=%d\n",
		dt, top.Label, top.Lang, top.Score, top.Hits)
	fmt.Printf("          terms: %v\n", top.TopTerms)
	if top.Score > 0 && len(hits) > 1 {
		fmt.Printf("          runner-up: %s (%.1f) -> delegation margin %.1fx\n",
			hits[1].Label, hits[1].Score, top.Score/hits[1].Score)
	}
}

func atoi(s string) int {
	var n int = 0
	var i int = 0
	for i = 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0
		}
		n = n*10 + int(s[i]-'0')
	}
	return n
}

// nowMs: monotonic milliseconds.
func nowMs() int64 {
	return time.Now().UnixMilli()
}
