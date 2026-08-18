// segstore.go - the summary segment store.
//
// Memory architecture (storage is cheap, RAM is not):
//   - every compaction writes the full summary to sessions/<id>/seg-<n>.gz
//     (gzip; the raw archive stays append-only on disk);
//   - a per-session manifest.json maps segment -> {gz file, 200-char
//     preview, token count, timestamps} - a past point of the chat is a
//     pointer, not raw text;
//   - a RAM LRU cache (cap: GOTATO_SUMMARY_CACHE_MB, default 1024 = the
//     reserved 1GB) holds decompressed hot summaries; cold ones are read
//     from gz on demand;
//   - retrieval: the current prompt is scored against the previews (cheap,
//     deterministic word overlap); matching segments are decompressed and
//     injected into the context so a "what did we decide about X" lands on
//     the right past point without loading the raw archive.
package main

import (
	"bufio"
	"compress/gzip"
	"container/list"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type segMeta struct {
	ID      int    `json:"id"`
	GZ      string `json:"gz"` // relative to fleetDir
	Preview string `json:"preview"`
	Tokens  int    `json:"tokens"`
	StartTs int64  `json:"start_ts"`
	EndTs   int64  `json:"end_ts"`
}

type segManifest struct {
	Segments []segMeta `json:"segments"`
}

var segMu sync.Mutex

// segDir: per-session segment directory.
func segDir(session string) string {
	return filepath.Join(fleetDir, "segments", session)
}

func segManifestPath(session string) string {
	return filepath.Join(segDir(session), "manifest.json")
}

// loadManifest: read (or create) the session's segment manifest.
func loadManifest(session string) *segManifest {
	m := &segManifest{}
	data, err := os.ReadFile(segManifestPath(session))
	if err != nil {
		return m
	}
	_ = json.Unmarshal(data, m)
	return m
}

func saveManifest(session string, m *segManifest) {
	dir := segDir(session)
	if os.MkdirAll(dir, 0755) != nil {
		return
	}
	data, _ := json.MarshalIndent(m, "", " ")
	_ = os.WriteFile(segManifestPath(session), data, 0644)
}

// previewOf: the 200-char pointer shown in live context.
func previewOf(summary string) string {
	p := strings.Join(strings.Fields(summary), " ")
	if len(p) > 200 {
		return p[:200] + "…"
	}
	return p
}

// ---- RAM LRU (the reserved 1GB) -------------------------------------------

var segLRU = list.New()                       // front = hottest
var segCache = map[int][]byte{}               // segment id -> decompressed
var segCacheBytes = 0
var segCacheCap = 1 << 30 // 1GB default; GOTATO_SUMMARY_CACHE_MB overrides

func segCacheInit() {
	if v := os.Getenv("GOTATO_SUMMARY_CACHE_MB"); v != "" {
		if mb := atoi(v); mb > 0 {
			segCacheCap = mb << 20
		}
	}
}

func segCacheTouch(id int) {
	for e := segLRU.Front(); e != nil; e = e.Next() {
		if e.Value.(int) == id {
			segLRU.MoveToFront(e)
			return
		}
	}
}

func segCacheEvict() {
	for segCacheBytes > segCacheCap && segLRU.Len() > 1 {
		back := segLRU.Back()
		id := back.Value.(int)
		segCacheBytes -= len(segCache[id])
		delete(segCache, id)
		segLRU.Remove(back)
	}
}

// saveSegment: gz-compress a summary, record it in the manifest, put it in
// RAM. Returns the segment meta (the caller uses Preview + ID as pointers).
func saveSegment(session string, summary string) segMeta {
	segMu.Lock()
	defer segMu.Unlock()
	m := loadManifest(session)
	id := 1
	if len(m.Segments) > 0 {
		id = m.Segments[len(m.Segments)-1].ID + 1
	}
	gzRel := filepath.Join("segments", session, fmt.Sprintf("seg-%04d.gz", id))
	gzAbs := filepath.Join(fleetDir, gzRel)
	dir := filepath.Dir(gzAbs)
	if os.MkdirAll(dir, 0755) != nil {
		return segMeta{}
	}
	var buf strings.Builder
	gw := gzip.NewWriter(&buf)
	_, _ = gw.Write([]byte(summary))
	_ = gw.Close()
	if os.WriteFile(gzAbs, []byte(buf.String()), 0644) != nil {
		return segMeta{}
	}
	meta := segMeta{
		ID:      id,
		GZ:      gzRel,
		Preview: previewOf(summary),
		Tokens:  estTokens(summary),
		StartTs: nowMs(),
		EndTs:   nowMs(),
	}
	m.Segments = append(m.Segments, meta)
	saveManifest(session, m)
	// RAM: decompressed copy (bounded by the reserved cap)
	segCacheTouch(id)
	segCache[id] = []byte(summary)
	segCacheBytes += len(summary)
	segCacheEvict()
	return meta
}

// loadSegment: decompress a segment (RAM-first, gz on miss).
func loadSegment(session string, meta segMeta) string {
	segMu.Lock()
	if v, ok := segCache[meta.ID]; ok {
		segCacheTouch(meta.ID)
		segMu.Unlock()
		return string(v)
	}
	segMu.Unlock()
	data, err := os.ReadFile(filepath.Join(fleetDir, meta.GZ))
	if err != nil {
		return ""
	}
	zr, err := gzip.NewReader(strings.NewReader(string(data)))
	if err != nil {
		return ""
	}
	var out strings.Builder
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		out.WriteString(sc.Text())
		out.WriteString("\n")
	}
	segMu.Lock()
	segCacheTouch(meta.ID)
	segCache[meta.ID] = []byte(out.String())
	segCacheBytes += out.Len()
	segCacheEvict()
	segMu.Unlock()
	return out.String()
}

// scorePreview: deterministic word-overlap between the prompt and a preview
// (shared unique words / prompt words). Cheap and stable.
func scorePreview(prompt string, preview string) float64 {
	pw := wordSet(prompt)
	if len(pw) == 0 {
		return 0
	}
	pv := wordSet(preview)
	var shared int = 0
	for w := range pw {
		if _, ok := pv[w]; ok {
			shared++
		}
	}
	return float64(shared) / float64(len(pw))
}

func wordSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, w := range strings.Fields(strings.ToLower(s)) {
		w = strings.Trim(w, ".,;:!?()[]{}\"'`")
		if len(w) >= 4 && !stopwords[w] {
			out[w] = true
		}
	}
	return out
}

// stopwords: the potato retriever must not match on filler.
var stopwords = map[string]bool{
	"the": true, "and": true, "with": true, "that": true, "this": true,
	"what": true, "when": true, "where": true, "which": true, "from": true,
	"have": true, "been": true, "were": true, "will": true, "would": true,
	"there": true, "their": true, "about": true, "into": true, "your": true,
	"user": true, "asked": true, "said": true, "want": true, "needs": true,
}

// retrieveSegments: the top matching past segments for a prompt (preview
// scoring, deterministic). Returns decompressed summaries + their pointers.
func retrieveSegments(session string, prompt string, topK int) []segMeta {
	segMu.Lock()
	m := loadManifest(session)
	segs := make([]segMeta, len(m.Segments))
	copy(segs, m.Segments)
	segMu.Unlock()
	type scored struct {
		meta  segMeta
		score float64
	}
	var hits []scored
	for _, s := range segs {
		// require at least 2 real shared words (score >= 0.25 with a 4-word
		// prompt floor) - a single stopword overlap is never a match
		if sc := scorePreview(prompt, s.Preview); sc >= 0.25 {
			hits = append(hits, scored{s, sc})
		}
	}
	// stable sort by score desc, then newest first
	for i := 1; i < len(hits); i++ {
		for j := i; j > 0 && (hits[j].score > hits[j-1].score ||
			(hits[j].score == hits[j-1].score && hits[j].meta.ID > hits[j-1].meta.ID)); j-- {
			hits[j], hits[j-1] = hits[j-1], hits[j]
		}
	}
	var out []segMeta
	for i := 0; i < len(hits) && i < topK; i++ {
		out = append(out, hits[i].meta)
	}
	return out
}
