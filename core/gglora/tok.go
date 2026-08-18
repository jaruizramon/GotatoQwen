// tok.go - tiktoken-style BPE tokenizer for Qwen3, built from the GGUF
// tokenizer metadata (model "gpt2", pre "qwen2"). Hand-rolled because Go's
// regexp (RE2) has no lookahead and the qwen2 pre-tokenizer needs one.
// Verified against llama-server's /tokenize before any training run.
package main

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type tokenizer struct {
	rank    map[string]int  // token string -> id
	merges  map[string]int  // "a b" -> rank
	byteEnc [256]rune       // gpt2 byte encoder
	byteDec map[rune]byte
}

func newTokenizer(g *ggufFile) *tokenizer {
	t := &tokenizer{rank: map[string]int{}, merges: map[string]int{}, byteDec: map[rune]byte{}}
	// byte encoder (standard gpt2 table)
	var bs []int
	var i int = 0
	for i = 0x21; i <= 0x7E; i++ {
		bs = append(bs, i)
	}
	for i = 0xA1; i <= 0xAC; i++ {
		bs = append(bs, i)
	}
	for i = 0xAE; i <= 0xFF; i++ {
		bs = append(bs, i)
	}
	var cs []int = append([]int{}, bs...)
	var n int = 0
	var b int = 0
	for b = 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	// byteEnc[byte] = the cs entry at the position where the byte appears in
	// bs (cs is indexed by POSITION, not by byte value).
	for i := 0; i < len(bs); i++ {
		t.byteEnc[byte(bs[i])] = rune(cs[i])
		t.byteDec[rune(cs[i])] = byte(bs[i])
	}
	// tokens
	toks := g.kvArray("tokenizer.ggml.tokens")
	for id := 0; id < len(toks); id++ {
		if s, ok := toks[id].(string); ok {
			t.rank[s] = id
		}
	}
	// merges
	merges := g.kvArray("tokenizer.ggml.merges")
	for rank := 0; rank < len(merges); rank++ {
		if s, ok := merges[rank].(string); ok {
			t.merges[s] = rank
		}
	}
	return t
}

// encodePiece: byte-encode one pre-token, then BPE it, then map to ids.
func (t *tokenizer) encodePiece(piece string) []int {
	var sb strings.Builder
	for i := 0; i < len(piece); {
		r, sz := utf8.DecodeRuneInString(piece[i:])
		if r == utf8.RuneError && sz == 1 {
			sb.WriteRune(t.byteEnc[piece[i]])
			i++
		} else {
			for _, by := range []byte(piece[i : i+sz]) {
				sb.WriteRune(t.byteEnc[by])
			}
			i += sz
		}
	}
	enc := sb.String()
	if id, ok := t.rank[enc]; ok {
		return []int{id}
	}
	// BPE: start from single chars; merge ONE pair per iteration (globally
	// lowest rank, leftmost on ties) exactly like tiktoken's reference - a
	// merge-all-occurrences variant diverges on repeated pairs.
	parts := make([]string, 0, len(enc))
	for _, r := range enc {
		parts = append(parts, string(r))
	}
	for {
		minIdx := -1
		minRank := 1 << 30
		for i := 0; i < len(parts)-1; i++ {
			key := parts[i] + " " + parts[i+1]
			if rk, ok := t.merges[key]; ok && rk < minRank {
				minRank = rk
				minIdx = i
			}
		}
		if minIdx < 0 {
			break
		}
		parts[minIdx] = parts[minIdx] + parts[minIdx+1]
		parts = append(parts[:minIdx+1], parts[minIdx+2:]...)
	}
	var ids []int
	for _, p := range parts {
		if id, ok := t.rank[p]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// isLetter / isNumber: Unicode classes used by the qwen2 pre-tokenizer.
func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }
func isWS(r rune) bool     { return unicode.IsSpace(r) }

// preTokenize: mirrors llama.cpp's unicode_regex_split_custom_qwen2 EXACTLY
// (the hand-written splitter in src/unicode.cpp, not the raw regex - the
// two differ on whitespace runs: N-1 spaces split off, the last space
// attaches to the following word). Verified against /tokenize.
func preTokenize(s string) []string {
	runes := []rune(s)
	n := len(runes)
	var out []string
	var pos int = 0
	for pos < n {
		start := pos
		cpt := runes[pos]
		// regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
		if cpt == '\'' && pos+1 < n {
			next := unicode.ToLower(runes[pos+1])
			if next == 's' || next == 't' || next == 'm' || next == 'd' {
				out = append(out, string(runes[pos:pos+2]))
				pos += 2
				continue
			}
			if pos+2 < n {
				nn := unicode.ToLower(runes[pos+2])
				if (next == 'r' && nn == 'e') || (next == 'v' && nn == 'e') ||
					(next == 'l' && nn == 'l') {
					out = append(out, string(runes[pos:pos+3]))
					pos += 3
					continue
				}
			}
		}
		// regex: [^\r\n\p{L}\p{N}]?\p{L}+  (the optional prefix can be a
		// space; the LAST space of a run attaches to the following word)
		if cpt != '\r' && cpt != '\n' && !isNumber(cpt) {
			if isLetter(cpt) || (pos+1 < n && isLetter(runes[pos+1])) {
				pos++
				for pos < n && isLetter(runes[pos]) {
					pos++
				}
				out = append(out, string(runes[start:pos]))
				continue
			}
		}
		// regex: \p{N}
		if isNumber(cpt) {
			out = append(out, string(cpt))
			pos++
			continue
		}
		// regex:  ?[^\s\p{L}\p{N}]+[\r\n]*
		var probe rune = cpt
		if cpt == ' ' && pos+1 < n {
			probe = runes[pos+1]
		}
		if !isWS(probe) && !isLetter(probe) && !isNumber(probe) {
			if cpt == ' ' {
				pos++
			}
			for pos < n && !isWS(runes[pos]) && !isLetter(runes[pos]) && !isNumber(runes[pos]) {
				pos++
			}
			for pos < n && (runes[pos] == '\r' || runes[pos] == '\n') {
				pos++
			}
			out = append(out, string(runes[start:pos]))
			continue
		}
		// whitespace run
		var num int = 0
		var lastNL int = -1
		for pos+num < n && isWS(runes[pos+num]) {
			if runes[pos+num] == '\r' || runes[pos+num] == '\n' {
				lastNL = pos + num + 1
			}
			num++
		}
		// regex: \s*[\r\n]+
		if lastNL > 0 {
			out = append(out, string(runes[pos:lastNL]))
			pos = lastNL
			continue
		}
		// regex: \s+(?!\S)  (llama: run >= 2 with a next char -> N-1)
		if num > 1 && pos+num < n {
			out = append(out, string(runes[pos:pos+num-1]))
			pos += num - 1
			continue
		}
		// regex: \s+
		if num > 0 {
			out = append(out, string(runes[pos:pos+num]))
			pos += num
			continue
		}
		// no match (unreachable for valid unicode)
		out = append(out, string(cpt))
		pos++
	}
	return out
}

// Encode: full pipeline, text -> token ids. addBos honors the model flag.
func (t *tokenizer) Encode(text string) []int {
	var ids []int
	for _, piece := range preTokenize(text) {
		ids = append(ids, t.encodePiece(piece)...)
	}
	return ids
}

// sortedTokenKeys: for debugging /tests.
func (t *tokenizer) sortedTokenKeys() []string {
	var keys []string
	for k := range t.rank {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
