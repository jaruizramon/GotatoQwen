package main

import "fmt"

func dbgMain5() {
	g, _ := loadGGUF("/home/pipo/slm-fleet/Qwen3-0.6B-Q8_0.gguf")
	tok := newTokenizer(g)
	toks := g.kvArray("tokenizer.ggml.tokens")
	for _, id := range []int{8917, 9190, 9122, 3379, 453, 2943} {
		if id < len(toks) {
			fmt.Printf("id %d = %q\n", id, toks[id])
		}
	}
	ids := tok.Encode("Summarize")
	fmt.Println("my ids:", ids)
	for _, id := range ids {
		if id < len(toks) {
			fmt.Printf("  %d = %q\n", id, toks[id])
		}
	}
	// server
	fmt.Println("(compare with server: 8917 + ...)")
}
