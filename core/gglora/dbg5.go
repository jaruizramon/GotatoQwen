package main

import "fmt"

func dbgMain5() {
	var g *ggufFile

	g, _ = loadGGUF("/home/pipo/slm-fleet/Qwen3-0.6B-Q8_0.gguf")
	var tok = newTokenizer(g)
	var toks = g.kvArray("tokenizer.ggml.tokens")
	var id int

	for _, id = range []int{8917, 9190, 9122, 3379, 453, 2943} {
		if id < len(toks) {
			fmt.Printf("id %d = %q\n", id, toks[id])
		}
	}
	var ids = tok.Encode("Summarize")
	fmt.Println("my ids:", ids)
	{
		var id int

		for _, id = range ids {
			if id < len(toks) {
				fmt.Printf("  %d = %q\n", id, toks[id])
			}
		}
	}
	// server
	fmt.Println("(compare with server: 8917 + ...)")
}
