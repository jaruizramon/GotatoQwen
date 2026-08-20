package main

import "fmt"

func dbgMain6() {
	var g *ggufFile

	g, _ = loadGGUF("/home/pipo/slm-fleet/Qwen3-0.6B-Q8_0.gguf")
	var toks = g.kvArray("tokenizer.ggml.tokens")
	var id int

	for _, id = range []int{257, 262, 60, 61, 62} {
		fmt.Printf("id %d = %q\n", id, toks[id])
	}
}
