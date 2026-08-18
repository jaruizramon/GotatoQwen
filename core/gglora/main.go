// main.go - proof of concept: one real-weight LoRA step in Go, then dump
// A/B/dA/dB so the torch side can verify parity.
//
// usage: gglora <model.gguf> <tensor-name> <m> <outfile.json>
package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 5 {
		fmt.Println("usage: gglora <model.gguf> <tensor> <m> <out.json>")
		os.Exit(2)
	}
	var path string = os.Args[1]
	var name string = os.Args[2]
	var m int = atoi(os.Args[3])
	var outPath string = os.Args[4]

	g, err := loadGGUF(path)
	if err != nil {
		fmt.Println("gguf:", err)
		os.Exit(1)
	}
	var info *tensorInfo
	var idx int = 0
	for idx = 0; idx < len(g.infos); idx++ {
		if g.infos[idx].Name == name {
			info = &g.infos[idx]
			break
		}
	}
	if info == nil {
		fmt.Println("tensor not found:", name)
		os.Exit(1)
	}
	if info.Ndim != 2 {
		fmt.Println("need a 2D tensor")
		os.Exit(1)
	}
	var out int = int(info.Dims[0])
	var in int = int(info.Dims[1])
	var W []float32 = g.tensorF32(info)
	fmt.Printf("loaded %s [%d x %d] (%d elems) align=%d off=%d abs=%d\n",
		name, out, in, len(W), g.align, info.Offset, g.align+info.Offset)

	var l *loraLinear = newLoraLinear(W, in, out)
	var x []float32 = make([]float32, m*in)
	var xi int = 0
	for xi = 0; xi < m*in; xi++ {
		x[xi] = randF32()*2 - 1
	}
	var y []float32 = make([]float32, m*out)
	var xh []float32 = l.forward(y, x, m)
	var dy []float32 = make([]float32, m*out)
	var yi int = 0
	for yi = 0; yi < m*out; yi++ {
		dy[yi] = y[yi] // "loss = 0.5 sum(y^2) -> dy = y"
	}
	// pre-step A/B for the parity file
	var A0 []float32 = append([]float32{}, l.A...)
	var B0 []float32 = append([]float32{}, l.B...)
	// pre-step grads (unstepped)
	var dA []float32
	var dB []float32
	dA, dB = l.grads(dy, x, xh, m)
	// now the real stepped update (for A1/B1 in the file)
	l.backward(dy, x, xh, m)

	var payload map[string]any = map[string]any{
		"tensor": name, "out": out, "in": in, "rank": loraRank,
		"alpha": loraAlpha, "m": m, "lr": adamLR,
		"x": x, "xh": xh, "dy": dy, "W": W, "B0": B0, "y": y,
		"A": A0, "B": B0, "dA": dA, "dB": dB,
		"A1": l.A, "B1": l.B,
	}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		fmt.Println("marshal:", err)
		os.Exit(1)
	}
	if os.WriteFile(outPath, jsonData, 0644) != nil {
		fmt.Println("write failed")
		os.Exit(1)
	}
	fmt.Printf("wrote %s (A %d, B %d elems; step=%d)\n", outPath, len(A0), len(B0), l.step)
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
