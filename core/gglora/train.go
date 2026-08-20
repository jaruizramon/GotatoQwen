// train.go - the Go LoRA trainer: load base GGUF, tokenize the corpus,
// sliding-window samples, forward/backward, AdamW on the LoRA pairs, and
// write the adapter DIRECTLY in the llama.cpp GGUF-LoRA format (the same
// tensor layout the python converter emits - verified against the existing
// python.gguf). No python anywhere in the loop.
//
// usage:
//
//	gglora train --base <model.gguf> --corpus <text> --out <adapter.gguf> \
//	             [--ctx 256] [--stride 128] [--epochs 2] [--threads 4]
//	gglora tokcheck --base <model.gguf> --text "..."          (print ids)
//	gglora tokcheck --base <model.gguf> --verify --server :8082 --text "..."
//	                           (compare against llama-server /tokenize)
package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"time"
)

var trainThreads int = 4

// parallel: run fn over [0,n) in `threads` goroutines over disjoint chunks.
func parallel(n int, fn func(lo int, hi int)) {
	if trainThreads <= 1 || n < 4 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	var chunk int = (n + trainThreads - 1) / trainThreads
	var t int = 0
	for t = 0; t < trainThreads; t++ {
		var lo int = t * chunk
		var hi int = lo + chunk
		if hi > n {
			hi = n
		}
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo int, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// ---- GGUF adapter writer ------------------------------------------------
// Emits the exact llama.cpp LoRA-adapter shape (verified against the
// existing adapters/python.gguf): KV set {general.architecture, general.type,
// adapter.type, general.name, adapter.lora.alpha, general.quantization_version}
// and f32 tensors <base>.lora_a [in, r] / <base>.lora_b [r, out], stored
// out-major (offset = j*in + k) exactly like the in-memory A/B slices.

// clipGrads: per-tensor max-norm clipping (max L2 norm 1.0). Standard
// stability guard: one bad sample must not explode the AdamW state.
func clipGrads(gs ...[]float32) {
	var g []float32

	for _, g = range gs {
		var n2 float64 = 0
		var i int = 0
		for i = 0; i < len(g); i++ {
			n2 += float64(g[i]) * float64(g[i])
		}
		if n2 > 1.0 {
			var scale float64 = 1.0 / math.Sqrt(n2)
			for i = 0; i < len(g); i++ {
				g[i] = float32(float64(g[i]) * scale)
			}
		}
	}
}

func writeAdapterGGUF(path string, name string, m *model, alpha float32) error {
	type tpair struct {
		base string
		a    []float32 // [in*r]
		b    []float32 // [r*out]
		in   int
		r    int
		out  int
	}
	var pairs []tpair
	var l int = 0
	for l = 0; l < m.NLayer; l++ {
		var layer = m.Layers[l]
		pairs = append(pairs,
			tpair{fmt.Sprintf("blk.%d.ffn_down.weight", l), layer.Down.A, layer.Down.B, layer.Down.In, layer.Down.R, layer.Down.Out},
			tpair{fmt.Sprintf("blk.%d.ffn_gate.weight", l), layer.Gate.A, layer.Gate.B, layer.Gate.In, layer.Gate.R, layer.Gate.Out},
			tpair{fmt.Sprintf("blk.%d.ffn_up.weight", l), layer.Up.A, layer.Up.B, layer.Up.In, layer.Up.R, layer.Up.Out},
			tpair{fmt.Sprintf("blk.%d.attn_k.weight", l), layer.K.A, layer.K.B, layer.K.In, layer.K.R, layer.K.Out},
			tpair{fmt.Sprintf("blk.%d.attn_output.weight", l), layer.O.A, layer.O.B, layer.O.In, layer.O.R, layer.O.Out},
			tpair{fmt.Sprintf("blk.%d.attn_q.weight", l), layer.Q.A, layer.Q.B, layer.Q.In, layer.Q.R, layer.Q.Out},
			tpair{fmt.Sprintf("blk.%d.attn_v.weight", l), layer.V.A, layer.V.B, layer.V.In, layer.V.R, layer.V.Out})
	}
	var ntensors int = len(pairs) * 2
	var f *os.File
	var err error
	f, err = os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var buf []byte = make([]byte, 0, 8<<20)
	var w32 = func(v uint32) { buf = append(buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }
	var w64 = func(v uint64) {
		buf = append(buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24),
			byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56))
	}
	var wstr = func(s string) {
		w64(uint64(len(s)))
		buf = append(buf, s...)
	}
	var wf32 = func(v float32) { w32(math.Float32bits(v)) }
	// header
	w32(0x46554747) // GGUF
	w32(3)          // version
	w64(uint64(ntensors))
	w64(6) // kv count
	// KVs
	var kv = func(key string, vtype uint32, payload func()) {
		wstr(key)
		w32(vtype)
		payload()
	}
	kv("general.architecture", 8, func() { wstr("qwen3") })
	kv("general.type", 8, func() { wstr("adapter") })
	kv("adapter.type", 8, func() { wstr("lora") })
	kv("general.name", 8, func() { wstr(name) })
	kv("adapter.lora.alpha", 6, func() { wf32(alpha) })
	kv("general.quantization_version", 4, func() { w32(2) })
	// info section: names, dims, types, offsets (patched below)
	var info []byte
	var iw32 = func(v uint32) { info = append(info, byte(v), byte(v>>8), byte(v>>16), byte(v>>24)) }
	var iw64 = func(v uint64) {
		info = append(info, byte(v), byte(v>>8), byte(v>>16), byte(v>>24),
			byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56))
	}
	var iwstr = func(s string) {
		iw64(uint64(len(s)))
		info = append(info, s...)
	}
	var offsets []uint64
	var i = 0
	for i = 0; i < len(pairs); i++ {
		var p = pairs[i]
		iwstr(p.base + ".lora_a")
		iw32(2)
		iw64(uint64(p.in))
		iw64(uint64(p.r))
		iw32(0) // type: f32
		iw64(0) // offset placeholder (patched below)
		offsets = append(offsets, 0)
		iwstr(p.base + ".lora_b")
		iw32(2)
		iw64(uint64(p.r))
		iw64(uint64(p.out))
		iw32(0)
		iw64(0)
		offsets = append(offsets, 0)
	}
	// data start: 32-aligned after header + info. Tensor offsets are
	// RELATIVE to the data section start (llama.cpp validates the first
	// tensor at offset 0 and accumulates per-tensor alignment).
	var dataStart uint64 = (uint64(len(buf)) + uint64(len(info)) + 31) &^ 31
	var off uint64 = 0
	{
		var i = 0
		for i = 0; i < len(pairs); i++ {
			offsets[i*2] = off
			off += uint64(len(pairs[i].a)) * 4
			offsets[i*2+1] = off
			off += uint64(len(pairs[i].b)) * 4
		}
	}
	// patch offsets into info (each offset field is the last 8 bytes of its entry)
	var scan uint64 = 0
	var oi int = 0
	for scan = 0; scan < uint64(len(info)); {
		var n uint64 = binary.LittleEndian.Uint64(info[scan : scan+8])
		scan += 8 + n
		var nd uint32 = binary.LittleEndian.Uint32(info[scan : scan+4])
		scan += 4 + uint64(nd)*8
		scan += 4 // type
		binary.LittleEndian.PutUint64(info[scan:scan+8], offsets[oi])
		oi++
		scan += 8
	}
	// assemble
	buf = append(buf, info...)
	for uint64(len(buf)) < dataStart {
		buf = append(buf, 0)
	}
	{
		var i = 0
		for i = 0; i < len(pairs); i++ {
			var a []byte = f32Bytes(pairs[i].a)
			var b []byte = f32Bytes(pairs[i].b)
			buf = append(buf, a...)
			buf = append(buf, b...)
		}
	}
	if _, err = f.Write(buf); err != nil {
		return err
	}
	return nil
}

func f32Bytes(v []float32) []byte {
	var out []byte = make([]byte, len(v)*4)
	var i int = 0
	for i = 0; i < len(v); i++ {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v[i]))
	}
	return out
}

// ---- training driver -----------------------------------------------------

func trainCmd(args []string) {
	var base, corpus, out, name string
	var ctx, stride, epochs int = 256, 128, 2
	var window int = 0 // 0 = full causal attention (parity harness default)
	var profilePath string = ""
	var i int = 0
	for i = 0; i < len(args); i++ {
		switch args[i] {
		case "--base":
			base = args[i+1]
			i++
		case "--corpus":
			corpus = args[i+1]
			i++
		case "--out":
			out = args[i+1]
			i++
		case "--name":
			name = args[i+1]
			i++
		case "--ctx":
			ctx = atoi(args[i+1])
			i++
		case "--stride":
			stride = atoi(args[i+1])
			i++
		case "--epochs":
			epochs = atoi(args[i+1])
			i++
		case "--threads":
			trainThreads = atoi(args[i+1])
			i++
		case "--window":
			window = atoi(args[i+1])
			i++
		case "--profile":
			profilePath = args[i+1]
			i++
		}
	}
	if base == "" || corpus == "" || out == "" {
		fmt.Println("usage: gglora train --base <model.gguf> --corpus <text> --out <adapter.gguf> [--ctx] [--stride] [--epochs] [--threads] [--window]")
		os.Exit(2)
	}
	if name == "" {
		name = "gotato-slice"
	}
	var t0 = time.Now()
	if profilePath != "" {
		var pf *os.File
		pf, _ = os.Create(profilePath)
		if pf != nil {
			_ = pprof.StartCPUProfile(pf)
			defer pprof.StopCPUProfile()
		}
	}
	fmt.Printf("[train] loading %s ...\n", base)
	var m, err = loadModel(base)
	if err != nil {
		fmt.Println("load:", err)
		os.Exit(1)
	}
	m.Window = window
	fmt.Printf("[train] model: %d layers, hidden %d, heads %d/%d, ffn %d, vocab %d (%.0fs)\n",
		m.NLayer, m.Hidden, m.NHead, m.NKvHead, m.Ffn, m.Vocab, time.Since(t0).Seconds())
	var g *ggufFile

	g, err = loadGGUF(base)
	if err != nil {
		fmt.Println("load gguf:", err)
		os.Exit(1)
	}
	var tok = newTokenizer(g)
	g.unmap() // weights are dequantized into RAM; drop the file mapping
	var corpusData []byte

	corpusData, err = os.ReadFile(corpus)
	if err != nil {
		fmt.Println("corpus:", err)
		os.Exit(1)
	}
	var ids = tok.Encode(string(corpusData))
	fmt.Printf("[train] corpus: %d chars -> %d tokens\n", len(corpusData), len(ids))
	if len(ids) < 64 {
		fmt.Println("corpus too small")
		os.Exit(1)
	}
	// sliding windows
	var samples [][]int
	for i = 0; i+ctx <= len(ids); i += stride {
		var s []int = make([]int, ctx)
		copy(s, ids[i:i+ctx])
		samples = append(samples, s)
	}
	if len(samples) == 0 {
		// corpus shorter than ctx: train on the whole sequence as one sample
		var s []int = make([]int, len(ids))
		copy(s, ids)
		samples = append(samples, s)
	}
	fmt.Printf("[train] %d samples (ctx %d, stride %d), %d threads\n",
		len(samples), ctx, stride, trainThreads)
	var s *scratch = newScratch(ctx, m, trainThreads)
	// the manual heap (heap.go): one flat arena sized for the worst sample.
	// Per sample the arena holds every h (7 projections x layers x R),
	// every dyh backward row, and the dXHead buffer; heapReset frees it
	// all at the next sample boundary. No make() ever runs on the hot path.
	var perSample int = 14*m.NLayer*ctx*loraRank + ctx*m.Hidden
	heapInit(perSample)
	fmt.Printf("[train] manual heap: %.1f MB arena, reset per sample (no GC)\n", float64(heapBytes())/1e6)
	var step int = 0
	var ep = 1
	for ep = 1; ep <= epochs; ep++ {
		var tot float32 = 0
		var si int = 0
		for si = 0; si < len(samples); si++ {
			step++
			heapReset() // explicit free: the previous sample's h/dyh/dXHead
			var tokens []int = samples[si]
			// zero grads
			var l int = 0
			for l = 0; l < m.NLayer; l++ {
				var layer = m.Layers[l]
				layer.Q.zeroGrads()
				layer.K.zeroGrads()
				layer.V.zeroGrads()
				layer.O.zeroGrads()
				layer.Gate.zeroGrads()
				layer.Up.zeroGrads()
				layer.Down.zeroGrads()
			}
			s.T = len(tokens) // short corpora: the sample may be shorter than ctx
			var loss = m.forward(s, tokens)
			if math.IsNaN(float64(loss)) {
				fmt.Println("[train] NaN loss - aborting")
				os.Exit(1)
			}
			m.backward(s, tokens)
			// gradient clipping (per-layer max norm 1.0): one exploding
			// sample (long file tails, rare tokens) must not NaN the
			// AdamW state - measured: stable ~11.96 then NaN on the last
			// sample of epoch 1 without it.
			for l = 0; l < m.NLayer; l++ {
				var layer = m.Layers[l]
				clipGrads(layer.Q.dA, layer.Q.dB)
				clipGrads(layer.K.dA, layer.K.dB)
				clipGrads(layer.V.dA, layer.V.dB)
				clipGrads(layer.O.dA, layer.O.dB)
				clipGrads(layer.Gate.dA, layer.Gate.dB)
				clipGrads(layer.Up.dA, layer.Up.dB)
				clipGrads(layer.Down.dA, layer.Down.dB)
			}
			// each layer's LoRA update is independent: one parallel pass
			// over the 28 layers instead of a sequential chain of 7*28 steps
			parallel(m.NLayer, func(lo int, hi int) {
				var l int = lo
				for l = lo; l < hi; l++ {
					var layer = m.Layers[l]
					layer.Q.adamStep(step)
					layer.K.adamStep(step)
					layer.V.adamStep(step)
					layer.O.adamStep(step)
					layer.Gate.adamStep(step)
					layer.Up.adamStep(step)
					layer.Down.adamStep(step)
				}
			})
			tot += loss
			if si%10 == 9 || si == len(samples)-1 {
				fmt.Printf("[train] epoch %d/%d sample %d/%d loss %.4f (%.0fs)\n",
					ep, epochs, si+1, len(samples), tot/float32(si+1), time.Since(t0).Seconds())
			}
		}
		fmt.Printf("[train] epoch %d loss %.4f\n", ep, tot/float32(len(samples)))
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		fmt.Printf("[mem] heap_alloc %.0f MB heap_objects %d num_gc %d arena %.1f MB (flat = no leak)\n",
			float64(ms.HeapAlloc)/1e6, ms.HeapObjects, ms.NumGC, float64(heapBytes())/1e6)
	}
	{
		var err = writeAdapterGGUF(out, name, m, 32)
		if err != nil {
			fmt.Println("write adapter:", err)
			os.Exit(1)
		}
	}
	fmt.Printf("[train] adapter -> %s (%.0fs total)\n", out, time.Since(t0).Seconds())
}

// ---- tokenizer check ------------------------------------------------------

func tokcheckCmd(args []string) {
	var base, text, server string
	var verify bool
	var i int = 0
	for i = 0; i < len(args); i++ {
		switch args[i] {
		case "--base":
			base = args[i+1]
			i++
		case "--text":
			text = args[i+1]
			i++
		case "--verify":
			verify = true
		case "--server":
			server = args[i+1]
			i++
		}
	}
	if base == "" || text == "" {
		fmt.Println("usage: gglora tokcheck --base <model.gguf> --text <s> [--verify --server :8082]")
		os.Exit(2)
	}
	var g, err = loadGGUF(base)
	if err != nil {
		fmt.Println("load:", err)
		os.Exit(1)
	}
	var tok = newTokenizer(g)
	var ids = tok.Encode(text)
	fmt.Println("ids:", ids)
	if verify {
		var body []byte

		body, _ = json.Marshal(map[string]any{"content": text, "add_special": false})
		var resp, err = http.Post(server+"/tokenize", "application/json", strings.NewReader(string(body)))
		if err != nil {
			fmt.Println("server:", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		var raw []byte

		raw, _ = io.ReadAll(resp.Body)
		var out struct {
			Tokens []int `json:"tokens"`
		}
		if json.Unmarshal(raw, &out) != nil {
			fmt.Println("server response:", string(raw[:200]))
			os.Exit(1)
		}
		var ok = len(out.Tokens) == len(ids)
		if ok {
			for i = 0; i < len(ids); i++ {
				if ids[i] != out.Tokens[i] {
					ok = false
					fmt.Printf("MISMATCH at %d: got %d want %d\n", i, ids[i], out.Tokens[i])
					break
				}
			}
		}
		if ok {
			fmt.Printf("tokcheck VERIFIED: %d ids match llama-server\n", len(ids))
		} else {
			fmt.Printf("tokcheck FAILED: len got %d want %d\n", len(ids), len(out.Tokens))
			os.Exit(1)
		}
	}
	_ = strconv.Itoa
}
