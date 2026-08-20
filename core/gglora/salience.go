// salience.go - the domain salience accumulator: the "more memory for
// indexing" design. One forward pass per corpus sample folds per-head and
// per-neuron activation moments into a tiny RAM-resident accumulator
// (~2MB per domain); a domain mask is then a QUERY over the accumulator
// (specificity = domain_mean / global_mean, TF-IDF over neurons), not a
// training run.
//
// RAM cost per domain: heads 28*16 floats + neurons 28*2*3072 floats of
// energy + a token counter - a few MB, inside the reserved-index budget.
// Storage: salience/<domain>.json (storage is cheap; reload on demand).
package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// salCollector: activation moments for one domain corpus.
type salCollector struct {
	Domain  string    `json:"domain"`
	Tokens  int64     `json:"tokens"`
	Heads   []float64 `json:"heads"`   // [layer*nHead + head] sum of per-token head energies
	Neurons []float64 `json:"neurons"` // [layer*ffn + neuron] sum of per-token neuron energies (gate, post-silu)
}

func newSalCollector(domain string, nLayer int, nHead int, ffn int) *salCollector {
	return &salCollector{
		Domain:  domain,
		Heads:   make([]float64, nLayer*nHead),
		Neurons: make([]float64, nLayer*ffn),
	}
}

// accumulateLayer: fold ONE layer's activation energies (called from inside
// forward, where the per-layer scratch buffers are still live).
func (c *salCollector) accumulateLayer(s *scratch, m *model, l int, t int) {
	var base = l * m.NHead
	var hp = 0
	for hp = 0; hp < m.NHead; hp++ {
		var e float64 = 0
		var pos = 0
		for pos = 0; pos < t; pos++ {
			var acc float64 = 0
			var ob = pos*m.NHead*m.HeadDim + hp*m.HeadDim
			var d = 0
			for d = 0; d < m.HeadDim; d++ {
				var v = float64(s.AttnOut[ob+d])
				acc += v * v
			}
			e += acc
		}
		c.Heads[base+hp] += e
	}
	var nb = l * m.Ffn
	var j = 0
	for j = 0; j < m.Ffn; j++ {
		var e float64 = 0
		var pos = 0
		for pos = 0; pos < t; pos++ {
			var v = float64(s.Gate[pos*m.Ffn+j])
			e += v * v
		}
		c.Neurons[nb+j] += e
	}
}

var salienceDir string = "salience-data"
var fleetRoot string = "."

func init() {
	var v = os.Getenv("GOTATO_FLEET")
	if v != "" {
		salienceDir = v + "/salience"
		fleetRoot = v
	}
}

func saliencePath(domain string) string {
	return filepath.Join(salienceDir, domain+".json")
}

func loadSalience(domain string, m *model) *salCollector {
	var c = newSalCollector(domain, m.NLayer, m.NHead, m.Ffn)
	var data, err = os.ReadFile(saliencePath(domain))
	if err != nil {
		return c
	}
	if json.Unmarshal(data, c) == nil && len(c.Heads) == m.NLayer*m.NHead {
		return c
	}
	return newSalCollector(domain, m.NLayer, m.NHead, m.Ffn)
}

func saveSalience(c *salCollector) {
	var dir = filepath.Dir(saliencePath(c.Domain))
	if os.MkdirAll(dir, 0755) != nil {
		return
	}
	var data []byte

	data, _ = json.Marshal(c)
	_ = os.WriteFile(saliencePath(c.Domain), data, 0644)
}

// foldGlobal: global moments = sum over all domain accumulators on disk.
func foldGlobal(m *model) *salCollector {
	var g = newSalCollector("global", m.NLayer, m.NHead, m.Ffn)
	var dir = salienceDir
	var entries, err = os.ReadDir(dir)
	if err != nil {
		return g
	}
	var e fs.DirEntry

	for _, e = range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		var domain = e.Name()[:len(e.Name())-5]
		var c = loadSalience(domain, m)
		if c.Tokens == 0 {
			continue
		}
		var i int

		for i = range g.Heads {
			g.Heads[i] += c.Heads[i]
		}
		{
			var i int

			for i = range g.Neurons {
				g.Neurons[i] += c.Neurons[i]
			}
		}
		g.Tokens += c.Tokens
	}
	return g
}

// collectCmd: gglora collect --base <model.gguf> --corpus <text> --domain <name>
// One forward pass per sliding window; folds into salience/<domain>.json.
func collectCmd(args []string) {
	var base, corpus, domain string
	var ctx, stride int = 256, 128
	var i = 0
	for i = 0; i < len(args); i++ {
		switch args[i] {
		case "--base":
			base = args[i+1]
			i++
		case "--corpus":
			corpus = args[i+1]
			i++
		case "--domain":
			domain = args[i+1]
			i++
		case "--ctx":
			ctx = atoi(args[i+1])
			i++
		case "--stride":
			stride = atoi(args[i+1])
			i++
		case "--threads":
			trainThreads = atoi(args[i+1])
			i++
		}
	}
	if base == "" || corpus == "" || domain == "" {
		fmt.Println("usage: gglora collect --base <model.gguf> --corpus <text> --domain <name> [--ctx 256] [--stride 128]")
		os.Exit(2)
	}
	var m, err = loadModel(base)
	if err != nil {
		fmt.Println("load:", err)
		os.Exit(1)
	}
	var g *ggufFile

	g, err = loadGGUF(base)
	if err != nil {
		fmt.Println("load gguf:", err)
		os.Exit(1)
	}
	var tok = newTokenizer(g)
	g.unmap()
	var data []byte

	data, err = os.ReadFile(corpus)
	if err != nil {
		fmt.Println("corpus:", err)
		os.Exit(1)
	}
	var ids = tok.Encode(string(data))
	if len(ids) < 32 {
		fmt.Println("corpus too small")
		os.Exit(1)
	}
	var samples [][]int
	{
		var i = 0
		for i = 0; i+ctx <= len(ids); i += stride {
			var s = make([]int, ctx)
			copy(s, ids[i:i+ctx])
			samples = append(samples, s)
		}
	}
	if len(samples) == 0 {
		// corpus shorter than ctx: train on the whole sequence as one sample
		samples = append(samples, ids)
	}
	fmt.Printf("[collect] domain=%s %d samples (ctx %d, stride %d)\n", domain, len(samples), ctx, stride)
	var c = loadSalience(domain, m)
	var scr = newScratch(ctx, m, 1)
	// the manual heap: forward-only, so the arena only needs the per-sample
	// h tensors (7 projections x layers x R); reset per sample.
	heapInit(7*m.NLayer*ctx*loraRank + ctx*m.Hidden)
	var t0 = time.Now().UnixMilli()
	var si int

	var s []int

	for si, s = range samples {
		heapReset()
		scr.T = len(s) // short corpora: the sample may be shorter than ctx
		m.Col = c
		m.forward(scr, s)
		m.Col = nil
		c.Tokens += int64(ctx)
		if si%20 == 19 || si == len(samples)-1 {
			fmt.Printf("[collect] %d/%d samples, tokens=%d (%.0fs)\n",
				si+1, len(samples), c.Tokens, float64(time.Now().UnixMilli()-t0)/1000)
		}
	}
	saveSalience(c)
	fmt.Printf("[collect] saved salience/%s.json: %d tokens, %d heads, %d neurons\n",
		domain, c.Tokens, len(c.Heads), len(c.Neurons))
}
