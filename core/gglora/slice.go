// slice.go - ggslice: turn the salience accumulator into a sliced GGUF.
//
//	gglora mask --base <model.gguf> --domain <name> --out <masks/name.gguf>
//	            [--heads-keep 0.8] [--neurons-keep 0.35]
//
// A domain mask is a QUERY over the accumulator: specificity(neuron) =
// domain_mean / global_mean (TF-IDF over neurons). The top-k heads and
// neurons by specificity are KEPT; everything else is ZEROED in a copy of
// the base GGUF (Q8_0 int8 values -> 0 is exact). The result is a fully
// valid model file - llama-server loads it with -m masks/name.gguf, no
// adapter needed. Storage is cheap: one 635MB file per domain.
//
// Zeroing is exact in Q8_0 (block = f16 scale + 32 int8; q=0 -> x=0), so
// no requantization is involved.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// maskCmd: compute the domain mask and write the sliced GGUF.
func maskCmd(args []string) {
	var base, domain, out string
	var headsKeep, neuronsKeep float64 = 0.8, 0.35
	var i = 0
	for i = 0; i < len(args); i++ {
		switch args[i] {
		case "--base":
			base = args[i+1]
			i++
		case "--domain":
			domain = args[i+1]
			i++
		case "--out":
			out = args[i+1]
			i++
		case "--heads-keep":
			headsKeep = atof(args[i+1])
			i++
		case "--neurons-keep":
			neuronsKeep = atof(args[i+1])
			i++
		}
	}
	if base == "" || domain == "" || out == "" {
		fmt.Println("usage: gglora mask --base <model.gguf> --domain <name> --out <masks/name.gguf> [--heads-keep 0.8] [--neurons-keep 0.35]")
		os.Exit(2)
	}
	var m, err = loadModel(base)
	if err != nil {
		fmt.Println("load:", err)
		os.Exit(1)
	}
	var dom = loadSalience(domain, m)
	if dom.Tokens == 0 {
		fmt.Println("no salience for domain", domain, "- run collect first")
		os.Exit(1)
	}
	var glob = foldGlobal(m)
	if glob.Tokens == 0 {
		glob = dom // single-domain bootstrap: specificity vs itself is 1.0
	}
	// specificity scores
	var heads = make([]float64, len(dom.Heads))
	{
		var i int

		for i = range heads {
			var dm = dom.Heads[i] / float64(dom.Tokens)
			var gm = glob.Heads[i] / float64(glob.Tokens)
			if gm > 0 {
				heads[i] = dm / gm
			}
		}
	}
	var neurons = make([]float64, len(dom.Neurons))
	{
		var i int

		for i = range neurons {
			var dm = dom.Neurons[i] / float64(dom.Tokens)
			var gm = glob.Neurons[i] / float64(glob.Tokens)
			if gm > 0 {
				neurons[i] = dm / gm
			}
		}
	}
	// top-k by specificity
	var headKeep = maskSelect(heads, headsKeep)
	var neuronKeep = maskSelect(neurons, neuronsKeep)
	var hk, nk int = 0, 0
	var v bool

	for _, v = range headKeep {
		if v {
			hk++
		}
	}
	{
		var v bool

		for _, v = range neuronKeep {
			if v {
				nk++
			}
		}
	}
	fmt.Printf("[mask] domain=%s tokens=%d: keeping %d/%d heads, %d/%d neurons\n",
		domain, dom.Tokens, hk, len(heads), nk, len(neurons))
	{

		var err = writeMaskedGGUF(base, out, m, headKeep, neuronKeep)
		if err != nil {
			fmt.Println("mask write:", err)
			os.Exit(1)
		}
	}
	{
		// record the mask in index.json alongside the adapters (fleet-relative)
		var rel, err = filepath.Rel(fleetRoot, out)
		if err == nil {
			out = rel
		}
	}
	recordMaskIndex(domain, out, hk, len(heads), nk, len(neurons))
	fmt.Printf("[mask] sliced model -> %s\n", out)
}

// maskSelect: indices whose specificity ranks in the top `keep` fraction.
// Returns a boolean mask (true = KEEP the element).
func maskSelect(scores []float64, keep float64) []bool {
	var n = len(scores)
	var order = make([]int, n)
	var i int

	for i = range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return scores[order[a]] > scores[order[b]] })
	var out = make([]bool, n)
	var k = int(float64(n) * keep)
	if k < 1 {
		k = 1
	}
	{
		var i = 0
		for i = 0; i < k && i < n; i++ {
			out[order[i]] = true
		}
	}
	return out
}

// writeMaskedGGUF: copy the base file and zero the masked weights.
// headKeep/neuronKeep are indexed [layer*nHead+head] / [layer*ffn+neuron].
func writeMaskedGGUF(base string, out string, m *model, headKeep []bool, neuronKeep []bool) error {
	var src, err = os.Open(base)
	if err != nil {
		return err
	}
	defer src.Close()
	var dst *os.File

	dst, err = os.Create(out)
	if err != nil {
		return err
	}
	defer dst.Close()
	if _, err = io.Copy(dst, src); err != nil {
		return err
	}
	// parse the base to locate the tensors
	var g *ggufFile

	g, err = loadGGUF(base)
	if err != nil {
		return err
	}
	// Q8_0 storage: each block covers 32 consecutive elements of the row-major
	// [in,out] layout (out-major: element (k,j) at j*in+k). A row j occupies
	// element offsets [j*in, (j+1)*in) -> blocks [j*in/32, (j+1)*in/32).
	// Each block is 34 bytes at data + blockIndex*34.
	var zeroOutRows = func(name string, rowStart int, rowCount int, in int) error {
		var info = g.findTensor(name)
		if info == nil || info.Type != ggmlQ8_0 {
			return fmt.Errorf("tensor %s missing or not Q8_0", name)
		}
		var baseOff = int64(g.align + info.Offset)
		var payload [32]byte
		var j = rowStart
		for j = rowStart; j < rowStart+rowCount; j++ {
			var block0 = int64(j) * int64(in) / 32
			var block1 = int64(j+1) * int64(in) / 32
			var b = block0
			for b = block0; b < block1; b++ {
				var err error

				_, err = dst.WriteAt(payload[:], baseOff+b*34+2)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}
	var zeroInCols = func(name string, colStart int, colLen int, in int) error {
		// zero input columns [colStart, colStart+colLen) across all out rows:
		// element (k, j) at j*in+k -> the payload byte inside its block.
		var info = g.findTensor(name)
		if info == nil || info.Type != ggmlQ8_0 {
			return fmt.Errorf("tensor %s missing or not Q8_0", name)
		}
		var outDim = int(info.Dims[1])
		var baseOff = int64(g.align + info.Offset)
		var j = 0
		for j = 0; j < outDim; j++ {
			var k = colStart
			for k = colStart; k < colStart+colLen; k++ {
				var idx = int64(j)*int64(in) + int64(k)
				var blk = idx / 32
				var pos = idx % 32
				var one [1]byte
				var err error

				_, err = dst.WriteAt(one[:], baseOff+blk*34+2+pos)
				if err != nil {
					return err
				}
			}
		}
		return nil
	}
	// apply the mask per layer.
	// Layout note ([in, out], out-major: element (k,j) at j*in+k):
	//   q/k/v  carry HEADS on the OUT dimension -> head h = out rows
	//          [h*hd, (h+1)*hd);
	//   attn_output carries HEADS on the IN dimension -> head h = in cols
	//          [h*hd, (h+1)*hd) across all out rows;
	//   gate/up carry NEURONS on the OUT dimension -> neuron j = out row j.
	var hd = m.HeadDim
	var l = 0
	for l = 0; l < m.NLayer; l++ {
		var hp = 0
		for hp = 0; hp < m.NHead; hp++ {
			if headKeep[l*m.NHead+hp] {
				continue
			}
			var err = zeroOutRows(fmt.Sprintf("blk.%d.attn_q.weight", l), hp*hd, hd, m.Hidden)
			if err != nil {
				return err
			}
			var kg = hp % m.NKvHead
			{
				var err = zeroOutRows(fmt.Sprintf("blk.%d.attn_k.weight", l), kg*hd, hd, m.Hidden)
				if err != nil {
					return err
				}
			}
			{
				var err = zeroOutRows(fmt.Sprintf("blk.%d.attn_v.weight", l), kg*hd, hd, m.Hidden)
				if err != nil {
					return err
				}
			}
			{
				var err = zeroInCols(fmt.Sprintf("blk.%d.attn_output.weight", l), hp*hd, hd, m.NHead*hd)
				if err != nil {
					return err
				}
			}
		}
		var j = 0
		for j = 0; j < m.Ffn; j++ {
			if neuronKeep[l*m.Ffn+j] {
				continue
			}
			var err = zeroOutRows(fmt.Sprintf("blk.%d.ffn_gate.weight", l), j, 1, m.Hidden)
			if err != nil {
				return err
			}
			{
				var err = zeroOutRows(fmt.Sprintf("blk.%d.ffn_up.weight", l), j, 1, m.Hidden)
				if err != nil {
					return err
				}
			}
		}
	}
	return dst.Sync()
}

// recordMaskIndex: publish the mask into index.json (same file the gateway
// reads) so autostart can serve masks/python.gguf like an adapter.
func recordMaskIndex(domain string, maskFile string, headsKept int, headsTotal int, neuronsKept int, neuronsTotal int) {
	var idxPath = filepath.Join(fleetRoot, "index.json")
	var idx = map[string]any{}
	{
		var data, err = os.ReadFile(idxPath)
		if err == nil {
			_ = json.Unmarshal(data, &idx)
		}
	}
	var e map[string]any

	e, _ = idx[domain].(map[string]any)
	if e == nil {
		e = map[string]any{}
	}
	e["status"] = "ready"
	e["mask"] = maskFile
	e["heads_kept"] = headsKept
	e["heads_total"] = headsTotal
	e["neurons_kept"] = neuronsKept
	e["neurons_total"] = neuronsTotal
	e["kind"] = "mask"
	idx[domain] = e
	var data []byte

	data, _ = json.MarshalIndent(idx, "", " ")
	_ = os.WriteFile(idxPath, data, 0644)
}

func atof(s string) float64 {
	var v float64
	_, _ = fmt.Sscanf(s, "%f", &v)
	if math.IsNaN(v) || v <= 0 {
		return 1
	}
	return v
}
