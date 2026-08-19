// model.go - full Qwen3-0.6B forward/backward with LoRA on the 7 target
// projections, stdlib only, explicit loops (the gglora philosophy: the math
// is written out so a future SIMD/C port can verify against it).
//
// Storage follows GGUF exactly: every weight is logically [in, out] and
// stored OUT-major (element (k, j) at offset j*in + k), i.e. y[j] =
// sum_k x[k]*W[j*in+k]. token_embd is [hidden, vocab] as stored; token id
// picks the contiguous 1024-float block at id*hidden.
package main

import (
	"fmt"
	"math"
)

// ---- LoRA linear ----------------------------------------------------------

type loraLayer struct {
	In, Out, R int
	Alpha      float32
	W          []float32 // frozen base weights [in*out]
	WT         []float32 // transposed copy [out*in] for the AVX2 bwd kernel
	A          []float32 // [in*r], offset r'*in + k
	B          []float32 // [r*out], offset j*r + r'
	mA         []float32 // AdamW first moments
	vA         []float32 // AdamW second moments
	mB         []float32
	vB         []float32
	// per-sample gradient accumulators (zeroed before each backward)
	dA []float32
	dB []float32
}

func newLoraLayer(W []float32, in int, out int, r int, alpha float32) *loraLayer {
	l := &loraLayer{In: in, Out: out, R: r, Alpha: alpha,
		W: W, A: make([]float32, in*r), B: make([]float32, r*out),
		mA: make([]float32, in*r), vA: make([]float32, in*r),
		mB: make([]float32, r*out), vB: make([]float32, r*out),
		dA: make([]float32, in*r), dB: make([]float32, r*out)}
	// WT[i*out+j] = W[j*in+i]: the bwd kernel reads W rows contiguously.
	l.WT = make([]float32, in*out)
	var i int = 0
	for i = 0; i < in; i++ {
		var j int = 0
		for j = 0; j < out; j++ {
			l.WT[i*out+j] = W[j*in+i]
		}
	}
	// A ~ U(-sqrt(1/r), sqrt(1/r)); B = 0 (standard LoRA init)
	var bound float32 = float32(math.Sqrt(float64(1.0 / r)))
	for i = 0; i < len(l.A); i++ {
		l.A[i] = (randF32()*2 - 1) * bound
	}
	return l
}

// dot4: unrolled dot product - 4 independent accumulator chains break the
// FMA latency chain (a single scalar accumulator stalls ~4-8 cycles per
// element; 4 chains run at throughput). ~3-4x on the FFN/head GEMMs, pure
// Go, no deps. NOTE: the grouped final sum changes float rounding vs the
// scalar loop - gglora is now the reference math (the torch parity harness
// was retired with the python trainer).
func dot4(x []float32, w []float32, n int) float32 {
	var acc0 float32 = 0
	var acc1 float32 = 0
	var acc2 float32 = 0
	var acc3 float32 = 0
	var i int = 0
	for i = 0; i+4 <= n; i += 4 {
		acc0 += x[i] * w[i]
		acc1 += x[i+1] * w[i+1]
		acc2 += x[i+2] * w[i+2]
		acc3 += x[i+3] * w[i+3]
	}
	for ; i < n; i++ {
		acc0 += x[i] * w[i]
	}
	return (acc0 + acc1) + (acc2 + acc3)
}

// axpy4: 4-wide streaming dy*w += dx (memory-bound; ILP helps the copy).
func axpy4(dx []float32, w []float32, n int, v float32) {
	var i int = 0
	for i = 0; i+4 <= n; i += 4 {
		dx[i] += v * w[i]
		dx[i+1] += v * w[i+1]
		dx[i+2] += v * w[i+2]
		dx[i+3] += v * w[i+3]
	}
	for ; i < n; i++ {
		dx[i] += v * w[i]
	}
}

// fwdW: y[m*out+j] += sum_k x[m*in+k]*W[j*in+k]. The actual GEMM runs in
// the AVX2 C kernel (gemm.c, via gemmFwdSimd) - 8 output rows x 8 k-lanes
// with FMA memory operands, ~6x the pure-Go loop (Go has no auto-
// vectorizer). Row-chunked so all cores work. The scalar fallback kicks
// in without cgo/amd64.
func (l *loraLayer) fwdW(y []float32, x []float32, m int) {
	gemmFwdSimd(y, x, l.W, m, l.In, l.Out)
}

// fwd: y[m*out+j] += sum_k x[m*in+k]*W[j*in+k] + (alpha/r) * sum_r' h[m*r+r']*B[j*r+r']
// where h[m*r+r'] = sum_k x[m*in+k]*A[r'*in+k]. Returns h (needed by back).
func (l *loraLayer) fwd(y []float32, x []float32, m int) []float32 {
	var h []float32 = make([]float32, m*l.R)
	l.fwdW(y, x, m)
	parallel(m, func(lo int, hi int) {
		var row int = lo
		for row = lo; row < hi; row++ {
			var xr int = row * l.In
			var rp int = 0
			for rp = 0; rp < l.R; rp++ {
				h[row*l.R+rp] = dot4(x[xr:xr+l.In], l.A[rp*l.In:rp*l.In+l.In], l.In)
			}
			var scale float32 = l.Alpha / float32(l.R)
			var j int = 0
			for j = 0; j < l.Out; j++ {
				y[row*l.Out+j] += scale * dot4(h[row*l.R:row*l.R+l.R], l.B[j*l.R:j*l.R+l.R], l.R)
			}
		}
	})
	return h
}

// back: gradients of the LoRA path + the frozen base path. dA/dB accumulate
// into the layer's own buffers (zeroed via zeroGrads before use).
func (l *loraLayer) back(dx []float32, dy []float32, x []float32, h []float32, m int) {
	// W pass: dx[m*in] += dy[m*out]·WT[in*out] via the AVX2 kernel on the
	// pre-transposed weights (chunked internally over the reduction dim;
	// dx was zeroed by the caller).
	gemmBwdSimd(dx, dy, l.WT, m, l.In, l.Out)
	var dyh []float32 = make([]float32, m*l.R)
	var scale float32 = l.Alpha / float32(l.R)
	parallel(m, func(lo int, hi int) {
		var row int = lo
		for row = lo; row < hi; row++ {
			var xr int = row * l.In
			var yr int = row * l.Out
			// dyh[rp] = scale * sum_j dy[j]*B[j*R+rp]; B is row-major so each
			// 16-float row is read once and all rp outputs accumulate per j
			// (the old j-outer/rp-inner order touched a different cache line
			// per element - 16x more B traffic).
			var rp int = 0
			for rp = 0; rp < l.R; rp++ {
				dyh[row*l.R+rp] = 0
			}
			var j int = 0
			for j = 0; j < l.Out; j++ {
				var dyv float32 = dy[yr+j]
				var bj int = j * l.R
				for rp = 0; rp < l.R; rp++ {
					dyh[row*l.R+rp] += dyv * l.B[bj+rp]
				}
			}
			for rp = 0; rp < l.R; rp++ {
				dyh[row*l.R+rp] *= scale
			}
			for rp = 0; rp < l.R; rp++ {
				axpy4(dx[xr:xr+l.In], l.A[rp*l.In:rp*l.In+l.In], l.In, dyh[row*l.R+rp])
			}
			for rp = 0; rp < l.R; rp++ {
				var dyhv float32 = dyh[row*l.R+rp]
				var ar int = rp * l.In
				var k int = 0
				for k = 0; k < l.In; k++ {
					l.dA[ar+k] += dyhv * x[xr+k]
				}
			}
			for j = 0; j < l.Out; j++ {
				var dyv float32 = dy[yr+j] * scale
				var bj int = j * l.R
				var rp int = 0
				for rp = 0; rp < l.R; rp++ {
					l.dB[bj+rp] += dyv * h[row*l.R+rp]
				}
			}
		}
	})
}

func (l *loraLayer) zeroGrads() {
	var i int = 0
	for i = 0; i < len(l.dA); i++ {
		l.dA[i] = 0
	}
	for i = 0; i < len(l.dB); i++ {
		l.dB[i] = 0
	}
}

// adamStep: AdamW on A and B (explicit, matches the verified PoC constants).
func (l *loraLayer) adamStep(step int) {
	var beta1 float32 = 0.9
	var beta2 float32 = 0.999
	var eps float32 = 1e-8
	var lr float32 = 5e-5 // lowered from 2e-4: the 46KB html-css corpus
	// diverged to NaN on its last epoch-1 sample at 2e-4 (the clipping
	// contains updates but cannot heal an exploding forward); correctness
	// over speed - smaller corpora train a bit slower but stay finite.
	var bc1 float32 = 1 - float32(math.Pow(float64(beta1), float64(step)))
	var bc2 float32 = 1 - float32(math.Pow(float64(beta2), float64(step)))
	var i int = 0
	for i = 0; i < len(l.A); i++ {
		l.mA[i] = beta1*l.mA[i] + (1-beta1)*l.dA[i]
		l.vA[i] = beta2*l.vA[i] + (1-beta2)*l.dA[i]*l.dA[i]
		l.A[i] -= lr * (l.mA[i]/bc1) / (float32(math.Sqrt(float64(l.vA[i]/bc2))) + eps)
	}
	for i = 0; i < len(l.B); i++ {
		l.mB[i] = beta1*l.mB[i] + (1-beta1)*l.dB[i]
		l.vB[i] = beta2*l.vB[i] + (1-beta2)*l.dB[i]*l.dB[i]
		l.B[i] -= lr * (l.mB[i]/bc1) / (float32(math.Sqrt(float64(l.vB[i]/bc2))) + eps)
	}
}

// ---- RMSNorm ---------------------------------------------------------------

func rmsNorm(y []float32, x []float32, w []float32, n int, eps float32) {
	var i int = 0
	var ss float32 = 0
	for i = 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	var inv float32 = 1 / float32(math.Sqrt(float64(ss/float32(n))+float64(eps)))
	for i = 0; i < n; i++ {
		y[i] = x[i] * inv * w[i]
	}
}

// rmsNormBack: dx += w*inv*(dy - x*inv*sum(dy*x)/n)
func rmsNormBack(dx []float32, dy []float32, x []float32, w []float32, n int, eps float32) {
	var ss float32 = 0
	var i int = 0
	for i = 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	var inv float32 = 1 / float32(math.Sqrt(float64(ss/float32(n))+float64(eps)))
	var dot float32 = 0
	for i = 0; i < n; i++ {
		dot += dy[i] * x[i]
	}
	var factor float32 = inv * dot / float32(n)
	for i = 0; i < n; i++ {
		dx[i] += w[i] * inv * (dy[i] - x[i]*factor)
	}
}

// ---- RoPE (Qwen3: full head_dim rotation, theta 1e6) ----------------------

// ---- RoPE (Qwen3: full head_dim rotation, theta 1e6) ----------------------
// The cos/sin per (position, dim-half) are deterministic, so they are
// precomputed ONCE per training run (ropeEnsure, called from forward/back-
// ward) and table-looked-up in the hot loops - math.Cos/Sin are ~40-100
// cycles each and ropeApply runs ~4M times per step.
var ropeCosT []float32
var ropeSinT []float32
var ropeCachedHeadDim int = 0
var ropeCachedTheta float64 = 0

// ropeEnsure: grow the tables to cover positions [0, t).
func ropeEnsure(t int, headDim int, theta float64) {
	if ropeCachedHeadDim == headDim && ropeCachedTheta == theta {
		var need int = t * (headDim / 2)
		if len(ropeCosT) >= need {
			return
		}
	}
	var half int = headDim / 2
	var n int = t * half
	ropeCosT = make([]float32, n)
	ropeSinT = make([]float32, n)
	var pos int = 0
	for pos = 0; pos < t; pos++ {
		var i int = 0
		for i = 0; i < half; i++ {
			var freq float64 = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			var angle float64 = float64(pos) * freq
			ropeCosT[pos*half+i] = float32(math.Cos(angle))
			ropeSinT[pos*half+i] = float32(math.Sin(angle))
		}
	}
	ropeCachedHeadDim = headDim
	ropeCachedTheta = theta
}

func ropeApply(q []float32, t int, headDim int, theta float64) {
	var half int = headDim / 2
	var base int = t * half
	var i int = 0
	for i = 0; i < half; i++ {
		var c float64 = float64(ropeCosT[base+i])
		var s float64 = float64(ropeSinT[base+i])
		var a float32 = q[2*i]
		var b float32 = q[2*i+1]
		q[2*i] = float32(float64(a)*c - float64(b)*s)
		q[2*i+1] = float32(float64(a)*s + float64(b)*c)
	}
}

func ropeBack(dq []float32, t int, headDim int, theta float64) {
	var half int = headDim / 2
	var base int = t * half
	var i int = 0
	for i = 0; i < half; i++ {
		var c float64 = float64(ropeCosT[base+i])
		var s float64 = float64(ropeSinT[base+i])
		var a float32 = dq[2*i]
		var b float32 = dq[2*i+1]
		dq[2*i] = float32(float64(a)*c + float64(b)*s)
		dq[2*i+1] = float32(-float64(a)*s + float64(b)*c)
	}
}

// ---- the model ------------------------------------------------------------

type model struct {
	// Col: when set, forward() folds per-layer activation energies into the
	// salience accumulator (the ggslice collection mode; nil = no overhead).
	Col *salCollector
	// Window: sliding-window attention in the training loop (0 = full
	// causal attention). The train command defaults to 0 so the parity
	// harness stays bit-exact; expertd build passes 128 (O(t*w) attention).
	Window int
	NLayer, NHead, NKvHead, HeadDim, Hidden, Ffn int
	Eps                                        float32
	Theta                                      float64
	TokEmb                                     []float32 // [vocab*hidden]
	ET                                         []float32 // [hidden*vocab] transposed for the AVX2 head backward
	OutNorm                                    []float32 // [hidden]
	Vocab                                      int
	Layers                                     []*modelLayer
}

type modelLayer struct {
	AttnNorm, FfnNorm []float32
	QNorm, KNorm      []float32 // per-head RMSNorm gains [headDim]
	Q, K, V, O        *loraLayer
	Gate, Up, Down    *loraLayer
	// per-sample lora hidden vectors (captured in forward for backward)
	hQ, hK, hV, hO []float32
	hGate, hUp     []float32
	hDown          []float32
	// per-sample residual snapshots: xIn = layer input (x0),
	// oOut = attention output projection result (x1 = xIn + oOut)
	xIn  []float32
	oOut []float32
}

// scratch: per-sample activation buffers, ALL preallocated once per run and
// reused across samples and layers. Training must not allocate on the hot
// path: the per-layer buffers were previously `make`d ~30 times per layer
// per sample (~100MB of garbage per sample, plus a fresh T*vocab logits
// array every forward). With reuse, heap stays flat and GC stays quiet.
type scratch struct {
	T int
	X        []float32 // [T*H] activations (residual accumulator)
	Hn       []float32 // [T*H] normed input
	Q, K, V  []float32 // post-proj
	Qn, Kn   []float32 // post norm+rope
	AttnOut  []float32 // [T*2048] per-head outs
	Scores   []float32 // [T*nHead*T]
	Gate, Up []float32 // [T*3072] post-silu gate / up
	GatePre  []float32 // [T*3072] pre-silu gate (for the silu' backward)
	DownIn   []float32 // [T*3072]
	Logits   []float32 // [T*vocab] (preallocated; rows fully written per forward)
	G        []float32 // [T*vocab] dense (softmax-onehot) grads for the head backward
	// per-layer backward buffers (max-sized, reused across layers)
	Out       []float32 // [T*H]  O-proj output
	DAttnOut  []float32 // [T*2048]
	DX1       []float32 // [T*H]
	DDownIn   []float32 // [T*Ffn]
	DGatePre  []float32 // [T*Ffn]
	DUp       []float32 // [T*Ffn]
	DHn       []float32 // [T*H]
	X1        []float32 // [T*H] xIn+oOut
	DQn       []float32 // [T*2048]
	DKn       []float32 // [T*1024]
	DV        []float32 // [T*1024]
	DQ        []float32 // [T*2048]
	DK        []float32 // [T*1024]
	DHn2      []float32 // [T*H]
	Ds        []float32 // [T*nHead*T] per-(pos,head) softmax grad rows
	DVp       []float32 // [threads*T*nkv*headDim] per-thread DV partials
	DKnP      []float32 // [threads*T*nkv*headDim] per-thread DKn partials
	Threads   int
}

func newScratch(t int, m *model, threads int) *scratch {
	return &scratch{T: t, Threads: threads,
		X: make([]float32, t*m.Hidden), Hn: make([]float32, t*m.Hidden),
		Q: make([]float32, t*m.NHead*m.HeadDim), K: make([]float32, t*m.NKvHead*m.HeadDim),
		V: make([]float32, t*m.NKvHead*m.HeadDim),
		Qn: make([]float32, t*m.NHead*m.HeadDim), Kn: make([]float32, t*m.NKvHead*m.HeadDim),
		AttnOut: make([]float32, t*m.NHead*m.HeadDim),
		Scores:  make([]float32, t*m.NHead*t),
		Gate:    make([]float32, t*m.Ffn), Up: make([]float32, t*m.Ffn),
		GatePre: make([]float32, t*m.Ffn),
		DownIn:  make([]float32, t*m.Ffn),
		Logits:  make([]float32, t*m.Vocab),
		G:       make([]float32, t*m.Vocab),
		Out:      make([]float32, t*m.Hidden),
		DAttnOut: make([]float32, t*m.NHead*m.HeadDim),
		DX1:      make([]float32, t*m.Hidden),
		DDownIn:  make([]float32, t*m.Ffn),
		DGatePre: make([]float32, t*m.Ffn),
		DUp:      make([]float32, t*m.Ffn),
		DHn:      make([]float32, t*m.Hidden),
		X1:       make([]float32, t*m.Hidden),
		DQn:      make([]float32, t*m.NHead*m.HeadDim),
		DKn:      make([]float32, t*m.NKvHead*m.HeadDim),
		DV:       make([]float32, t*m.NKvHead*m.HeadDim),
		DQ:       make([]float32, t*m.NHead*m.HeadDim),
		DK:       make([]float32, t*m.NKvHead*m.HeadDim),
		DHn2:     make([]float32, t*m.Hidden),
		Ds:       make([]float32, t*m.NHead*t),
		DVp:      make([]float32, threads*t*m.NKvHead*m.HeadDim),
		DKnP:     make([]float32, threads*t*m.NKvHead*m.HeadDim),
	}
}

// zeroF32: memset-style zero (the hot buffers are reused; anything that
// ACCUMULATES across a layer must be cleared before the layer's backward).
func zeroF32(v []float32) {
	for i := range v {
		v[i] = 0
	}
}

// forward: logits into s.Logits (allocates T*vocab), returns the mean
// cross-entropy loss over T positions.
func (m *model) forward(s *scratch, tokens []int) float32 {
	var t int = s.T
	ropeEnsure(t, m.HeadDim, m.Theta)
	var i int = 0
	for i = 0; i < t; i++ {
		copy(s.X[i*m.Hidden:], m.TokEmb[tokens[i]*m.Hidden:(tokens[i]+1)*m.Hidden])
	}
	var l int = 0
	for l = 0; l < m.NLayer; l++ {
		layer := m.Layers[l]
		layer.xIn = append(layer.xIn[:0], s.X...)
		// attention
		rmsNorm(s.Hn, s.X, layer.AttnNorm, m.Hidden, m.Eps)
		layer.hQ = layer.Q.fwd(s.Q, s.Hn, t)
		layer.hK = layer.K.fwd(s.K, s.Hn, t)
		layer.hV = layer.V.fwd(s.V, s.Hn, t)
		parallel(t, func(lo int, hi int) {
			var hp int = 0
			for hp = 0; hp < m.NHead; hp++ {
				var pos int = lo
				for pos = lo; pos < hi; pos++ {
					var base int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var kbase int = pos*m.NKvHead*m.HeadDim + (hp % m.NKvHead) * m.HeadDim
					rmsNorm(s.Qn[base:base+m.HeadDim], s.Q[base:base+m.HeadDim], layer.QNorm, m.HeadDim, m.Eps)
					ropeApply(s.Qn[base:base+m.HeadDim], pos, m.HeadDim, m.Theta)
					rmsNorm(s.Kn[kbase:kbase+m.HeadDim], s.K[kbase:kbase+m.HeadDim], layer.KNorm, m.HeadDim, m.Eps)
					ropeApply(s.Kn[kbase:kbase+m.HeadDim], pos, m.HeadDim, m.Theta)
				}
			}
		})
		parallel(t, func(lo int, hi int) {
			var pos int = lo
			for pos = lo; pos < hi; pos++ {
				var hp int = 0
				for hp = 0; hp < m.NHead; hp++ {
					var qbase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var kg int = hp % m.NKvHead
					var srow int = (pos*m.NHead + hp) * t
					// sliding window: attend to the last W keys incl. self.
					// O(t*w*headDim) instead of O(t*t*headDim); window 0 = full.
					var jlo int = 0
					if m.Window > 0 {
						jlo = pos - m.Window + 1
						if jlo < 0 {
							jlo = 0
						}
					}
					var maxv float32 = -1e30
					var j int = 0
					for j = 0; j < jlo; j++ {
						s.Scores[srow+j] = -1e30
					}
					for j = jlo; j <= pos; j++ {
						var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
						var acc float32 = dot4(s.Qn[qbase:qbase+m.HeadDim], s.Kn[kbase:kbase+m.HeadDim], m.HeadDim)
						s.Scores[srow+j] = acc
						if acc > maxv {
							maxv = acc
						}
					}
					for j = pos + 1; j < t; j++ {
						s.Scores[srow+j] = -1e30
					}
					var sum float32 = 0
					for j = jlo; j <= pos; j++ {
						s.Scores[srow+j] = float32(math.Exp(float64(s.Scores[srow+j] - maxv)))
						sum += s.Scores[srow+j]
					}
					for j = jlo; j <= pos; j++ {
						s.Scores[srow+j] /= sum
					}
					var obase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var d int = 0
					// j-outer/d-inner: V rows are contiguous per j (the old
					// d-outer order touched a different cache line per element).
					for d = 0; d < m.HeadDim; d++ {
						s.AttnOut[obase+d] = 0
					}
					for j = jlo; j <= pos; j++ {
						var sc float32 = s.Scores[srow+j]
						var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
						axpy4(s.AttnOut[obase:obase+m.HeadDim], s.V[kbase:kbase+m.HeadDim], m.HeadDim, sc)
					}
				}
			}
		})
		if m.Col != nil {
			m.Col.accumulateLayer(s, m, l, t)
		}
		// output projection + residual
		zeroF32(s.Out)
		layer.hO = layer.O.fwd(s.Out, s.AttnOut, t)
		layer.oOut = append(layer.oOut[:0], s.Out...)
		for i = 0; i < t*m.Hidden; i++ {
			s.X[i] += s.Out[i]
		}
		// MLP
		rmsNorm(s.Hn, s.X, layer.FfnNorm, m.Hidden, m.Eps)
		layer.hGate = layer.Gate.fwd(s.Gate, s.Hn, t)
		layer.hUp = layer.Up.fwd(s.Up, s.Hn, t)
		parallel(t, func(lo int, hi int) {
			var row int = lo
			for row = lo; row < hi; row++ {
				var base int = row * m.Ffn
				var g int = 0
				for g = 0; g < m.Ffn; g++ {
					s.GatePre[base+g] = s.Gate[base+g] // keep pre-silu for backward
					s.Gate[base+g] = s.Gate[base+g] / (1 + float32(math.Exp(float64(-s.Gate[base+g])))) // silu
					s.DownIn[base+g] = s.Gate[base+g] * s.Up[base+g]
				}
			}
		})
		layer.hDown = layer.Down.fwd(s.X, s.DownIn, t) // adds into s.X (residual)
	}
	// final norm
	rmsNorm(s.Hn, s.X, m.OutNorm, m.Hidden, m.Eps)
	// head (tied) + cross entropy: every logits row is computed ONCE via the
	// AVX2 kernel (Logits is zeroed first; the kernel accumulates). The
	// sequential pass only softmaxes + accumulates the loss. (The old code
	// computed every row twice - ~40G wasted MACs per step at vocab 151936.)
	var total float32 = 0
	var vocab int = m.Vocab
	zeroF32(s.Logits)
	gemmFwdSimd(s.Logits, s.Hn, m.TokEmb, t, m.Hidden, m.Vocab)
	for i = 0; i < t; i++ {
		var row []float32 = s.Logits[i*vocab : (i+1)*vocab]
		var maxv float32 = row[0]
		var v int = 1
		for v = 1; v < vocab; v++ {
			if row[v] > maxv {
				maxv = row[v]
			}
		}
		var sum float32 = 0
		for v = 0; v < vocab; v++ {
			sum += float32(math.Exp(float64(row[v] - maxv)))
		}
		var lse float32 = maxv + float32(math.Log(float64(sum)))
		total += lse - row[tokens[i]]
	}
	return total / float32(t)
}

// backward: full graph. The embedding table is frozen (not a LoRA target),
// so only dX (the gradient flowing INTO the embeddings) is produced.
// Assumes forward() ran on the same tokens and every loraLayer's dA/dB are
// zeroed.
func (m *model) backward(s *scratch, tokens []int) {
	var t int = s.T
	var vocab int = m.Vocab
	ropeEnsure(t, m.HeadDim, m.Theta)
	// ---- head backward: dXHead = (softmax - onehot) @ E. The dense g
	// matrix goes through the AVX2 kernel instead of the old per-vocab
	// axpy loop, which streamed the whole 622MB embedding table once PER
	// TOKEN (160GB per sample - the #1 profile hotspot).
	var dXHead []float32 = make([]float32, t*m.Hidden)
	var i int = 0
	parallel(t, func(lo int, hi int) {
		var i int = lo
		for i = lo; i < hi; i++ {
			var row []float32 = s.Logits[i*vocab : (i+1)*vocab]
			var maxv float32 = row[0]
			var v int = 0
			for v = 1; v < vocab; v++ {
				if row[v] > maxv {
					maxv = row[v]
				}
			}
			var sum float32 = 0
			for v = 0; v < vocab; v++ {
				sum += float32(math.Exp(float64(row[v] - maxv)))
			}
			var inv float32 = 1 / sum
			var target int = tokens[i]
			var gr []float32 = s.G[i*vocab : (i+1)*vocab]
			for v = 0; v < vocab; v++ {
				var g float32 = float32(math.Exp(float64(row[v]-maxv))) * inv
				if v == target {
					g -= 1
				}
				gr[v] = g
			}
		}
	})
	gemmBwdSimd(dXHead, s.G, m.ET, t, m.Hidden, m.Vocab)
	// note: the embedding table is frozen (not a LoRA target); the head's
	// outer-product dEmb is intentionally not computed.
	// final norm backward: s.X (still the post-mlp activations) is the input
	rmsNormBack(s.X, dXHead, s.X, m.OutNorm, m.Hidden, m.Eps)
	// ---- layers in reverse
	var l int = 0
	for l = m.NLayer - 1; l >= 0; l-- {
		layer := m.Layers[l]
		// ---- MLP backward: s.X holds d(x2)
		copy(s.DX1, s.X) // residual: d(x1) += d(x2)
		zeroF32(s.DDownIn) // Down.back accumulates into its dx
		layer.Down.back(s.DDownIn, s.X, s.DownIn, layer.hDown, t)
		parallel(t, func(lo int, hi int) {
			var row int = lo
			for row = lo; row < hi; row++ {
				var base int = row * m.Ffn
				var g int = 0
				for g = 0; g < m.Ffn; g++ {
					// silu(z) = z*sig(z); silu'(z) = sig(z)*(1 + z*(1-sig(z)))
					var z float32 = s.GatePre[base+g]
					var sig float32 = 1 / (1 + float32(math.Exp(float64(-z))))
					var dsilu float32 = sig * (1 + z*(1-sig))
					s.DGatePre[base+g] = s.DDownIn[base+g] * dsilu * s.Up[base+g]
					s.DUp[base+g] = s.DDownIn[base+g] * s.Gate[base+g]
				}
			}
		})
		zeroF32(s.DHn)
		layer.Gate.back(s.DHn, s.DGatePre, s.Hn, layer.hGate, t)
		layer.Up.back(s.DHn, s.DUp, s.Hn, layer.hUp, t)
		// ffn norm backward; input was x1 = xIn + oOut
		for i = 0; i < t*m.Hidden; i++ {
			s.X1[i] = layer.xIn[i] + layer.oOut[i]
		}
		rmsNormBack(s.DX1, s.DHn, s.X1, layer.FfnNorm, m.Hidden, m.Eps)
		copy(s.X, s.DX1) // s.X = d(x1)
		// ---- attention backward (parallel over query positions; DV/DKn are
		// shared accumulators, so each goroutine keeps per-thread partials
		// and a deterministic sequential reduction follows).
		zeroF32(s.DAttnOut) // O.back accumulates into its dx
		layer.O.back(s.DAttnOut, s.X, s.AttnOut, layer.hO, t) // dy = d(x1), dx = d(attn outs)
		zeroF32(s.DQn)
		var pv int = t * m.NKvHead * m.HeadDim
		var chunk int = (t + trainThreads - 1) / trainThreads
		zeroF32(s.DVp)
		zeroF32(s.DKnP)
		parallel(t, func(lo int, hi int) {
			var tid int = lo / chunk
			var pos int = lo
			for pos = lo; pos < hi; pos++ {
				var hp int = 0
				for hp = 0; hp < m.NHead; hp++ {
					var obase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var kg int = hp % m.NKvHead
					var srow int = (pos*m.NHead + hp) * t
					// sliding window (mirror of forward): gradients only flow to
					// the attended keys/values within the window.
					var jlo int = 0
					if m.Window > 0 {
						jlo = pos - m.Window + 1
						if jlo < 0 {
							jlo = 0
						}
					}
					var dsRow []float32 = s.Ds[srow : srow+t]
					var z int = 0
					for z = jlo; z <= pos; z++ {
						dsRow[z] = 0
					}
					// ds[j] = dAttnOut . V[j] ; DVp[j][d] += scores[j]*dAttnOut[d]
					// (j-outer/d-inner: V and DVp rows are contiguous per j; the
					// old d-outer order touched a fresh cache line per element).
					var j int = 0
					for j = jlo; j <= pos; j++ {
						var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
						dsRow[j] = dot4(s.DAttnOut[obase:obase+m.HeadDim], s.V[kbase:kbase+m.HeadDim], m.HeadDim)
						var sc float32 = s.Scores[srow+j]
						axpy4(s.DVp[tid*pv+kbase:tid*pv+kbase+m.HeadDim], s.DAttnOut[obase:obase+m.HeadDim], m.HeadDim, sc)
					}
					// softmax backward
					var dot float32 = 0
					for j = jlo; j <= pos; j++ {
						dot += s.Scores[srow+j] * dsRow[j]
					}
					for j = jlo; j <= pos; j++ {
						dsRow[j] = s.Scores[srow+j] * (dsRow[j] - dot)
					}
					// dQn[pos,hp] += ds[j]*Kn[j,kg] ; dKn[j,kg] += ds[j]*Qn[pos,hp]
					// (j-outer/d-inner: Kn/Qn rows contiguous per j)
					var qbase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					for j = jlo; j <= pos; j++ {
						var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
						var dsj float32 = dsRow[j]
						axpy4(s.DQn[qbase:qbase+m.HeadDim], s.Kn[kbase:kbase+m.HeadDim], m.HeadDim, dsj)
						axpy4(s.DKnP[tid*pv+kbase:tid*pv+kbase+m.HeadDim], s.Qn[qbase:qbase+m.HeadDim], m.HeadDim, dsj)
					}
				}
			}
		})
		// deterministic reduction of the per-thread partials (fixed order)
		zeroF32(s.DV)
		zeroF32(s.DKn)
		var th int = 0
		for th = 0; th < s.Threads; th++ {
			var off int = th * pv
			for i = 0; i < pv; i++ {
				s.DV[i] += s.DVp[off+i]
				s.DKn[i] += s.DKnP[off+i]
			}
		}
		// per-head norm + rope backward (parallel over positions; each
		// (pos,head) row is private)
		zeroF32(s.DQ)
		zeroF32(s.DK)
		parallel(t, func(lo int, hi int) {
			var hp int = 0
			for hp = 0; hp < m.NHead; hp++ {
				var pos int = lo
				for pos = lo; pos < hi; pos++ {
					var base int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var kbase int = pos*m.NKvHead*m.HeadDim + (hp % m.NKvHead) * m.HeadDim
					ropeBack(s.DQn[base:base+m.HeadDim], pos, m.HeadDim, m.Theta)
					rmsNormBack(s.DQ[base:base+m.HeadDim], s.DQn[base:base+m.HeadDim],
						s.Q[base:base+m.HeadDim], layer.QNorm, m.HeadDim, m.Eps)
					ropeBack(s.DKn[kbase:kbase+m.HeadDim], pos, m.HeadDim, m.Theta)
					rmsNormBack(s.DK[kbase:kbase+m.HeadDim], s.DKn[kbase:kbase+m.HeadDim],
						s.K[kbase:kbase+m.HeadDim], layer.KNorm, m.HeadDim, m.Eps)
				}
			}
		})
		zeroF32(s.DHn2)
		layer.Q.back(s.DHn2, s.DQ, s.Hn, layer.hQ, t)
		layer.K.back(s.DHn2, s.DK, s.Hn, layer.hK, t)
		layer.V.back(s.DHn2, s.DV, s.Hn, layer.hV, t)
		// attn norm backward; input was xIn
		rmsNormBack(s.X, s.DHn2, layer.xIn, layer.AttnNorm, m.Hidden, m.Eps)
		// s.X is now d(xIn) -> continue to the previous layer
	}
	// s.X holds d(embedding output); the embedding is frozen, nothing to do.
}

// loadModel: read the base GGUF into a trainable model (dequant to fp32).
func loadModel(path string) (*model, error) {
	g, err := loadGGUF(path)
	if err != nil {
		return nil, err
	}
	m := &model{
		NLayer: g.kvInt("qwen3.block_count"), NHead: g.kvInt("qwen3.attention.head_count"),
		NKvHead: g.kvInt("qwen3.attention.head_count_kv"), HeadDim: g.kvInt("qwen3.attention.key_length"),
		Hidden: g.kvInt("qwen3.embedding_length"), Ffn: g.kvInt("qwen3.feed_forward_length"),
		Eps: float32(g.kvFloat("qwen3.attention.layer_norm_rms_epsilon")),
		Theta: g.kvFloat("qwen3.rope.freq_base"),
	}
	if m.Theta == 0 {
		m.Theta = 1000000
	}
	tokEmb := g.findTensor("token_embd.weight")
	if tokEmb == nil {
		return nil, errNoTensor("token_embd.weight")
	}
	m.TokEmb = g.tensorF32(tokEmb)
	m.Vocab = int(tokEmb.Dims[1])
	// ET[i*vocab+j] = TokEmb[j*hidden+i] - the head backward streams it
	// contiguously through the AVX2 kernel (the old axpy loop read the
	// whole 622MB table once PER TOKEN - 160GB per sample).
	m.ET = make([]float32, m.Hidden*m.Vocab)
	var ehidx int = 0
	for ehidx = 0; ehidx < m.Hidden; ehidx++ {
		var ev int = 0
		for ev = 0; ev < m.Vocab; ev++ {
			m.ET[ehidx*m.Vocab+ev] = m.TokEmb[ev*m.Hidden+ehidx]
		}
	}
	outNorm := g.findTensor("output_norm.weight")
	if outNorm == nil {
		return nil, errNoTensor("output_norm.weight")
	}
	m.OutNorm = g.tensorF32(outNorm)
	m.Layers = make([]*modelLayer, 0, m.NLayer)
	var l int = 0
	for l = 0; l < m.NLayer; l++ {
		layer := &modelLayer{}
		layer.AttnNorm = mustTensor(g, fmt.Sprintf("blk.%d.attn_norm.weight", l))
		layer.FfnNorm = mustTensor(g, fmt.Sprintf("blk.%d.ffn_norm.weight", l))
		layer.QNorm = mustTensor(g, fmt.Sprintf("blk.%d.attn_q_norm.weight", l))
		layer.KNorm = mustTensor(g, fmt.Sprintf("blk.%d.attn_k_norm.weight", l))
		var r int = 16
		var alpha float32 = 32
		layer.Q = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.attn_q.weight", l)), m.Hidden, m.NHead*m.HeadDim, r, alpha)
		layer.K = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.attn_k.weight", l)), m.Hidden, m.NKvHead*m.HeadDim, r, alpha)
		layer.V = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.attn_v.weight", l)), m.Hidden, m.NKvHead*m.HeadDim, r, alpha)
		layer.O = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.attn_output.weight", l)), m.NHead*m.HeadDim, m.Hidden, r, alpha)
		layer.Gate = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.ffn_gate.weight", l)), m.Hidden, m.Ffn, r, alpha)
		layer.Up = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.ffn_up.weight", l)), m.Hidden, m.Ffn, r, alpha)
		layer.Down = newLoraLayer(mustTensor(g, fmt.Sprintf("blk.%d.ffn_down.weight", l)), m.Ffn, m.Hidden, r, alpha)
		m.Layers = append(m.Layers, layer)
	}
	return m, nil
}

type errNoTensor string

func (e errNoTensor) Error() string { return "tensor not found: " + string(e) }

func mustTensor(g *ggufFile, name string) []float32 {
	info := g.findTensor(name)
	if info == nil {
		panic(errNoTensor(name))
	}
	w := g.tensorF32(info)
	if w == nil {
		panic("unsupported/ragged tensor type for " + name + " (type " +
			fmt.Sprint(info.Type) + " dims " + fmt.Sprint(info.Dims) + ")")
	}
	return w
}
