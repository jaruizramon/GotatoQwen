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
	// A ~ U(-sqrt(1/r), sqrt(1/r)); B = 0 (standard LoRA init)
	var bound float32 = float32(math.Sqrt(float64(1.0 / r)))
	var i int = 0
	for i = 0; i < len(l.A); i++ {
		l.A[i] = (randF32()*2 - 1) * bound
	}
	return l
}

// fwd: y[m*out+j] += sum_k x[m*in+k]*W[j*in+k] + (alpha/r) * sum_r' h[m*r+r']*B[j*r+r']
// where h[m*r+r'] = sum_k x[m*in+k]*A[r'*in+k]. Returns h (needed by back).
func (l *loraLayer) fwd(y []float32, x []float32, m int) []float32 {
	var h []float32 = make([]float32, m*l.R)
	parallel(m, func(lo int, hi int) {
		var row int = lo
		for row = lo; row < hi; row++ {
			var xr int = row * l.In
			var j int = 0
			for j = 0; j < l.Out; j++ {
				var acc float32 = 0
				var wj int = j * l.In
				var k int = 0
				for k = 0; k < l.In; k++ {
					acc += x[xr+k] * l.W[wj+k]
				}
				y[row*l.Out+j] += acc
			}
			var rp int = 0
			for rp = 0; rp < l.R; rp++ {
				var acc float32 = 0
				var ar int = rp * l.In
				var k int = 0
				for k = 0; k < l.In; k++ {
					acc += x[xr+k] * l.A[ar+k]
				}
				h[row*l.R+rp] = acc
			}
			var scale float32 = l.Alpha / float32(l.R)
			for j = 0; j < l.Out; j++ {
				var acc float32 = 0
				var bj int = j * l.R
				var rp int = 0
				for rp = 0; rp < l.R; rp++ {
					acc += h[row*l.R+rp] * l.B[bj+rp]
				}
				y[row*l.Out+j] += scale * acc
			}
		}
	})
	return h
}

// back: gradients of the LoRA path + the frozen base path. dA/dB accumulate
// into the layer's own buffers (zeroed via zeroGrads before use).
func (l *loraLayer) back(dx []float32, dy []float32, x []float32, h []float32, m int) {
	var dyh []float32 = make([]float32, m*l.R)
	var scale float32 = l.Alpha / float32(l.R)
	parallel(m, func(lo int, hi int) {
		var row int = lo
		for row = lo; row < hi; row++ {
			var xr int = row * l.In
			var yr int = row * l.Out
			var rp int = 0
			for rp = 0; rp < l.R; rp++ {
				var acc float32 = 0
				var br int = rp
				var j int = 0
				for j = 0; j < l.Out; j++ {
					acc += dy[yr+j] * l.B[j*l.R+br]
				}
				dyh[row*l.R+rp] = scale * acc
			}
			var j int = 0
			for j = 0; j < l.Out; j++ {
				var dyv float32 = dy[yr+j]
				var wj int = j * l.In
				var k int = 0
				for k = 0; k < l.In; k++ {
					dx[xr+k] += dyv * l.W[wj+k]
				}
			}
			for rp = 0; rp < l.R; rp++ {
				var dyhv float32 = dyh[row*l.R+rp]
				var ar int = rp * l.In
				var k int = 0
				for k = 0; k < l.In; k++ {
					dx[xr+k] += dyhv * l.A[ar+k]
				}
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
	var lr float32 = 2e-4
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

func ropeApply(q []float32, t int, headDim int, theta float64) {
	var half int = headDim / 2
	var i int = 0
	for i = 0; i < half; i++ {
		var freq float64 = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
		var angle float64 = float64(t) * freq
		var c float64 = math.Cos(angle)
		var s float64 = math.Sin(angle)
		var a float32 = q[2*i]
		var b float32 = q[2*i+1]
		q[2*i] = float32(float64(a)*c - float64(b)*s)
		q[2*i+1] = float32(float64(a)*s + float64(b)*c)
	}
}

func ropeBack(dq []float32, t int, headDim int, theta float64) {
	var half int = headDim / 2
	var i int = 0
	for i = 0; i < half; i++ {
		var freq float64 = 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
		var angle float64 = float64(t) * freq
		var c float64 = math.Cos(angle)
		var s float64 = math.Sin(angle)
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
	NLayer, NHead, NKvHead, HeadDim, Hidden, Ffn int
	Eps                                        float32
	Theta                                      float64
	TokEmb                                     []float32 // [vocab*hidden]
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
	Ds        []float32 // [T] per-(pos,head) softmax grad
}

func newScratch(t int, m *model) *scratch {
	return &scratch{T: t,
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
		Ds:       make([]float32, t),
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
					var maxv float32 = -1e30
					var j int = 0
					for j = 0; j <= pos; j++ {
						var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
						var acc float32 = 0
						var d int = 0
						for d = 0; d < m.HeadDim; d++ {
							acc += s.Qn[qbase+d] * s.Kn[kbase+d]
						}
						s.Scores[srow+j] = acc
						if acc > maxv {
							maxv = acc
						}
					}
					for j = pos + 1; j < t; j++ {
						s.Scores[srow+j] = -1e30
					}
					var sum float32 = 0
					for j = 0; j <= pos; j++ {
						s.Scores[srow+j] = float32(math.Exp(float64(s.Scores[srow+j] - maxv)))
						sum += s.Scores[srow+j]
					}
					for j = 0; j <= pos; j++ {
						s.Scores[srow+j] /= sum
					}
					var obase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
					var d int = 0
					for d = 0; d < m.HeadDim; d++ {
						var acc float32 = 0
						var j int = 0
						for j = 0; j <= pos; j++ {
							acc += s.Scores[srow+j] * s.V[j*m.NKvHead*m.HeadDim+kg*m.HeadDim+d]
						}
						s.AttnOut[obase+d] = acc
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
		var g int = 0
		for g = 0; g < t*m.Ffn; g++ {
			s.GatePre[g] = s.Gate[g] // keep the pre-silu value for backward
			s.Gate[g] = s.Gate[g] / (1 + float32(math.Exp(float64(-s.Gate[g])))) // silu
			s.DownIn[g] = s.Gate[g] * s.Up[g]
		}
		layer.hDown = layer.Down.fwd(s.X, s.DownIn, t) // adds into s.X (residual)
	}
	// final norm
	rmsNorm(s.Hn, s.X, m.OutNorm, m.Hidden, m.Eps)
	// head (tied) + cross entropy (s.Logits preallocated; rows fully written)
	var total float32 = 0
	var vocab int = m.Vocab
	parallel(t, func(lo int, hi int) {
		var i int = lo
		for i = lo; i < hi; i++ {
			var row []float32 = s.Logits[i*vocab : (i+1)*vocab]
			var xr []float32 = s.Hn[i*m.Hidden : (i+1)*m.Hidden]
			var v int = 0
			for v = 0; v < vocab; v++ {
				var acc float32 = 0
				var er int = v * m.Hidden
				var k int = 0
				for k = 0; k < m.Hidden; k++ {
					acc += xr[k] * m.TokEmb[er+k]
				}
				row[v] = acc
			}
		}
	})
	for i = 0; i < t; i++ {
		var row []float32 = s.Logits[i*vocab : (i+1)*vocab]
		var xr []float32 = s.Hn[i*m.Hidden : (i+1)*m.Hidden]
		var v int = 0
		for v = 0; v < vocab; v++ {
			var acc float32 = 0
			var er int = v * m.Hidden
			var k int = 0
			for k = 0; k < m.Hidden; k++ {
				acc += xr[k] * m.TokEmb[er+k]
			}
			row[v] = acc
		}
		var maxv float32 = row[0]
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
	// ---- head backward: dXHead = (softmax - onehot) @ E ; dEmb += outer
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
			var dr []float32 = dXHead[i*m.Hidden : (i+1)*m.Hidden]
			for v = 0; v < vocab; v++ {
				var p float32 = float32(math.Exp(float64(row[v]-maxv))) * inv
				var g float32 = p
				if v == target {
					g -= 1
				}
				if g == 0 {
					continue
				}
				var er int = v * m.Hidden
				var k int = 0
				for k = 0; k < m.Hidden; k++ {
					dr[k] += g * m.TokEmb[er+k]
				}
			}
		}
	})
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
		var g int = 0
		for g = 0; g < t*m.Ffn; g++ {
			// silu(z) = z*sig(z); silu'(z) = sig(z)*(1 + z*(1-sig(z)))
			var z float32 = s.GatePre[g]
			var sig float32 = 1 / (1 + float32(math.Exp(float64(-z))))
			var dsilu float32 = sig * (1 + z*(1-sig))
			s.DGatePre[g] = s.DDownIn[g] * dsilu * s.Up[g]
			s.DUp[g] = s.DDownIn[g] * s.Gate[g]
		}
		zeroF32(s.DHn)
		layer.Gate.back(s.DHn, s.DGatePre, s.Hn, layer.hGate, t)
		layer.Up.back(s.DHn, s.DUp, s.Hn, layer.hUp, t)
		// ffn norm backward; input was x1 = xIn + oOut
		for i = 0; i < t*m.Hidden; i++ {
			s.X1[i] = layer.xIn[i] + layer.oOut[i]
		}
		rmsNormBack(s.DX1, s.DHn, s.X1, layer.FfnNorm, m.Hidden, m.Eps)
		copy(s.X, s.DX1) // s.X = d(x1)
		// ---- attention backward
		zeroF32(s.DAttnOut) // O.back accumulates into its dx
		layer.O.back(s.DAttnOut, s.X, s.AttnOut, layer.hO, t) // dy = d(x1), dx = d(attn outs)
		zeroF32(s.DQn)
		zeroF32(s.DKn)
		zeroF32(s.DV)
		var pos int = 0
		for pos = 0; pos < t; pos++ {
			var hp int = 0
			for hp = 0; hp < m.NHead; hp++ {
				var obase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
				var kg int = hp % m.NKvHead
				var srow int = (pos*m.NHead + hp) * t
				// ds[j] = dAttnOut . V[j] ; dv[j] += scores[j]*dAttnOut
				for i = 0; i < t; i++ {
					s.Ds[i] = 0
				}
				var d int = 0
				for d = 0; d < m.HeadDim; d++ {
					var dyv float32 = s.DAttnOut[obase+d]
					var j int = 0
					for j = 0; j <= pos; j++ {
						s.Ds[j] += dyv * s.V[j*m.NKvHead*m.HeadDim+kg*m.HeadDim+d]
						s.DV[j*m.NKvHead*m.HeadDim+kg*m.HeadDim+d] += s.Scores[srow+j] * dyv
					}
				}
				// softmax backward
				var dot float32 = 0
				var j int = 0
				for j = 0; j <= pos; j++ {
					dot += s.Scores[srow+j] * s.Ds[j]
				}
				for j = 0; j <= pos; j++ {
					s.Ds[j] = s.Scores[srow+j] * (s.Ds[j] - dot)
				}
				// dQn[pos,hp] += ds[j]*Kn[j,kg] ; dKn[j,kg] += ds[j]*Qn[pos,hp]
				var qbase int = pos*m.NHead*m.HeadDim + hp*m.HeadDim
				for j = 0; j <= pos; j++ {
					var kbase int = j*m.NKvHead*m.HeadDim + kg*m.HeadDim
					var dsj float32 = s.Ds[j]
					var d int = 0
					for d = 0; d < m.HeadDim; d++ {
						s.DQn[qbase+d] += dsj * s.Kn[kbase+d]
						s.DKn[kbase+d] += dsj * s.Qn[qbase+d]
					}
				}
			}
		}
		// per-head norm + rope backward
		zeroF32(s.DQ)
		zeroF32(s.DK)
		var hp int = 0
		for hp = 0; hp < m.NHead; hp++ {
			for pos = 0; pos < t; pos++ {
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
