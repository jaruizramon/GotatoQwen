// lora.go - one LoRA linear layer: forward, backward, AdamW. All explicit.
//
//   y = x W^T + (alpha/r) * (x A^T) B^T
//   dA = (dy B) x        (chain: d(xA^T) = dy B, then x A^T outer)
//   dB = dy^T (x A^T)
// Weights are row-major [out][in], matching GGUF matmul layout.
package main

import "math"

const (
	loraRank  int     = 16
	loraAlpha float32 = 32.0
	adamBeta1 float32 = 0.9
	adamBeta2 float32 = 0.999
	adamEps   float32 = 1e-8
	adamLR    float32 = 2e-4
)

// mulAdd: y[m][n] += x[m][k] * w[n][k]  (w is [n][k], row-major)
func mulAdd(y []float32, x []float32, w []float32, m int, n int, k int) {
	var row int = 0
	for row = 0; row < m; row++ {
		var col int = 0
		for col = 0; col < n; col++ {
			var acc float32 = 0
			var xi int = row * k
			var wj int = col * k
			var t int = 0
			for t = 0; t < k; t++ {
				acc += x[xi+t] * w[wj+t]
			}
			y[row*n+col] += acc
		}
	}
}

// gemmXAT: out[m][r] = x[m][k] * a[r][k]^T  ->  out = x A^T
func gemmXAT(out []float32, x []float32, a []float32, m int, k int, r int) {
	var row int = 0
	for row = 0; row < m; row++ {
		var j int = 0
		for j = 0; j < r; j++ {
			var acc float32 = 0
			var xi int = row * k
			var aj int = j * k
			var t int = 0
			for t = 0; t < k; t++ {
				acc += x[xi+t] * a[aj+t]
			}
			out[row*r+j] = acc
		}
	}
}

// gemmXBT: out[m][n] = xh[m][r] * b[n][r]^T  ->  out = xh B^T
func gemmXBT(out []float32, xh []float32, b []float32, m int, r int, n int) {
	var row int = 0
	for row = 0; row < m; row++ {
		var j int = 0
		for j = 0; j < n; j++ {
			var acc float32 = 0
			var xi int = row * r
			var bj int = j * r
			var t int = 0
			for t = 0; t < r; t++ {
				acc += xh[xi+t] * b[bj+t]
			}
			out[row*n+j] = acc
		}
	}
}

type loraLinear struct {
	W    []float32 // [out][in], frozen
	A    []float32 // [r][in]
	B    []float32 // [out][r]
	mA   []float32 // adam first moment
	vA   []float32
	mB   []float32
	vB   []float32
	in   int
	out  int
	r    int
	step int
}

func newLoraLinear(w []float32, in int, out int) *loraLinear {
	var r int = loraRank
	var scale float32 = float32(math.Sqrt(5.0 / float64(in)))
	var l *loraLinear = &loraLinear{
		W: w, in: in, out: out, r: r,
		A: make([]float32, r*in), B: make([]float32, out*r),
		mA: make([]float32, r*in), vA: make([]float32, r*in),
		mB: make([]float32, out*r), vB: make([]float32, out*r),
	}
	var idx int = 0
	for idx = 0; idx < r*in; idx++ {
		l.A[idx] = (randF32()*2 - 1) * scale
	}
	return l
}

// forward: y = xW^T + scale * (x A^T) B^T ; returns xh = x A^T (needed for bwd)
func (l *loraLinear) forward(y []float32, x []float32, m int) []float32 {
	var scale float32 = loraAlpha / float32(l.r)
	var xh []float32 = make([]float32, m*l.r)
	gemmXAT(xh, x, l.A, m, l.in, l.r)
	var idx int = 0
	for idx = 0; idx < m*l.out; idx++ {
		y[idx] = 0
	}
	mulAdd(y, x, l.W, m, l.out, l.in)
	var tmp []float32 = make([]float32, m*l.out)
	gemmXBT(tmp, xh, l.B, m, l.r, l.out)
	for idx = 0; idx < m*l.out; idx++ {
		y[idx] += scale * tmp[idx]
	}
	return xh
}

// grads: given dy (m x out), returns dA (r x in) and dB (out x r), unstepped.
func (l *loraLinear) grads(dy []float32, x []float32, xh []float32, m int) ([]float32, []float32) {
	var scale float32 = loraAlpha / float32(l.r)
	var dB []float32 = make([]float32, l.out*l.r)
	var j int = 0
	for j = 0; j < l.out; j++ {
		var t int = 0
		for t = 0; t < l.r; t++ {
			var acc float32 = 0
			var i int = 0
			for i = 0; i < m; i++ {
				acc += dy[i*l.out+j] * xh[i*l.r+t]
			}
			dB[j*l.r+t] = scale * acc
		}
	}
	var dyB []float32 = make([]float32, m*l.r)
	gemmXBT(dyB, dy, l.B, m, l.out, l.r)
	var dA []float32 = make([]float32, l.r*l.in)
	var t2 int = 0
	for t2 = 0; t2 < l.r; t2++ {
		var k int = 0
		for k = 0; k < l.in; k++ {
			var acc float32 = 0
			var i int = 0
			for i = 0; i < m; i++ {
				acc += dyB[i*l.r+t2] * x[i*l.in+k]
			}
			dA[t2*l.in+k] = scale * acc
		}
	}
	return dA, dB
}

// backward: grads + one AdamW step on A and B.
func (l *loraLinear) backward(dy []float32, x []float32, xh []float32, m int) {
	var dA []float32
	var dB []float32
	dA, dB = l.grads(dy, x, xh, m)
	l.step++
	var b1t float32 = float32(math.Pow(float64(adamBeta1), float64(l.step)))
	var b2t float32 = float32(math.Pow(float64(adamBeta2), float64(l.step)))
	var idx int = 0
	for idx = 0; idx < l.r*l.in; idx++ {
		l.mA[idx] = adamBeta1*l.mA[idx] + (1-adamBeta1)*dA[idx]
		l.vA[idx] = adamBeta2*l.vA[idx] + (1-adamBeta2)*dA[idx]*dA[idx]
		var mh float32 = l.mA[idx] / (1 - b1t)
		var vh float32 = l.vA[idx] / (1 - b2t)
		l.A[idx] -= adamLR * mh / (float32(math.Sqrt(float64(vh))) + adamEps)
	}
	for idx = 0; idx < l.out*l.r; idx++ {
		l.mB[idx] = adamBeta1*l.mB[idx] + (1-adamBeta1)*dB[idx]
		l.vB[idx] = adamBeta2*l.vB[idx] + (1-adamBeta2)*dB[idx]*dB[idx]
		var mh float32 = l.mB[idx] / (1 - b1t)
		var vh float32 = l.vB[idx] / (1 - b2t)
		l.B[idx] -= adamLR * mh / (float32(math.Sqrt(float64(vh))) + adamEps)
	}
}

// simple xorshift rng for reproducible init
var rngState uint64 = 0x9E3779B97F4A7C15

func randF32() float32 {
	rngState ^= rngState << 13
	rngState ^= rngState >> 7
	rngState ^= rngState << 17
	return float32(rngState>>40) / float32(1<<24)
}
