//go:build !amd64 || !cgo

// gemm_scalar.go - pure-Go fallback GEMM kernels (no cgo / non-amd64).
// Same entry points as the AVX2 path (gemm_simd.go); ~10 GF/s instead of
// the SIMD kernels. Used when CGO_ENABLED=0.
package main

// gemmFwdSimd: y[m*out+j] += sum_k x[m*in+k]*W[j*in+k], dot4 per row.
func gemmFwdSimd(y []float32, x []float32, w []float32, m int, in int, out int) {
	var row int = 0
	for row = 0; row < m; row++ {
		var xr int = row * in
		var yr int = row * out
		var j int = 0
		for j = 0; j < out; j++ {
			y[yr+j] += dot4(x[xr:xr+in], w[j*in:j*in+in], in)
		}
	}
}

// gemmBwdSimd: dx[m*in+i] += sum_k dy[m*out+k]*W[k*in+i]. The caller must
// zero dx before the first chunk.
func gemmBwdSimd(dx []float32, dy []float32, w []float32, m int, in int, out int) {
	var row int = 0
	for row = 0; row < m; row++ {
		var yr int = row * out
		var xr int = row * in
		var i int = 0
		for i = 0; i < in; i++ {
			var acc float32 = 0
			var k int = 0
			for k = 0; k < out; k++ {
				acc += dy[yr+k] * w[k*in+i]
			}
			dx[xr+i] += acc
		}
	}
}
