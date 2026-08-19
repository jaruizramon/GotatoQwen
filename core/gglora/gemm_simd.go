//go:build amd64 && cgo

// gemm_simd.go - cgo bridge to the AVX2 GEMM kernels (gemm.c).
//
// The GoTorch split: Go owns orchestration (sessions, routing, the training
// loop); C intrinsics own the hot GEMMs. The caller-facing signatures stay
// the whole-GEMM form; the wrappers chunk the output/reduction dimension
// (gemmChunk columns per call) and parallelize the chunks across the cores
// so each W-chunk is read from DRAM once and reused from L2 across all m
// rows. When CGO_ENABLED=0 or the target is not amd64, gemm_scalar.go
// provides the same entry points in pure Go.
package main

/*
#cgo CFLAGS: -O2
void gemmFwd(float *y, const float *x, const float *w, int n0, int nb, int m, int in, int out);
void gemmBwd(float *dx, const float *dy, const float *w, int k0, int kb, int m, int in, int out);
*/
import "C"

import "unsafe"

// gemmChunk: output/reduction columns per kernel call. 64 columns * in
// floats fits the per-core L2 (256KB on the i7-7700HQ) for in <= 1024.
var gemmChunk int = 64

// gemmFwdSimd: y[m*out+j] += sum_k x[m*in+k]*W[j*in+k] (AVX2, 8x8 block).
func gemmFwdSimd(y []float32, x []float32, w []float32, m int, in int, out int) {
	if len(y) == 0 || len(x) == 0 || len(w) == 0 {
		return
	}
	var nchunks int = (out + gemmChunk - 1) / gemmChunk
	parallel(nchunks, func(lo int, hi int) {
		var c int = lo
		for c = lo; c < hi; c++ {
			var n0 int = c * gemmChunk
			var nb int = gemmChunk
			if n0+nb > out {
				nb = out - n0
			}
			C.gemmFwd((*C.float)(unsafe.Pointer(&y[0])), (*C.float)(unsafe.Pointer(&x[0])),
				(*C.float)(unsafe.Pointer(&w[0])), C.int(n0), C.int(nb), C.int(m), C.int(in), C.int(out))
		}
	})
}

// gemmBwdSimd: dx[m*in+i] += sum_k dy[m*out+k]*W[k*in+i] (AVX2, 8 lanes).
// The caller must zero dx before the first chunk.
func gemmBwdSimd(dx []float32, dy []float32, w []float32, m int, in int, out int) {
	if len(dx) == 0 || len(dy) == 0 || len(w) == 0 {
		return
	}
	var nchunks int = (out + gemmChunk - 1) / gemmChunk
	parallel(nchunks, func(lo int, hi int) {
		var c int = lo
		for c = lo; c < hi; c++ {
			var k0 int = c * gemmChunk
			var kb int = gemmChunk
			if k0+kb > out {
				kb = out - k0
			}
			C.gemmBwd((*C.float)(unsafe.Pointer(&dx[0])), (*C.float)(unsafe.Pointer(&dy[0])),
				(*C.float)(unsafe.Pointer(&w[0])), C.int(k0), C.int(kb), C.int(m), C.int(in), C.int(out))
		}
	})
}
