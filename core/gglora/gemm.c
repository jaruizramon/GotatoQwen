// gemm.c - AVX2 GEMM kernels for the gglora training loop (cgo).
//
// Why C at all: Go's compiler does not auto-vectorize (go1.22 emits zero
// FMA instructions even with GOAMD64=v3), so pure-Go kernels cap at ~10
// GF/s on this box. PyTorch's speed comes from oneDNN/ATen AVX2 kernels;
// these kernels are the same idea for the shapes that dominate the trainer.
// The "GoTorch" split: Go owns orchestration and the training loop; C
// intrinsics own the hot GEMMs.
//
//   gemmFwd: y[m][out] += x[m][in] * W[out][in]   (W row-major)
//            FFN projections + the tied-embedding head.
//   gemmBwd: dx[m][in] += dy[m][out] * W[out][in]
//            the dW^T pass of the frozen base weights (and the head
//            softmax backward, dx = (p-onehot) * E).
//
// Cache structure (the measured bottleneck was DRAM, not FLOPs): each call
// covers a CHUNK of the output/reduction dimension, and the caller
// parallelizes over chunks. The W-chunk (nb*in floats) is streamed from
// DRAM once per call and reused across ALL m rows from L2 - W traffic is
// out*nb*4 bytes instead of out*in*4*m (m=256x less). The 8-row x 8-lane
// micro-kernel keeps 8 independent FMA accumulator chains in registers.
//
// Determinism: a fixed instruction stream is bit-reproducible on a given
// machine. The summation order differs from the Go scalar path, so the
// SIMD and scalar paths agree to float tolerance, not bit-exactly; gglora
// is the reference math (the torch parity harness was retired with the
// python trainer).
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define SIMD_ATTR __attribute__((target("avx2,fma")))
#else
#define SIMD_ATTR
#endif

SIMD_ATTR static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// gemmFwd: output columns [n0, n0+nb). 2x8 register block: two x-rows per
// pass, 8 W rows, 16 independent FMA accumulator chains (fits the 16 YMM
// registers). W-chunk L2 traffic halves vs the 1x8 block: each W row is
// reused for two x-rows before the next k-step.
SIMD_ATTR void gemmFwd(float *y, const float *x, const float *w, int n0, int nb, int m, int in, int out) {
    int row, jj, k;
    for (row = 0; row + 2 <= m; row += 2) {
        const float *xr0 = x + (row + 0) * in;
        const float *xr1 = x + (row + 1) * in;
        float *yr0 = y + (row + 0) * out;
        float *yr1 = y + (row + 1) * out;
        for (jj = n0; jj < n0 + nb; jj += 8) {
            const float *w0 = w + (jj + 0) * in;
            const float *w1 = w + (jj + 1) * in;
            const float *w2 = w + (jj + 2) * in;
            const float *w3 = w + (jj + 3) * in;
            const float *w4 = w + (jj + 4) * in;
            const float *w5 = w + (jj + 5) * in;
            const float *w6 = w + (jj + 6) * in;
            const float *w7 = w + (jj + 7) * in;
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
            __m256 a04 = _mm256_setzero_ps(), a05 = _mm256_setzero_ps();
            __m256 a06 = _mm256_setzero_ps(), a07 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
            __m256 a14 = _mm256_setzero_ps(), a15 = _mm256_setzero_ps();
            __m256 a16 = _mm256_setzero_ps(), a17 = _mm256_setzero_ps();
            for (k = 0; k + 8 <= in; k += 8) {
                __m256 xv0 = _mm256_loadu_ps(xr0 + k);
                __m256 xv1 = _mm256_loadu_ps(xr1 + k);
                a00 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w0 + k), a00);
                a01 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w1 + k), a01);
                a02 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w2 + k), a02);
                a03 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w3 + k), a03);
                a04 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w4 + k), a04);
                a05 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w5 + k), a05);
                a06 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w6 + k), a06);
                a07 = _mm256_fmadd_ps(xv0, _mm256_loadu_ps(w7 + k), a07);
                a10 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w0 + k), a10);
                a11 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w1 + k), a11);
                a12 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w2 + k), a12);
                a13 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w3 + k), a13);
                a14 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w4 + k), a14);
                a15 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w5 + k), a15);
                a16 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w6 + k), a16);
                a17 = _mm256_fmadd_ps(xv1, _mm256_loadu_ps(w7 + k), a17);
            }
            float t00 = 0, t01 = 0, t02 = 0, t03 = 0, t04 = 0, t05 = 0, t06 = 0, t07 = 0;
            float t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0, t16 = 0, t17 = 0;
            for (; k < in; k++) {
                float x0 = xr0[k], x1 = xr1[k];
                t00 += x0 * w0[k]; t01 += x0 * w1[k]; t02 += x0 * w2[k]; t03 += x0 * w3[k];
                t04 += x0 * w4[k]; t05 += x0 * w5[k]; t06 += x0 * w6[k]; t07 += x0 * w7[k];
                t10 += x1 * w0[k]; t11 += x1 * w1[k]; t12 += x1 * w2[k]; t13 += x1 * w3[k];
                t14 += x1 * w4[k]; t15 += x1 * w5[k]; t16 += x1 * w6[k]; t17 += x1 * w7[k];
            }
            yr0[jj + 0] += hsum8(a00) + t00; yr0[jj + 1] += hsum8(a01) + t01;
            yr0[jj + 2] += hsum8(a02) + t02; yr0[jj + 3] += hsum8(a03) + t03;
            yr0[jj + 4] += hsum8(a04) + t04; yr0[jj + 5] += hsum8(a05) + t05;
            yr0[jj + 6] += hsum8(a06) + t06; yr0[jj + 7] += hsum8(a07) + t07;
            yr1[jj + 0] += hsum8(a10) + t10; yr1[jj + 1] += hsum8(a11) + t11;
            yr1[jj + 2] += hsum8(a12) + t12; yr1[jj + 3] += hsum8(a13) + t13;
            yr1[jj + 4] += hsum8(a14) + t14; yr1[jj + 5] += hsum8(a15) + t15;
            yr1[jj + 6] += hsum8(a16) + t16; yr1[jj + 7] += hsum8(a17) + t17;
        }
    }
    for (; row < m; row++) {
        const float *xr = x + row * in;
        float *yr = y + row * out;
        for (jj = n0; jj < n0 + nb; jj += 8) {
            const float *w0 = w + (jj + 0) * in;
            const float *w1 = w + (jj + 1) * in;
            const float *w2 = w + (jj + 2) * in;
            const float *w3 = w + (jj + 3) * in;
            const float *w4 = w + (jj + 4) * in;
            const float *w5 = w + (jj + 5) * in;
            const float *w6 = w + (jj + 6) * in;
            const float *w7 = w + (jj + 7) * in;
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps();
            __m256 a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps();
            __m256 a7 = _mm256_setzero_ps();
            for (k = 0; k + 8 <= in; k += 8) {
                __m256 xv = _mm256_loadu_ps(xr + k);
                a0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + k), a0);
                a1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + k), a1);
                a2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + k), a2);
                a3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + k), a3);
                a4 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w4 + k), a4);
                a5 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w5 + k), a5);
                a6 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w6 + k), a6);
                a7 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w7 + k), a7);
            }
            float t0 = 0, t1 = 0, t2 = 0, t3 = 0;
            float t4 = 0, t5 = 0, t6 = 0, t7 = 0;
            for (; k < in; k++) {
                float xv = xr[k];
                t0 += xv * w0[k]; t1 += xv * w1[k];
                t2 += xv * w2[k]; t3 += xv * w3[k];
                t4 += xv * w4[k]; t5 += xv * w5[k];
                t6 += xv * w6[k]; t7 += xv * w7[k];
            }
            yr[jj + 0] += hsum8(a0) + t0;
            yr[jj + 1] += hsum8(a1) + t1;
            yr[jj + 2] += hsum8(a2) + t2;
            yr[jj + 3] += hsum8(a3) + t3;
            yr[jj + 4] += hsum8(a4) + t4;
            yr[jj + 5] += hsum8(a5) + t5;
            yr[jj + 6] += hsum8(a6) + t6;
            yr[jj + 7] += hsum8(a7) + t7;
        }
        for (; jj < n0 + nb; jj++) {
            const float *wj = w + jj * in;
            float acc = 0;
            for (k = 0; k < in; k++) {
                acc += xr[k] * wj[k];
            }
            yr[jj] += acc;
        }
    }
}

// gemmBwd: dx[m][in] += dy[m][out] * WT[in][out], reduction rows
// [k0, k0+kb) of the out dimension. WT is the pre-transposed W (built at
// load time), so the reduction reads WT[i] rows contiguously (4KB per row,
// 8 in-lanes per pass, 4 interleaved FMA chains). The WT-chunk is streamed
// once per row and reused across all m rows; dx is read-modify-written per
// chunk (the caller zeroes dx once before the first chunk).
SIMD_ATTR void gemmBwd(float *dx, const float *dy, const float *wt, int k0, int kb, int m, int in, int out) {
    int row, i, k;
    for (row = 0; row < m; row++) {
        const float *dyr = dy + row * out;
        float *dxr = dx + row * in;
        for (i = 0; i + 8 <= in; i += 8) {
            const float *wtrow = wt + i * out;
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            for (k = k0; k + 32 <= k0 + kb; k += 32) {
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(dyr[k + 0]), _mm256_loadu_ps(wtrow + k + 0), a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(dyr[k + 8]), _mm256_loadu_ps(wtrow + k + 8), a1);
                a2 = _mm256_fmadd_ps(_mm256_set1_ps(dyr[k + 16]), _mm256_loadu_ps(wtrow + k + 16), a2);
                a3 = _mm256_fmadd_ps(_mm256_set1_ps(dyr[k + 24]), _mm256_loadu_ps(wtrow + k + 24), a3);
            }
            for (; k < k0 + kb; k += 8) {
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(dyr[k]), _mm256_loadu_ps(wtrow + k), a0);
            }
            __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            __m256 d = _mm256_loadu_ps(dxr + i);
            _mm256_storeu_ps(dxr + i, _mm256_add_ps(d, s));
        }
        for (; i < in; i++) {
            const float *wtrow = wt + i * out;
            float acc = 0;
            for (k = k0; k < k0 + kb; k++) {
                acc += dyr[k] * wtrow[k];
            }
            dxr[i] += acc;
        }
    }
}
