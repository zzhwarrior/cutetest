// One Qwen3.5-9B full_attention layer, quantized to NVFP4 on CUTE.
//
// Config reference: https://modelscope.cn/models/Qwen/Qwen3.5-9B config.json
//   hidden_size=4096, num_attention_heads=16, num_key_value_heads=4 (GQA 4:1),
//   head_dim=256, partial_rotary_factor=0.25 (rope on first 64/256),
//   rope_theta=1e7, attn_output_gate=true.
// Note: only the "full_attention" layer variant is modeled; the 24 linear_attention
// layers of Qwen3.5 are out of scope.
//
// Data path:
//   A/B NVFP4 (E2M1, 4-bit, 2/byte, low nibble first — matches FP4toint.scala:1146)
//   per-16-element E4M3 block scales (matches CUTEParameters.scala:1082 nvfp4ScaleWidth)
//   accumulator FP32
//   no per-tensor FP32 global scale (CUTE differs from NVIDIA public NVFP4 here)
//
// Weights are zero-initialized (dummy) — this test measures cycle count only,
// not numerical correctness. CUTE PE schedule is data-independent so cycle
// counts are meaningful with zero inputs.

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>
#include <assert.h>
#include "cuteMarcoinstHelper.h"

// -------- After-op codes (private to this file) --------
#define NO_ACTIVATION                    0
#define FUSE_NVFP4_QNORM_PARTROPE_BF16   1   // Q/K path: per-head RMSnorm + partial RoPE (first 64 dims) + bf16 cvrt
#define FUSE_NVFP4_BF16CVRT_T            2   // V path: bf16 cvrt with transpose
#define FUSE_NVFP4_TO_F32                3   // gate path: FP32 pass-through
#define FUSE_MASKED_SOFTMAX_BF16CVRT     4   // score path: masked softmax + bf16 cvrt (input already fp32)
#define FUSE_NVFP4_RESADD                5   // o_proj path: residual add on FP32

// -------- Model constants (Qwen3.5-9B, full_attention layer) --------
#define SEQ_LEN         128
#define HIDDEN          4096
#define N_HEAD_Q        16
#define N_HEAD_KV       4
#define HEAD_DIM        256
#define ROPE_DIM        64                      // partial_rotary_factor 0.25 * 256
#define ROPE_THETA      10000000.0f
#define RMS_EPSILON     1e-6f
#define NVFP4_BLOCK     16                      // NVFP4 group size (fixed by HW)
#define INV_SQRT_HDIM   0.0625f                 // 1/sqrt(256)
#define MAX_CTX_LEN     8192

// FP4 packed 2/byte, so K-byte-stride = K/2 (bytes) along the reduce axis.
#define FP4_BYTES(K)    ((K) / 2)
#define SCALE_BYTES(K)  ((K) / NVFP4_BLOCK)

// -------- RoPE theta (only first ROPE_DIM/2 = 32 rotations) --------
// theta_j = 1 / rope_theta ^ (2j / ROPE_DIM), j = 0..31
static float rope_theta[ROPE_DIM/2] __attribute__((aligned(64))) = {
    1.0000000e+00f, 5.6234133e-01f, 3.1622776e-01f, 1.7782794e-01f,
    1.0000000e-01f, 5.6234132e-02f, 3.1622775e-02f, 1.7782794e-02f,
    1.0000000e-02f, 5.6234131e-03f, 3.1622776e-03f, 1.7782794e-03f,
    1.0000000e-03f, 5.6234130e-04f, 3.1622774e-04f, 1.7782794e-04f,
    1.0000000e-04f, 5.6234133e-05f, 3.1622776e-05f, 1.7782794e-05f,
    1.0000000e-05f, 5.6234132e-06f, 3.1622776e-06f, 1.7782794e-06f,
    1.0000000e-06f, 5.6234131e-07f, 3.1622774e-07f, 1.7782794e-07f,
    1.0000000e-07f, 5.6234130e-08f, 3.1622775e-08f, 1.7782793e-08f
};

// -------- Attention mask (causal) --------
static int8_t bitmask_ptr[SEQ_LEN][SEQ_LEN/8] __attribute__((aligned(64))) = {0};

// -------- Residual / input hidden state (FP32) --------
static float identity[SEQ_LEN][HIDDEN] __attribute__((aligned(64))) = {0};

// -------- Norm weights (all FP32) --------
static float attn_norm_w[HIDDEN] __attribute__((aligned(64))) = {0};
static float q_norm_w[HEAD_DIM] __attribute__((aligned(64))) = {0};
static float k_norm_w[HEAD_DIM] __attribute__((aligned(64))) = {0};

// -------- Weight tensors (NVFP4 packed 2/byte) --------
// Layout convention: [N][K/2] — same as llama's [N][K] int8 layout, halved.
static uint8_t proj_q_w_fp4[HIDDEN][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_k_w_fp4[N_HEAD_KV*HEAD_DIM][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_v_w_fp4[N_HEAD_KV*HEAD_DIM][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_o_w_fp4[HIDDEN][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t gate_w_fp4 [HIDDEN][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};

// -------- Weight block-scales (E4M3, one byte per NVFP4_BLOCK along K) --------
// Layout: [N][K/16] — one E4M3 per K-block per output channel.
static uint8_t proj_q_ws[HIDDEN][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_k_ws[N_HEAD_KV*HEAD_DIM][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_v_ws[N_HEAD_KV*HEAD_DIM][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t proj_o_ws[HIDDEN][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t gate_ws [HIDDEN][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};

// -------- Activation quant buffers (NVFP4 + E4M3 block scales) --------
// Input to Q/K/V/gate projections: quantized hidden state (post attn_norm).
static uint8_t x_norm_fp4[SEQ_LEN][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t x_norm_scale[SEQ_LEN][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
// Input to o_proj: quantized gated attention output.
static uint8_t attn_g_fp4[SEQ_LEN][FP4_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};
static uint8_t attn_g_scale[SEQ_LEN][SCALE_BYTES(HIDDEN)] __attribute__((aligned(64))) = {0};

// -------- Q / K / V post-projection buffers (BF16, after per-head norm + partial RoPE) --------
// Q: [SEQ, N_HEAD_Q, HEAD_DIM], K/V: [SEQ, N_HEAD_KV, HEAD_DIM]
static _Float16 q_buf_bf16[SEQ_LEN][N_HEAD_Q][HEAD_DIM] __attribute__((aligned(64))) = {0};
static _Float16 k_buf_bf16[SEQ_LEN][N_HEAD_KV][HEAD_DIM] __attribute__((aligned(64))) = {0};
static _Float16 v_buf_bf16[SEQ_LEN][N_HEAD_KV][HEAD_DIM] __attribute__((aligned(64))) = {0};

// -------- Score / attn intermediate buffers --------
static _Float16 scores_bf16[N_HEAD_Q][SEQ_LEN][SEQ_LEN] __attribute__((aligned(64))) = {0};
static float    attn_f32[SEQ_LEN][HIDDEN] __attribute__((aligned(64))) = {0};
static float    gate_f32[SEQ_LEN][HIDDEN] __attribute__((aligned(64))) = {0};

// -------- CUTE result ping-pong buffers (TCM at 0x70200000, 4 x 64x64 FP32 tiles = 64 KB) --------
static void *CUTE_result[4] = {
    (void *)(0x70200000),
    (void *)(0x70200000 + 64 * 64 * 4),
    (void *)(0x70200000 + 64 * 64 * 4 * 2),
    (void *)(0x70200000 + 64 * 64 * 4 * 3)
};
static int CUTE_result_index = 0;

static inline int next_result_idx(int i) { return (i == 3) ? 0 : (i + 1); }

// -------- RVV helpers: sin / cos / exp (copied from llama3_1B.c) --------

static inline vfloat32m4_t vec_sin_small(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl);
    vfloat32m4_t c1  = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c3  = __riscv_vfmv_v_f_f32m4(-0.166666666667f, vl);
    vfloat32m4_t c5  = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c7  = __riscv_vfmv_v_f_f32m4(-0.0001984126984f, vl);
    vfloat32m4_t c9  = __riscv_vfmv_v_f_f32m4(0.000002755731922f, vl);
    vfloat32m4_t c11 = __riscv_vfmv_v_f_f32m4(-0.000000025052108f, vl);
    vfloat32m4_t result;
    result = __riscv_vfmacc_vv_f32m4(c9,     c11, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c7,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c5,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c3,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c1,  result, x2, vl);
    result = __riscv_vfmul_vv_f32m4(result, x, vl);
    return result;
}

static inline vfloat32m4_t vec_sin(vfloat32m4_t x, size_t vl) {
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    vfloat32m4_t pi_vec = __riscv_vfmv_v_f_f32m4(PI, vl);
    vfloat32m4_t round = __riscv_vfdiv_vv_f32m4(new_rad, pi_vec, vl);
    vfloat32m4_t magic = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl);
    round = __riscv_vfadd_vv_f32m4(round, magic, vl);
    round = __riscv_vfsub_vv_f32m4(round, magic, vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmul_vv_f32m4(round, pi_vec, vl), vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    vfloat32m4_t sin_result = vec_sin_small(new_rad, vl);
    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl);
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

static inline vfloat32m4_t vec_cos(vfloat32m4_t x, size_t vl) {
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return vec_sin(new_rad, vl);
}

static inline vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl) {
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    const int32_t MAX_A = 127;
    const int32_t MIN_A = -126;

    vfloat32m4_t af = __riscv_vfmul_vf_f32m4(x, INV_LN2, vl);
    vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl);
    vfloat32m4_t a = __riscv_vfadd_vv_f32m4(af, r, vl);
    a = __riscv_vfsub_vv_f32m4(a, r, vl);
    vint32m4_t a_int = __riscv_vfcvt_x_f_v_i32m4(a, vl);
    vbool8_t mask_max = __riscv_vmsgt_vx_i32m4_b8(a_int, MAX_A, vl);
    vbool8_t mask_min = __riscv_vmslt_vx_i32m4_b8(a_int, MIN_A, vl);

    vint32m4_t biased_exponent = __riscv_vadd_vx_i32m4(a_int, 127, vl);
    biased_exponent = __riscv_vsll_vx_i32m4(biased_exponent, 23, vl);
    vfloat32m4_t a2 = __riscv_vreinterpret_v_i32m4_f32m4(biased_exponent);

    vfloat32m4_t b = __riscv_vfmacc_vf_f32m4(x, NEG_LN2, a, vl);

    vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.166666666667f, vl);
    vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.041666666667f, vl);
    vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c6 = __riscv_vfmv_v_f_f32m4(0.001388888889f, vl);
    vfloat32m4_t p;
    p = __riscv_vfmacc_vv_f32m4(c5, c6, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c4,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, b, vl);
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(INFINITY, vl), mask_max, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(0.0f, vl), mask_min, vl);
    return p;
}

static inline vfloat32m4_t vec_sigmoid(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t neg = __riscv_vfneg_v_f32m4(x, vl);
    vfloat32m4_t enx = vec_exp(neg, vl);
    vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t denom = __riscv_vfadd_vv_f32m4(one, enx, vl);
    return __riscv_vfdiv_vv_f32m4(one, denom, vl);
}

// FP4 E2M1: sign(1) exp(2) mantissa(1). Representable magnitudes:
//   0000=0.0, 0001=0.5, 0010=1.0, 0011=1.5, 0100=2.0, 0101=3.0, 0110=4.0, 0111=6.0
// Rounding: nearest-even to the eight magnitude codes.
static inline uint8_t fp32_to_fp4_nibble(float x) {
    uint32_t bits = *(uint32_t *)&x;
    uint8_t sign = (bits >> 31) & 1;
    float ax = x < 0 ? -x : x;
    uint8_t mag;
    if      (ax < 0.25f) mag = 0;   // 0.0
    else if (ax < 0.75f) mag = 1;   // 0.5
    else if (ax < 1.25f) mag = 2;   // 1.0
    else if (ax < 1.75f) mag = 3;   // 1.5
    else if (ax < 2.5f)  mag = 4;   // 2.0
    else if (ax < 3.5f)  mag = 5;   // 3.0
    else if (ax < 5.0f)  mag = 6;   // 4.0
    else                 mag = 7;   // 6.0
    return (uint8_t)((sign << 3) | mag);
}

// E4M3: sign(1) exp(4) mantissa(3), bias=7. Range ~[2^-9, 448]. No inf; 0xFF=NaN.
// Encode |x| directly (block scale is unsigned in practice — sign bit forced 0).
static inline uint8_t fp32_to_e4m3(float x) {
    uint32_t bits = *(uint32_t *)&x;
    uint32_t f_exp = (bits >> 23) & 0xFF;
    uint32_t f_mant = bits & 0x7FFFFF;
    if (f_exp == 0) return 0;
    int32_t e = (int32_t)f_exp - 127 + 7;
    if (e <= 0) return 0;
    if (e >= 15) return 0x7E;               // clamp near-max (avoid NaN code 0x7F)
    uint32_t m3 = f_mant >> 20;              // top 3 bits
    return (uint8_t)((e << 3) | m3);
}

static inline float e4m3_to_fp32(uint8_t b) {
    uint32_t e = (b >> 3) & 0xF;
    uint32_t m = b & 0x7;
    if (e == 0) return (float)m * (1.0f / 512.0f);
    uint32_t bits = (uint32_t)((e + 127 - 7) << 23) | (m << 20);
    return *(float *)&bits;
}

// Per-16-block NVFP4 quantization: [rows][cols] FP32 → packed FP4 + E4M3 scales.
// scale_out layout: [rows][cols/16]. fp4_out layout: [rows][cols/2], low nibble first.
static void quant_nvfp4_block16(const float *in, uint8_t *fp4_out, uint8_t *scale_out,
                                int rows, int cols,
                                uint64_t in_stride_bytes, uint64_t fp4_stride_bytes, uint64_t scale_stride_bytes)
{
    const float FP4_MAX = 6.0f;
    for (int r = 0; r < rows; r++) {
        const float *row_in = (const float *)((const uint8_t *)in + r * in_stride_bytes);
        uint8_t *row_fp4 = fp4_out + r * fp4_stride_bytes;
        uint8_t *row_scale = scale_out + r * scale_stride_bytes;
        for (int kb = 0; kb < cols; kb += NVFP4_BLOCK) {
            size_t vl = __riscv_vsetvl_e32m4(NVFP4_BLOCK);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(row_in + kb, vl);
            vfloat32m4_t vabs = __riscv_vfsgnj_vf_f32m4(vx, 1.0f, vl);
            float amax = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredmax_vs_f32m4_f32m1(vabs, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl));
            float scale = amax / FP4_MAX;
            uint8_t scale_e4m3 = fp32_to_e4m3(scale);
            row_scale[kb / NVFP4_BLOCK] = scale_e4m3;
            float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            for (int j = 0; j < NVFP4_BLOCK; j += 2) {
                float v0 = row_in[kb + j]     * inv_scale;
                float v1 = row_in[kb + j + 1] * inv_scale;
                uint8_t n0 = fp32_to_fp4_nibble(v0);
                uint8_t n1 = fp32_to_fp4_nibble(v1);
                row_fp4[(kb + j) / 2] = (uint8_t)(n0 | (n1 << 4));
            }
        }
    }
}

// FP32 RMSNorm over `cols`, weight vector `w[cols]`, then write FP32 result.
// scratch is a per-token scratch of size `cols` floats (aligned).
static void rmsnorm_fp32(const float *x, float *y, const float *w,
                         int rows, int cols,
                         uint64_t x_stride_bytes, uint64_t y_stride_bytes)
{
    for (int r = 0; r < rows; r++) {
        const float *xr = (const float *)((const uint8_t *)x + r * x_stride_bytes);
        float *yr = (float *)((uint8_t *)y + r * y_stride_bytes);
        size_t vl_full = __riscv_vsetvl_e32m4(cols);
        vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl_full);
        size_t vl;
        for (int k = 0, avl = cols; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(xr + k, vl);
            vsum = __riscv_vfmacc_vv_f32m4(vsum, v, v, vl);
        }
        float sum_sq = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.0f, vl_full), vl_full));
        float mean_sq = sum_sq / (float)cols;
        float rms_inv;
        __asm__ volatile ("fsqrt.s %0, %1" : "=f"(rms_inv) : "f"(mean_sq + RMS_EPSILON));
        rms_inv = 1.0f / rms_inv;
        for (int k = 0, avl = cols; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v  = __riscv_vle32_v_f32m4(xr + k, vl);
            vfloat32m4_t vw = __riscv_vle32_v_f32m4(w + k, vl);
            v = __riscv_vfmul_vf_f32m4(v, rms_inv, vl);
            v = __riscv_vfmul_vv_f32m4(v, vw, vl);
            __riscv_vse32_v_f32m4(yr + k, v, vl);
        }
    }
}

// Per-head RMSnorm + partial RoPE (first ROPE_DIM of HEAD_DIM), applied in-place
// on bf16 buffer [seq][n_head][head_dim]. Norm weight is head_dim-shaped.
static void qk_norm_and_partial_rope_bf16(_Float16 *qk_bf16, const float *norm_w,
                                          int seq, int n_head, int pos_base)
{
    for (int s = 0; s < seq; s++) {
        int pos_ = s + pos_base;
        for (int h = 0; h < n_head; h++) {
            _Float16 *head = qk_bf16 + (s * n_head + h) * HEAD_DIM;

            // per-head RMSnorm over HEAD_DIM
            size_t vl_full = __riscv_vsetvl_e32m4(HEAD_DIM);
            vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl_full);
            size_t vl;
            for (int k = 0, avl = HEAD_DIM; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat16m2_t vh = __riscv_vle16_v_f16m2(head + k, vl);
                vfloat32m4_t vf = __riscv_vfwcvt_f_f_v_f32m4(vh, vl);
                vsum = __riscv_vfmacc_vv_f32m4(vsum, vf, vf, vl);
            }
            float sum_sq = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m4_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.0f, vl_full), vl_full));
            float rms_inv;
            __asm__ volatile ("fsqrt.s %0, %1" : "=f"(rms_inv) : "f"(sum_sq / (float)HEAD_DIM + RMS_EPSILON));
            rms_inv = 1.0f / rms_inv;

            // scale + weight, write back f32 to a small stack buffer for RoPE step
            float head_f32[HEAD_DIM] __attribute__((aligned(64)));
            for (int k = 0, avl = HEAD_DIM; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat16m2_t vh = __riscv_vle16_v_f16m2(head + k, vl);
                vfloat32m4_t vf = __riscv_vfwcvt_f_f_v_f32m4(vh, vl);
                vfloat32m4_t vw = __riscv_vle32_v_f32m4(norm_w + k, vl);
                vf = __riscv_vfmul_vf_f32m4(vf, rms_inv, vl);
                vf = __riscv_vfmul_vv_f32m4(vf, vw, vl);
                __riscv_vse32_v_f32m4(head_f32 + k, vf, vl);
            }

            // Partial RoPE on first ROPE_DIM (=64) dims, layout: interleaved real/imag pairs.
            // Same rotate formula as llama's rope:
            //   real' = real*cos - imag*sin
            //   imag' = real*sin + imag*cos
            const int half = ROPE_DIM / 2;
            for (int k = 0, avl = half; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t theta_vec = __riscv_vle32_v_f32m4(rope_theta + k, vl);
                vfloat32m4_t angle_vec = __riscv_vfmul_vf_f32m4(theta_vec, (float)pos_, vl);
                vfloat32m4_t sin_vec = vec_sin(angle_vec, vl);
                vfloat32m4_t cos_vec = vec_cos(angle_vec, vl);
                vfloat32m4_t r = __riscv_vlse32_v_f32m4(head_f32 + 2 * k,     2 * sizeof(float), vl);
                vfloat32m4_t i = __riscv_vlse32_v_f32m4(head_f32 + 2 * k + 1, 2 * sizeof(float), vl);
                vfloat32m4_t rout = __riscv_vfmsub_vv_f32m4(r, cos_vec,
                                        __riscv_vfmul_vv_f32m4(i, sin_vec, vl), vl);
                vfloat32m4_t iout = __riscv_vfmacc_vv_f32m4(
                                        __riscv_vfmul_vv_f32m4(r, sin_vec, vl), i, cos_vec, vl);
                __riscv_vsse32_v_f32m4(head_f32 + 2 * k,     2 * sizeof(float), rout, vl);
                __riscv_vsse32_v_f32m4(head_f32 + 2 * k + 1, 2 * sizeof(float), iout, vl);
            }

            // Write back full HEAD_DIM as bf16 (both RoPE'd prefix and untouched tail).
            for (int k = 0, avl = HEAD_DIM; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vf = __riscv_vle32_v_f32m4(head_f32 + k, vl);
                vfloat16m2_t vh = __riscv_vfncvt_f_f_w_f16m2(vf, vl);
                __riscv_vse16_v_f16m2(head + k, vh, vl);
            }
        }
    }
}

// sigmoid(g_f32) * attn_f32 → attn_out_f32 (in place on attn_f32).
static void sigmoid_hadamard_f32(float *attn, const float *g, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float *ar = attn + r * cols;
        const float *gr = g + r * cols;
        size_t vl;
        for (int k = 0, avl = cols; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vg = __riscv_vle32_v_f32m4(gr + k, vl);
            vfloat32m4_t vs = vec_sigmoid(vg, vl);
            vfloat32m4_t va = __riscv_vle32_v_f32m4(ar + k, vl);
            va = __riscv_vfmul_vv_f32m4(va, vs, vl);
            __riscv_vse32_v_f32m4(ar + k, va, vl);
        }
    }
}

// -------- NVFP4 fused after-op kernels (CUTE tile → downstream format).
// After-op signature (matches llama's convention so the dispatcher is uniform):
//   input        : CUTE FP32 tile output (64 x tile_N floats, from CUTE_result[idx])
//   output       : write pointer in the target buffer
//   input_scale  : (unused for NVFP4 — CUTE already applied per-block scale)
//   weight_scale : (unused)
//   dim_i, dim_j : tile row/col count
//   input_stride, output_stride : byte strides in respective buffers
//   extra        : op-specific (RoPE pos ptr, gate buffer ptr, etc.)

typedef void (*after_op_fn)(void *, void *, void *, void *, int, int, uint64_t, uint64_t, void *);

// FUSE_NVFP4_QNORM_PARTROPE_BF16: after Q/K projection.
// The CUTE tile is [64 x head_slice]; we scatter to bf16 buffer without doing
// norm/rope here (both are done in a separate pass over the full [SEQ, n_head, head_dim] buffer
// after all tiles land, because per-head norm needs the entire head_dim contiguously).
// So this fused op is just fp32→bf16 cvrt to the right output stride.
static void fuse_nvfp4_bf16_cvrt(void *input, void *output,
                                 void *input_scale, void *weight_scale,
                                 int dim_i, int dim_j,
                                 uint64_t input_stride, uint64_t output_stride,
                                 void *extra)
{
    (void)input_scale; (void)weight_scale; (void)extra;
    for (int j = 0; j < dim_i; j++) {
        float    *in_row  = (float *)((uint8_t *)input  + j * input_stride);
        _Float16 *out_row = (_Float16 *)((uint8_t *)output + j * output_stride);
        size_t vl;
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(in_row + k, vl);
            vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(v, vl);
            __riscv_vse16_v_f16m2(out_row + k, h, vl);
        }
    }
}

// FUSE_NVFP4_TO_F32: fp32 pass-through (gate projection lands in FP32 buffer).
static void fuse_nvfp4_to_f32(void *input, void *output,
                              void *input_scale, void *weight_scale,
                              int dim_i, int dim_j,
                              uint64_t input_stride, uint64_t output_stride,
                              void *extra)
{
    (void)input_scale; (void)weight_scale; (void)extra;
    for (int j = 0; j < dim_i; j++) {
        float *in_row  = (float *)((uint8_t *)input  + j * input_stride);
        float *out_row = (float *)((uint8_t *)output + j * output_stride);
        size_t vl;
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(in_row + k, vl);
            __riscv_vse32_v_f32m4(out_row + k, v, vl);
        }
    }
}

// FUSE_NVFP4_RESADD: o_proj + residual add (destination is FP32 identity buffer).
static void fuse_nvfp4_resadd(void *input, void *output,
                              void *input_scale, void *weight_scale,
                              int dim_i, int dim_j,
                              uint64_t input_stride, uint64_t output_stride,
                              void *extra)
{
    (void)input_scale; (void)weight_scale; (void)extra;
    for (int j = 0; j < dim_i; j++) {
        float *in_row  = (float *)((uint8_t *)input  + j * input_stride);
        float *out_row = (float *)((uint8_t *)output + j * output_stride);
        size_t vl;
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_in  = __riscv_vle32_v_f32m4(in_row + k, vl);
            vfloat32m4_t v_res = __riscv_vle32_v_f32m4(out_row + k, vl);
            vfloat32m4_t v_sum = __riscv_vfadd_vv_f32m4(v_in, v_res, vl);
            __riscv_vse32_v_f32m4(out_row + k, v_sum, vl);
        }
    }
}

// FUSE_MASKED_SOFTMAX_BF16CVRT: for the score matmul. Input is bf16-cvrt'd fp32,
// output is bf16. Applies INV_SQRT_HDIM scale, causal mask, softmax, then bf16.
static void fuse_masked_softmax_bf16(void *input, void *output,
                                     void *input_scale, void *weight_scale,
                                     int dim_i, int dim_j,
                                     uint64_t input_stride, uint64_t output_stride,
                                     void *pos_ptr)
{
    (void)input_scale; (void)weight_scale;
    int pos_ = *((int *)pos_ptr);
    for (int i = 0; i < dim_i; i++) {
        float    *row_x = (float *)((uint8_t *)input  + i * input_stride);
        _Float16 *row_y = (_Float16 *)((uint8_t *)output + i * output_stride);
        size_t vl_full = __riscv_vsetvl_e32m4(dim_j);
        size_t vl;

        // stage 1: multiply by INV_SQRT_HDIM
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
            v = __riscv_vfmul_vf_f32m4(v, INV_SQRT_HDIM, vl);
            __riscv_vse32_v_f32m4(row_x + k, v, vl);
        }

        // stage 2: masked max
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(-INFINITY, vl_full);
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
            vbool8_t mask = __riscv_vlm_v_b8(
                (uint8_t *)bitmask_ptr + ((i + pos_) * dim_j + k) / 8, vl);
            v = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), v, mask, vl);
            max_vec = __riscv_vfmax_vv_f32m4(max_vec, v, vl);
        }
        float max_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, vl_full), vl_full));

        // stage 3: exp(x-max), masked; sum for normalization
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_full);
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
            v = __riscv_vfsub_vf_f32m4(v, max_val, vl);
            vbool8_t mask = __riscv_vlm_v_b8(
                (uint8_t *)bitmask_ptr + ((i + pos_) * dim_j + k) / 8, vl);
            v = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-90.0f, vl), v, mask, vl);
            vfloat32m4_t ev = vec_exp(v, vl);
            __riscv_vse32_v_f32m4(row_x + k, ev, vl);
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, ev, vl);
        }
        float sum_exp = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_full), vl_full));

        // stage 4: normalize + cvrt to bf16
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
            v = __riscv_vfmul_vf_f32m4(v, inv_sum, vl);
            vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(v, vl);
            __riscv_vse16_v_f16m2(row_y + k, h, vl);
        }
    }
}

// -------- matmul wrappers --------
//
// matmul_cute_nvfp4: NVFP4 GEMM for M/N/K all multiples of 64. Tiled 64x64x(K),
// double-buffered on CUTE_result[]. Each tile has an after-op run on CPU
// concurrently with the next tile's CUTE dispatch.
//
// A layout: [DIM_M][DIM_K/2] FP4 packed, byte stride = stride_A_bytes (typically DIM_K/2)
// B layout: [DIM_N][DIM_K/2] FP4 packed, byte stride = stride_B_bytes (typically DIM_K/2)
// A_scale : [DIM_M][DIM_K/16] E4M3, byte stride = scale_stride_A_bytes
// B_scale : [DIM_N][DIM_K/16] E4M3, byte stride = scale_stride_B_bytes
// C layout: caller-specific; passed to after-op via `output` + `stride_C`.
//
// after_op is one of the FUSE_NVFP4_* codes; extra is op-specific.

static void matmul_cute_nvfp4(
    size_t DIM_M, size_t DIM_N, size_t DIM_K,
    const void *A_fp4, const void *B_fp4, void *C,
    void *element_wise_tensor,
    const void *A_scale, const void *B_scale,
    size_t stride_A_bytes, size_t stride_B_bytes,
    size_t scale_stride_A_bytes, size_t scale_stride_B_bytes,
    size_t stride_C_bytes,
    int after_op, void *extra)
{
    if (!(DIM_M % 64 == 0 && DIM_N % 64 == 0 && DIM_K % 64 == 0)) {
        printf("matmul_cute_nvfp4: dims not multiple of 64\n");
        exit(1);
    }

    after_op_fn fn = NULL;
    switch (after_op) {
        case FUSE_NVFP4_QNORM_PARTROPE_BF16:
            // Actual norm/rope done after all tiles land — this fuse writes bf16.
            fn = fuse_nvfp4_bf16_cvrt;
            break;
        case FUSE_NVFP4_BF16CVRT_T:
            fn = fuse_nvfp4_bf16_cvrt;      // same cvrt, transpose handled by caller layout
            break;
        case FUSE_NVFP4_TO_F32:
            fn = fuse_nvfp4_to_f32;
            break;
        case FUSE_NVFP4_RESADD:
            fn = fuse_nvfp4_resadd;
            break;
        default:
            printf("matmul_cute_nvfp4: unknown after_op %d\n", after_op);
            exit(1);
    }

    int tile_M = 64, tile_N = 64;
    int Tile_I = DIM_M / tile_M;
    int Tile_J = DIM_N / tile_N;

    int C_elem_size = (after_op == FUSE_NVFP4_QNORM_PARTROPE_BF16 || after_op == FUSE_NVFP4_BF16CVRT_T) ? 2 : 4;

    int Application_stride_C_tile = tile_N * 4;  // FP32 tile stride in TCM

    // First tile
    int i = 0, j = 0;
    const uint8_t *tile_A = (const uint8_t *)A_fp4;
    const uint8_t *tile_B = (const uint8_t *)B_fp4;
    const uint8_t *tile_As = (const uint8_t *)A_scale;
    const uint8_t *tile_Bs = (const uint8_t *)B_scale;
    void *tile_C = CUTE_result[CUTE_result_index];

    uint64_t task_pre = issue_cute_matmul_marco_inst_nvfp4(
        (uint64_t)tile_A, stride_A_bytes,
        (uint64_t)tile_B, stride_B_bytes,
        0, 0,
        (uint64_t)tile_C, Application_stride_C_tile,
        (uint64_t)tile_As, (uint64_t)tile_Bs,
        tile_M, tile_N, DIM_K,
        TaskTypeTensorZeroLoad, 0, 0);

    int pre_i = 0, pre_j = 0;

    for (i = 0; i < Tile_I; i++) {
        for (j = (i == 0 ? 1 : 0); j < Tile_J; j++) {
            CUTE_TASK_END(task_pre);
            int next_idx = next_result_idx(CUTE_result_index);

            tile_A  = (const uint8_t *)A_fp4  + i * tile_M * stride_A_bytes;
            tile_B  = (const uint8_t *)B_fp4  + j * tile_N * stride_B_bytes;
            tile_As = (const uint8_t *)A_scale + i * tile_M * scale_stride_A_bytes;
            tile_Bs = (const uint8_t *)B_scale + j * tile_N * scale_stride_B_bytes;
            tile_C  = CUTE_result[next_idx];

            task_pre = issue_cute_matmul_marco_inst_nvfp4(
                (uint64_t)tile_A, stride_A_bytes,
                (uint64_t)tile_B, stride_B_bytes,
                0, 0,
                (uint64_t)tile_C, Application_stride_C_tile,
                (uint64_t)tile_As, (uint64_t)tile_Bs,
                tile_M, tile_N, DIM_K,
                TaskTypeTensorZeroLoad, 0, 0);

            // Run after-op on the just-completed tile.
            void *out_addr = (uint8_t *)C + pre_i * tile_M * stride_C_bytes
                                          + pre_j * tile_N * C_elem_size;
            fn(CUTE_result[CUTE_result_index], out_addr, NULL, NULL,
               tile_M, tile_N, Application_stride_C_tile, stride_C_bytes, extra);

            CUTE_result_index = next_idx;
            pre_i = i;
            pre_j = j;
        }
    }
    CUTE_TASK_END(task_pre);
    {
        void *out_addr = (uint8_t *)C + pre_i * tile_M * stride_C_bytes
                                      + pre_j * tile_N * C_elem_size;
        fn(CUTE_result[CUTE_result_index], out_addr, NULL, NULL,
           tile_M, tile_N, Application_stride_C_tile, stride_C_bytes, extra);
    }
    (void)element_wise_tensor;
}

// matmul_cute_bf16: BF16 GEMM (score = Q @ K^T and attn = scores @ V paths).
// element_type = CUTEDataTypeBF16BF16F32 = 2.
static void matmul_cute_bf16(
    size_t DIM_M, size_t DIM_N, size_t DIM_K,
    const void *A_bf16, const void *B_bf16, void *C,
    size_t stride_A_bytes, size_t stride_B_bytes, size_t stride_C_bytes,
    int after_op, void *extra)
{
    if (!(DIM_M % 64 == 0 && DIM_N % 64 == 0 && DIM_K % 64 == 0)) {
        printf("matmul_cute_bf16: dims not multiple of 64\n");
        exit(1);
    }

    if (after_op == NO_ACTIVATION) {
        uint64_t tid = issue_cute_matmul_marco_inst(
            (uint64_t)A_bf16, stride_A_bytes,
            (uint64_t)B_bf16, stride_B_bytes,
            0, 0,
            (uint64_t)C, stride_C_bytes,
            DIM_M, DIM_N, DIM_K,
            CUTEDataTypeBF16BF16F32,
            TaskTypeTensorZeroLoad, 0, 0);
        CUTE_TASK_END(tid);
        return;
    }

    // FUSE_MASKED_SOFTMAX_BF16CVRT path — tile along M only, since dim_j equals DIM_N.
    int tile_M = 64;
    int Tile_I = DIM_M / tile_M;
    int Application_stride_C_tile = DIM_N * 4;

    const uint8_t *tile_A = (const uint8_t *)A_bf16;
    void *tile_C = CUTE_result[CUTE_result_index];
    uint64_t task_pre = issue_cute_matmul_marco_inst(
        (uint64_t)tile_A, stride_A_bytes,
        (uint64_t)B_bf16, stride_B_bytes,
        0, 0,
        (uint64_t)tile_C, Application_stride_C_tile,
        tile_M, DIM_N, DIM_K,
        CUTEDataTypeBF16BF16F32,
        TaskTypeTensorZeroLoad, 0, 0);

    int pre_i = 0;
    for (int i = 1; i < Tile_I; i++) {
        CUTE_TASK_END(task_pre);
        int next_idx = next_result_idx(CUTE_result_index);
        tile_A = (const uint8_t *)A_bf16 + i * tile_M * stride_A_bytes;
        tile_C = CUTE_result[next_idx];
        task_pre = issue_cute_matmul_marco_inst(
            (uint64_t)tile_A, stride_A_bytes,
            (uint64_t)B_bf16, stride_B_bytes,
            0, 0,
            (uint64_t)tile_C, Application_stride_C_tile,
            tile_M, DIM_N, DIM_K,
            CUTEDataTypeBF16BF16F32,
            TaskTypeTensorZeroLoad, 0, 0);
        void *out_addr = (uint8_t *)C + pre_i * tile_M * stride_C_bytes;
        fuse_masked_softmax_bf16(CUTE_result[CUTE_result_index], out_addr,
                                 NULL, NULL, tile_M, DIM_N,
                                 Application_stride_C_tile, stride_C_bytes, extra);
        CUTE_result_index = next_idx;
        pre_i = i;
    }
    CUTE_TASK_END(task_pre);
    {
        void *out_addr = (uint8_t *)C + pre_i * tile_M * stride_C_bytes;
        fuse_masked_softmax_bf16(CUTE_result[CUTE_result_index], out_addr,
                                 NULL, NULL, tile_M, DIM_N,
                                 Application_stride_C_tile, stride_C_bytes, extra);
    }
}

// -------- Causal mask fill: bitmask_ptr[i][j] = 1 iff i >= j --------
static void fill_causal_mask(void)
{
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int allowed = (j <= i);
            if (allowed) bitmask_ptr[i][j / 8] |= (int8_t)(1 << (j & 7));
        }
    }
}

// -------- Layer orchestrator --------
static void qwen_full_attn_block(void)
{
    uint64_t t_layer_start = mrdcycle();
    int rope_pos_base = 0;

    // 1. RMSnorm(x, attn_norm_w) → per-16-block E4M3 quant of x_norm
    //    Reuse identity as x_norm working buffer (in place is fine; we don't need the
    //    unnormed x for residual until step 11 which reads identity BEFORE this line's
    //    overwrite? no — identity IS the residual, we cannot overwrite. Use gate_f32 as scratch.
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm_input\n", SEQ_LEN, HIDDEN);
    uint64_t t0 = mrdcycle();
    rmsnorm_fp32(&identity[0][0], &gate_f32[0][0], attn_norm_w,
                 SEQ_LEN, HIDDEN, HIDDEN * 4, HIDDEN * 4);
    quant_nvfp4_block16(&gate_f32[0][0], &x_norm_fp4[0][0], &x_norm_scale[0][0],
                        SEQ_LEN, HIDDEN,
                        HIDDEN * 4, FP4_BYTES(HIDDEN), SCALE_BYTES(HIDDEN));
    printf("[WorkLoad-latency] RMSnorm+quant: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 2. Q projection: [SEQ, HIDDEN] × [HIDDEN, HIDDEN] → bf16 [SEQ, N_HEAD_Q, HEAD_DIM]
    printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]proj_q\n", SEQ_LEN, HIDDEN, HIDDEN);
    t0 = mrdcycle();
    matmul_cute_nvfp4(SEQ_LEN, HIDDEN, HIDDEN,
                      &x_norm_fp4[0][0], &proj_q_w_fp4[0][0], &q_buf_bf16[0][0][0],
                      NULL,
                      &x_norm_scale[0][0], &proj_q_ws[0][0],
                      FP4_BYTES(HIDDEN), FP4_BYTES(HIDDEN),
                      SCALE_BYTES(HIDDEN), SCALE_BYTES(HIDDEN),
                      HIDDEN * 2,   // bf16 stride: N_HEAD_Q*HEAD_DIM*2 = 4096*2 = HIDDEN*2
                      FUSE_NVFP4_QNORM_PARTROPE_BF16, NULL);
    printf("[WorkLoad-latency] q_proj: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 3. K projection: [SEQ, HIDDEN] × [N_HEAD_KV*HEAD_DIM, HIDDEN] → bf16 [SEQ, N_HEAD_KV, HEAD_DIM]
    printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]proj_k\n", SEQ_LEN, N_HEAD_KV*HEAD_DIM, HIDDEN);
    t0 = mrdcycle();
    matmul_cute_nvfp4(SEQ_LEN, N_HEAD_KV * HEAD_DIM, HIDDEN,
                      &x_norm_fp4[0][0], &proj_k_w_fp4[0][0], &k_buf_bf16[0][0][0],
                      NULL,
                      &x_norm_scale[0][0], &proj_k_ws[0][0],
                      FP4_BYTES(HIDDEN), FP4_BYTES(HIDDEN),
                      SCALE_BYTES(HIDDEN), SCALE_BYTES(HIDDEN),
                      N_HEAD_KV * HEAD_DIM * 2,
                      FUSE_NVFP4_QNORM_PARTROPE_BF16, NULL);
    printf("[WorkLoad-latency] k_proj: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 4. V projection: same shape as K, no norm/rope, bf16 layout ready for score→attn matmul
    printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]proj_v\n", SEQ_LEN, N_HEAD_KV*HEAD_DIM, HIDDEN);
    t0 = mrdcycle();
    matmul_cute_nvfp4(SEQ_LEN, N_HEAD_KV * HEAD_DIM, HIDDEN,
                      &x_norm_fp4[0][0], &proj_v_w_fp4[0][0], &v_buf_bf16[0][0][0],
                      NULL,
                      &x_norm_scale[0][0], &proj_v_ws[0][0],
                      FP4_BYTES(HIDDEN), FP4_BYTES(HIDDEN),
                      SCALE_BYTES(HIDDEN), SCALE_BYTES(HIDDEN),
                      N_HEAD_KV * HEAD_DIM * 2,
                      FUSE_NVFP4_BF16CVRT_T, NULL);
    printf("[WorkLoad-latency] v_proj: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 5. gate projection: [SEQ, HIDDEN] → gate_f32 (FP32)
    printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]proj_gate\n", SEQ_LEN, HIDDEN, HIDDEN);
    t0 = mrdcycle();
    matmul_cute_nvfp4(SEQ_LEN, HIDDEN, HIDDEN,
                      &x_norm_fp4[0][0], &gate_w_fp4[0][0], &gate_f32[0][0],
                      NULL,
                      &x_norm_scale[0][0], &gate_ws[0][0],
                      FP4_BYTES(HIDDEN), FP4_BYTES(HIDDEN),
                      SCALE_BYTES(HIDDEN), SCALE_BYTES(HIDDEN),
                      HIDDEN * 4,
                      FUSE_NVFP4_TO_F32, NULL);
    printf("[WorkLoad-latency] gate_proj: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 6. Q/K per-head RMSnorm + partial RoPE (first ROPE_DIM of HEAD_DIM). CPU-only pass.
    t0 = mrdcycle();
    qk_norm_and_partial_rope_bf16(&q_buf_bf16[0][0][0], q_norm_w, SEQ_LEN, N_HEAD_Q, rope_pos_base);
    qk_norm_and_partial_rope_bf16(&k_buf_bf16[0][0][0], k_norm_w, SEQ_LEN, N_HEAD_KV, rope_pos_base);
    printf("[WorkLoad-latency] qk_norm+partrope: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 7. Score: per Q head, scores[h] = Q[h] @ K[h/gqa]^T with layout [SEQ, HEAD_DIM] * [SEQ, HEAD_DIM]^T
    //    matmul_cute_bf16 expects [M][K] × [N][K] (B pre-transposed in strided sense), which matches
    //    our layout: for head i, A = q_buf_bf16[:, i, :], B = k_buf_bf16[:, i/4, :].
    t0 = mrdcycle();
    int pos_softmax = 0;
    for (int h = 0; h < N_HEAD_Q; h++) {
        void *A = &q_buf_bf16[0][h][0];
        void *B = &k_buf_bf16[0][h / (N_HEAD_Q / N_HEAD_KV)][0];
        void *C = &scores_bf16[h][0][0];
        printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]score_h%d\n", SEQ_LEN, SEQ_LEN, HEAD_DIM, h);
        matmul_cute_bf16(SEQ_LEN, SEQ_LEN, HEAD_DIM,
                         A, B, C,
                         N_HEAD_Q  * HEAD_DIM * 2,
                         N_HEAD_KV * HEAD_DIM * 2,
                         SEQ_LEN * 2,
                         FUSE_MASKED_SOFTMAX_BF16CVRT, &pos_softmax);
    }
    printf("[WorkLoad-latency] all_score+softmax: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 8. Attention: per Q head, attn[h] = scores[h] @ V[h/gqa]  (bf16 × bf16 → fp32).
    //    Result laid out as [SEQ][N_HEAD_Q * HEAD_DIM] in attn_f32.
    t0 = mrdcycle();
    for (int h = 0; h < N_HEAD_Q; h++) {
        void *A = &scores_bf16[h][0][0];
        void *B = &v_buf_bf16[0][h / (N_HEAD_Q / N_HEAD_KV)][0];
        void *C = (uint8_t *)&attn_f32[0][0] + h * HEAD_DIM * 4;
        printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]attn_h%d\n", SEQ_LEN, HEAD_DIM, SEQ_LEN, h);
        matmul_cute_bf16(SEQ_LEN, HEAD_DIM, SEQ_LEN,
                         A, B, C,
                         SEQ_LEN * 2,
                         N_HEAD_KV * HEAD_DIM * 2,
                         HIDDEN * 4,
                         NO_ACTIVATION, NULL);
    }
    printf("[WorkLoad-latency] all_attn: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 9. sigmoid(gate_f32) * attn_f32  (attn_output_gate) → attn_f32 in-place
    t0 = mrdcycle();
    sigmoid_hadamard_f32(&attn_f32[0][0], &gate_f32[0][0], SEQ_LEN, HIDDEN);
    printf("[WorkLoad-latency] sigmoid_hadamard: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 10. Per-16-block NVFP4 quant of gated attn for o_proj input
    t0 = mrdcycle();
    quant_nvfp4_block16(&attn_f32[0][0], &attn_g_fp4[0][0], &attn_g_scale[0][0],
                        SEQ_LEN, HIDDEN,
                        HIDDEN * 4, FP4_BYTES(HIDDEN), SCALE_BYTES(HIDDEN));
    printf("[WorkLoad-latency] gated_attn_quant: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    // 11. o_proj: [SEQ, HIDDEN] × [HIDDEN, HIDDEN] → FP32 + residual add (identity in place)
    printf("[WorkLoad-(%5d,%5d,%5d)LayerWise]proj_o\n", SEQ_LEN, HIDDEN, HIDDEN);
    t0 = mrdcycle();
    matmul_cute_nvfp4(SEQ_LEN, HIDDEN, HIDDEN,
                      &attn_g_fp4[0][0], &proj_o_w_fp4[0][0], &identity[0][0],
                      NULL,
                      &attn_g_scale[0][0], &proj_o_ws[0][0],
                      FP4_BYTES(HIDDEN), FP4_BYTES(HIDDEN),
                      SCALE_BYTES(HIDDEN), SCALE_BYTES(HIDDEN),
                      HIDDEN * 4,
                      FUSE_NVFP4_RESADD, NULL);
    printf("[WorkLoad-latency] o_proj+resadd: %lu cycles\n", (unsigned long)(mrdcycle() - t0));

    uint64_t t_layer_end = mrdcycle();
    printf("[WorkLoad-total] qwen_full_attn_layer: %lu cycles\n",
           (unsigned long)(t_layer_end - t_layer_start));
}

int main(void)
{
    printf("[qwen_full_attn_nvfp4] start\n");
    printf("[qwen_full_attn_nvfp4] SEQ_LEN=%d HIDDEN=%d N_HEAD_Q=%d N_HEAD_KV=%d HEAD_DIM=%d ROPE_DIM=%d\n",
           SEQ_LEN, HIDDEN, N_HEAD_Q, N_HEAD_KV, HEAD_DIM, ROPE_DIM);
    fill_causal_mask();
    qwen_full_attn_block();
    printf("[qwen_full_attn_nvfp4] done\n");
    return 0;
}
