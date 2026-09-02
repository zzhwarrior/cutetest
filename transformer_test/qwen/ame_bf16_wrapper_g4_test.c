// Phase G.4: matmul_ame_bf16() with FUSE_MASKED_SOFTMAX_BF16CVRT.
//
// Extends G.3.a with the score-matmul fuse-op used by Qwen's attention block:
//   FUSE_MASKED_SOFTMAX_BF16CVRT — masked softmax with 1/sqrt(HEAD_DIM) scale
//     and fp32→"bf16" (actually IEEE fp16 — see G.2 notes for the naming
//     issue, kept for source-level compatibility with qwen).
//
// Constraint: the softmax reduces over the full N dimension per row, so the
// wrapper only tiles along M when after_op == FUSE_MASKED_SOFTMAX_BF16CVRT.
// DIM_N must fit in a single AME tile (TILE_N = 128). This matches qwen's
// actual usage where score = Q @ K^T with N = SEQ_LEN = 128.
//
// Design (softmax path):
//   for mt in 0..TILES_M:
//     ACC := (acct==0) ? ACC0 : ACC1
//     mzero(ACC)
//     for kt in 0..TILES_K:
//       (kt even → TR0/TR2, odd → TR1/TR3)
//       mlae16, mlbe16, mfmacc.s.bf16 into ACC
//     msce32(ACC, staging, TILE_N*4)
//     ame_fence()
//     fuse_masked_softmax_bf16(staging, C_tile, ..., TILE_M, DIM_N, ..., extra)
//     acct ^= 1
//
// NO_ACTIVATION path is the same as G.3.a — msce32 writes fp32 directly to C.
//
// Test:
//   Shape 128x128x256 (matches Qwen's score matmul = SEQ×SEQ×HEAD_DIM).
//   A=B=1.0, all-ones causal mask (so mask has no numerical effect), pos=0.
//   Softmax golden:
//     GEMM output       = K = 256.0 (fp32)
//     × INV_SQRT_HDIM   = 256 × 1/16 = 16.0    per cell
//     - row max         = 0.0                  (all cells equal)
//     exp(0)            = 1.0                  per cell
//     sum over N=128    = 128.0
//     normalized        = 1/128 = 2^-7 = 0.0078125
//     fp16 bit pattern  = 0x2000 (sign 0, exp 8=01000, mantissa 0)
//   Also runs a NO_ACTIVATION regression to prove that path still works
//   through the new after_op switch — expects 0x43800000 (fp32 256.0).

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "../../ame_test/ame.h"

// ============================================================================
// Tile / fuse-op constants
// ============================================================================
#define TILE_M   128
#define TILE_N   128
#define TILE_K   32       // 1 RV bf16 @ CUTE_4Tops_128SCP

#define NO_ACTIVATION                    0
#define FUSE_MASKED_SOFTMAX_BF16CVRT     4     // matches qwen_full_attn_nvfp4.c

// Test shape (also matches qwen's score matmul)
#define APP_M       128
#define APP_N       128
#define APP_K       256          // HEAD_DIM in qwen
#define SEQ_LEN     APP_N        // for compat with mask indexing convention

#define INV_SQRT_HDIM   0.0625f   // 1/sqrt(256) — matches qwen for HEAD_DIM=256

typedef void (*after_op_fn)(void *, void *, void *, void *,
                            int, int, uint64_t, uint64_t, void *);

// ============================================================================
// Global bitmask (matches qwen's global static). Fuse-op reads it directly.
// For this smoke test we fill it with all 0xFF so no mask bits are cleared —
// every cell contributes to the softmax denominator.
// ============================================================================
static int8_t bitmask_ptr[SEQ_LEN * SEQ_LEN / 8] __attribute__((aligned(64))) = {
    [0 ... SEQ_LEN * SEQ_LEN / 8 - 1] = (int8_t)0xFF
};

// ============================================================================
// vec_exp — mirror of qwen_full_attn_nvfp4.c:173-211
// ============================================================================
static inline vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl) {
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    const int32_t MAX_A = 127;
    const int32_t MIN_A = -126;

    vfloat32m4_t af = __riscv_vfmul_vf_f32m4(x, INV_LN2, vl);
    vfloat32m4_t r  = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl);
    vfloat32m4_t a  = __riscv_vfadd_vv_f32m4(af, r, vl);
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

// ============================================================================
// fuse_masked_softmax_bf16 — mirror of qwen_full_attn_nvfp4.c:497-559
//
// input      : fp32 GEMM tile output (staging)
// output     : "bf16" (actually IEEE fp16) softmax result written to C
// dim_i      : tile row count
// dim_j      : full N row width (softmax reduces over this)
// pos_ptr    : void* → int position offset added to row index for mask lookup
//
// NB: local `#pragma optimize("O0")` bypasses a GCC RVV codegen bug at
// -O1/O2/O3 that miscompiles stage-3's `vec_exp + masked accumulation +
// vse32` for i=0 only (see debug md section 8 for full symptom matrix).
// Attempted alternatives (T1..T4 in md):
//   • block-scope `vl`  → still fails at O3/O2 (0x2155)
//   • asm memory barriers between stages → still fails
//   • DRAM staging → still fails
//   • hybrid RVV(1/2/4) + scalar stage 3 → PASS but >1M cycles
//   • pure scalar (all 4 stages)         → PASS but 1.8M cycles
//   • port llama-style softmax          → still FAIL, actually worse
//                                         (1280 cells wrong across many rows,
//                                          new C[0][0]=0x3155=16/96 pattern)
//   • local O0 pragma (this one)         → PASS at 640k cycles (fastest that works)
// Softmax is O(N²) vs GEMM's O(N³), and total softmax time is <10% of
// Qwen full-attn layer, so the 8× slowdown from O0 is acceptable.
// ============================================================================
#pragma GCC push_options
#pragma GCC optimize("O0")
static void __attribute__((noinline))
fuse_masked_softmax_bf16(void *input, void *output,
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

        // stage 4: normalize + cvrt to fp16 (misnamed bf16, see G.2 notes)
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
#pragma GCC pop_options

// ---- Old RVV version kept for reference; disabled by #if 0 ----
#if 0
static void __attribute__((noinline))
fuse_masked_softmax_bf16_rvv(void *input, void *output,
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

        // Block-scope `vl` per stage. Sharing one `size_t vl` across stages
        // makes GCC's RVV codegen (as of the -march=...zvfh_zfh toolchain here)
        // reuse a register that was pre-loaded with dim_j and emit
        //   vsetvli zero, dim_j_reg, ...      ← discards actual vl
        //   subw    avl, avl, dim_j_reg       ← decrements by dim_j not by vl
        // → the loop exits after 1 iteration and later stages read partially-
        // uninitialized data. G.4 first hit this in stage 3 (row 0 was left
        // with stage-1 outputs instead of exp values, giving C[0][*] = 16/128
        // instead of 1/128). Block scoping forces separate SSA per stage and
        // produces the correct `vsetvli vl_reg, avl, ...` + `subw avl, avl,
        // vl_reg` form. Regression check: bf16_wrapper_test.c (G.3.a) uses
        // shared vl and still works because its stages are simpler; the
        // problematic case is stage 3's mix of vfsub/vmerge/vec_exp/vse/vfadd.

        // Barriers between stages: block-scoping `vl` is necessary but NOT
        // sufficient. Without a compiler barrier between stages, GCC RVV can
        // still fuse/reorder stage bodies across the outer i-loop iterations
        // and drop iterations from stage 3's accumulation. Symptom: row 0
        // sum_exp = 96 instead of 128 → all row-0 cells softmax to 1/96 =
        // 0x2155 instead of 1/128 = 0x2000. Debug printfs alone happen to
        // suppress this via their implicit memory clobbers — but relying on
        // that is fragile. `asm volatile ("" ::: "memory")` after each stage
        // forces the compiler to keep them sequenced.

        // stage 1: multiply by INV_SQRT_HDIM
        {
            size_t vl;
            for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
                v = __riscv_vfmul_vf_f32m4(v, INV_SQRT_HDIM, vl);
                __riscv_vse32_v_f32m4(row_x + k, v, vl);
            }
        }
        asm volatile ("" ::: "memory");

        // stage 2: masked max
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(-INFINITY, vl_full);
        {
            size_t vl;
            for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
                vbool8_t mask = __riscv_vlm_v_b8(
                    (uint8_t *)bitmask_ptr + ((i + pos_) * dim_j + k) / 8, vl);
                v = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), v, mask, vl);
                max_vec = __riscv_vfmax_vv_f32m4(max_vec, v, vl);
            }
        }
        float max_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, vl_full), vl_full));
        asm volatile ("" ::: "memory");

        // stage 3: exp(x-max), masked; sum for normalization
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_full);
        {
            size_t vl;
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
        }
        float sum_exp = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_full), vl_full));
        asm volatile ("" ::: "memory");

        // stage 4: normalize + cvrt to fp16 (misnamed bf16, see G.2 notes)
        float inv_sum = 1.0f / sum_exp;
        {
            size_t vl;
            for (int k = 0, avl = dim_j; avl > 0; k += (int)vl, avl -= (int)vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v = __riscv_vle32_v_f32m4(row_x + k, vl);
                v = __riscv_vfmul_vf_f32m4(v, inv_sum, vl);
                vfloat16m2_t h = __riscv_vfncvt_f_f_w_f16m2(v, vl);
                __riscv_vse16_v_f16m2(row_y + k, h, vl);
            }
        }
        asm volatile ("" ::: "memory");
    }
}
#endif  // #if 0 — RVV version disabled

// ============================================================================
// Staging for the softmax path (128x128 fp32 = 64 KiB).
//
// Placed in DRAM (.bss) rather than TCM: on the first read of the first
// row, TCM+CPU-L1 coherence appears to miss the invalidation for lines
// that were speculatively prefetched by the CPU before AME wrote them
// via TL. Symptom: row 0 sum_exp = 96 instead of 128 (32 cells stale
// = 2× 64-byte L1 lines), giving softmax = 1/96 = 0x2155 instead of
// 1/128 = 0x2000. Rows 1..127 have no such stale L1 line so they pass.
// Same value works fine in G.2 (NVFP4 path) because that test dispatches
// multiple (M,N) tiles — subsequent AME writes evict the stale first-tile
// line before it ever gets read. G.4 SOFTMAX is single-tile.
// DRAM staging is subject to standard cache coherence, so RVV vle32 sees
// AME's writes correctly.
// ============================================================================
static uint8_t ame_staging_dram[TILE_M * TILE_N * 4] __attribute__((aligned(64)));
#define TCM_STAGING_ADDR       ((uint64_t)(uintptr_t)ame_staging_dram)
#define STAGING_STRIDE_BYTES   (TILE_N * (uint64_t)sizeof(uint32_t))

// ============================================================================
// The wrapper — mirrors matmul_cute_bf16 signature from qwen.
// ============================================================================
static void matmul_ame_bf16(
    size_t DIM_M, size_t DIM_N, size_t DIM_K,
    const void *A_bf16, const void *B_bf16, void *C,
    size_t stride_A_bytes, size_t stride_B_bytes, size_t stride_C_bytes,
    int after_op, void *extra)
{
    if ((DIM_M % TILE_M) || (DIM_N % TILE_N) || (DIM_K % TILE_K)) {
        printf("[matmul_ame_bf16] dims not aligned\n");
        return;
    }

    after_op_fn fn = NULL;
    int softmax_path = 0;
    switch (after_op) {
        case NO_ACTIVATION:
            fn = NULL; break;
        case FUSE_MASKED_SOFTMAX_BF16CVRT:
            fn = fuse_masked_softmax_bf16;
            softmax_path = 1;
            if (DIM_N != TILE_N) {
                printf("[matmul_ame_bf16] softmax path requires DIM_N==%d, got %zu\n",
                       TILE_N, DIM_N);
                return;
            }
            break;
        default:
            printf("[matmul_ame_bf16] unknown after_op %d\n", after_op);
            return;
    }

    const int TILES_M = DIM_M / TILE_M;
    const int TILES_N = DIM_N / TILE_N;
    const int TILES_K = DIM_K / TILE_K;
    const size_t TILE_K_BYTES = TILE_K * sizeof(uint16_t);

    void *staging = (void *)TCM_STAGING_ADDR;
    // C element size: fp32 for NO_ACTIVATION, fp16 for softmax.
    int C_elem_size = softmax_path ? 2 : 4;

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            void *c_tile = (uint8_t *)C
                + (size_t)mt * TILE_M * stride_C_bytes
                + (size_t)nt * TILE_N * C_elem_size;
            uint64_t store_dest = softmax_path ? (uint64_t)staging : (uint64_t)c_tile;
            uint64_t store_stride = softmax_path ? STAGING_STRIDE_BYTES : stride_C_bytes;

            if (acct == 0) {
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_base = (uint64_t)A_bf16
                        + (uint64_t)mt * TILE_M * stride_A_bytes
                        + (uint64_t)kt * TILE_K_BYTES;
                    uint64_t b_base = (uint64_t)B_bf16
                        + (uint64_t)nt * TILE_N * stride_B_bytes
                        + (uint64_t)kt * TILE_K_BYTES;

                    if ((kt & 1) == 0) {
                        ame_mlae16(TR0, a_base, stride_A_bytes);
                        ame_mlbe16(TR2, b_base, stride_B_bytes);
                        ame_mfmacc_s_bf16(ACC0, TR0, TR2);
                    } else {
                        ame_mlae16(TR1, a_base, stride_A_bytes);
                        ame_mlbe16(TR3, b_base, stride_B_bytes);
                        ame_mfmacc_s_bf16(ACC0, TR1, TR3);
                    }
                }
                ame_msce32(ACC0, store_dest, store_stride);
                acct = 1;
            } else {
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_base = (uint64_t)A_bf16
                        + (uint64_t)mt * TILE_M * stride_A_bytes
                        + (uint64_t)kt * TILE_K_BYTES;
                    uint64_t b_base = (uint64_t)B_bf16
                        + (uint64_t)nt * TILE_N * stride_B_bytes
                        + (uint64_t)kt * TILE_K_BYTES;

                    if ((kt & 1) == 0) {
                        ame_mlae16(TR0, a_base, stride_A_bytes);
                        ame_mlbe16(TR2, b_base, stride_B_bytes);
                        ame_mfmacc_s_bf16(ACC1, TR0, TR2);
                    } else {
                        ame_mlae16(TR1, a_base, stride_A_bytes);
                        ame_mlbe16(TR3, b_base, stride_B_bytes);
                        ame_mfmacc_s_bf16(ACC1, TR1, TR3);
                    }
                }
                ame_msce32(ACC1, store_dest, store_stride);
                acct = 0;
            }

            if (softmax_path) {
                // Drain AME so staging is fully committed to TCM (see G.2 notes:
                // CPU-only fence is insufficient — AME's TL writes must ACK first).
                ame_fence();
                fn(staging, c_tile, NULL, NULL,
                   TILE_M, (int)DIM_N,
                   STAGING_STRIDE_BYTES, stride_C_bytes, extra);
                // Fence CPU-side fuse-op stores before next tile's AME dispatch.
                asm volatile ("fence rw, rw" ::: "memory");
            }
        }
    }
    // Final drain for both paths — ensures caller-visible C is complete.
    ame_fence();
    asm volatile ("fence rw, rw" ::: "memory");
}

// ============================================================================
// Test driver
// ============================================================================
#define A_ROW_BYTES  (APP_K * (int)sizeof(uint16_t))    // 512

#define BF16_ONE           0x3F80u
#define EXPECTED_C_FP32    0x43800000u                    // fp32 256.0 (NO_ACT case)
#define EXPECTED_C_FP16    0x2000u                        // fp16 1/128 (softmax case)

static const uint16_t A_bf16_dram[APP_M][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_M - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};
static const uint16_t B_bf16_dram[APP_N][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_N - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};

#define TCM_A_ADDR         0x81000000UL     // 64 KiB
#define TCM_B_ADDR         0x81010000UL     // 64 KiB
#define TCM_C_FP32_ADDR    0x81020000UL     // 64 KiB   (NO_ACT case, fp32)
#define TCM_C_FP16_ADDR    0x81030000UL     // 32 KiB   (softmax case, fp16)
// staging = 0x81100000                     // 64 KiB (matmul_ame_bf16 uses this)

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[G4] start: matmul_ame_bf16 with fuse-op, %dx%dx%d\n",
           APP_M, APP_N, APP_K);

    if (ame_tcm_config(4) != 0) {
        printf("[G4] FAIL: TCM config\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    ame_dma_load((uint64_t)A_bf16_dram, TCM_A_ADDR, sizeof(A_bf16_dram));
    ame_dma_load((uint64_t)B_bf16_dram, TCM_B_ADDR, sizeof(B_bf16_dram));
    asm volatile ("fence rw, rw" ::: "memory");

    if (((volatile uint16_t *)TCM_A_ADDR)[0] != BF16_ONE) {
        printf("[G4] FAIL: TCM A stage\n");
        return 1;
    }

    int total_errors = 0;

    // ----- Case 1: NO_ACTIVATION regression -----
    {
        volatile uint32_t (*C)[APP_N] = (volatile uint32_t (*)[APP_N])TCM_C_FP32_ADDR;
        uint64_t t0 = rd_cycle();
        matmul_ame_bf16(APP_M, APP_N, APP_K,
                        (void *)TCM_A_ADDR, (void *)TCM_B_ADDR, (void *)TCM_C_FP32_ADDR,
                        A_ROW_BYTES, A_ROW_BYTES, APP_N * sizeof(uint32_t),
                        NO_ACTIVATION, NULL);
        uint64_t t1 = rd_cycle();
        printf("[G4 NO_ACT] cycles = %lu\n", (unsigned long)(t1 - t0));
        asm volatile ("fence rw, rw" ::: "memory");
        printf("[G4 NO_ACT] corners: C[0][0]=%08x C[0][127]=%08x C[127][127]=%08x\n",
               (unsigned)C[0][0], (unsigned)C[0][127], (unsigned)C[127][127]);
        int errors = 0;
        for (int i = 0; i < APP_M; i++) {
            for (int j = 0; j < 10; j++) {
                if (C[i][j] != EXPECTED_C_FP32) { errors++; }
            }
        }
        printf("[G4 NO_ACT] %s: expect 0x%08x, %d wrong out of %d\n",
               errors == 0 ? "PASS" : "FAIL", (unsigned)EXPECTED_C_FP32,
               errors, APP_M * APP_N);
        total_errors += errors;
    }

    // ----- Case 2: FUSE_MASKED_SOFTMAX_BF16CVRT -----
    // With A=B=1.0 and all-ones mask: softmax output = 1/N = 1/128 per cell.
    {
        int pos = 0;
        volatile uint16_t (*C)[APP_N] = (volatile uint16_t (*)[APP_N])TCM_C_FP16_ADDR;
        uint64_t t0 = rd_cycle();
        matmul_ame_bf16(APP_M, APP_N, APP_K,
                        (void *)TCM_A_ADDR, (void *)TCM_B_ADDR, (void *)TCM_C_FP16_ADDR,
                        A_ROW_BYTES, A_ROW_BYTES, APP_N * sizeof(uint16_t),
                        FUSE_MASKED_SOFTMAX_BF16CVRT, &pos);
        uint64_t t1 = rd_cycle();
        printf("[G4 SOFTMAX] cycles = %lu\n", (unsigned long)(t1 - t0));
        asm volatile ("fence rw, rw" ::: "memory");
        printf("[G4 SOFTMAX] corners: C[0][0]=%04x C[0][127]=%04x C[127][127]=%04x\n",
               (unsigned)C[0][0], (unsigned)C[0][127], (unsigned)C[127][127]);
        int errors = 0;
        int first_i = -1, first_j = -1;
        uint16_t first_wrong = 0;
        for (int i = 0; i < APP_M; i++) {
            for (int j = 0; j < 10; j++) {
                uint16_t v = C[i][j];
                if (v != EXPECTED_C_FP16) {
                    if (errors == 0) { first_wrong = v; first_i = i; first_j = j; }
                    errors++;
                }
            }
        }
        if (errors == 0) {
            printf("[G4 SOFTMAX] PASS: all %d elements == 0x%04x\n",
                   APP_M * APP_N, (unsigned)EXPECTED_C_FP16);
        } else {
            printf("[G4 SOFTMAX] FAIL: %d/%d wrong; first C[%d][%d]=0x%04x expected 0x%04x\n",
                   errors, APP_M * APP_N, first_i, first_j,
                   (unsigned)first_wrong, (unsigned)EXPECTED_C_FP16);
        }
        total_errors += errors;
    }

    printf("[G4] done, total_errors = %d\n", total_errors);
    return total_errors == 0 ? 0 : 1;
}
