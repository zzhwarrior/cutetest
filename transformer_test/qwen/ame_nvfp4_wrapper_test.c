// Phase G.1: matmul_ame_nvfp4() wrapper — standalone driver test.
//
// This is the reusable building block that Phase G will call from
// qwen_full_attn_nvfp4_ame.c. Signature mirrors matmul_cute_nvfp4() from
// qwen_full_attn_nvfp4.c so the outer layer code (rmsnorm/rope/softmax) needs
// zero changes — only the matmul kernel swap.
//
// This G.1 test does NOT wire in fuse-ops yet (that's G.2). Raw NVFP4 GEMM
// only: A * B → C fp32, no after-op.
//
// -----------------------------------------------------------------------------
// Wrapper contract (documented for future Qwen integration)
// -----------------------------------------------------------------------------
//   A_fp4    : uint8_t [DIM_M][DIM_K/2]  packed FP4, row_stride = stride_A_bytes
//   B_fp4    : uint8_t [DIM_N][DIM_K/2]  packed FP4, row_stride = stride_B_bytes
//   C_fp32   : uint32_t [DIM_M][DIM_N]   fp32 output, row_stride = stride_C_bytes
//   A_scale  : uint8_t [TILES_K][DIM_M][TILE_K/16]  TILE-MAJOR E4M3 scales
//   B_scale  : uint8_t [TILES_K][DIM_N][TILE_K/16]  TILE-MAJOR E4M3 scales
//   DIM_{M,N,K} : must be multiples of TILE_{M,N,K} = 128
//
// The tile-major scale layout is a HARDWARE requirement (AME ScaleLoader reads
// TILE_M × SCALE_ROW bytes contiguously with no row-stride awareness). Qwen's
// quant_nvfp4_block16 currently emits row-major [M][K/16] — Phase G.5 must
// either patch the quantizer or add a pre-shuffle step before calling this
// wrapper.
//
// -----------------------------------------------------------------------------
// Test setup
// -----------------------------------------------------------------------------
// Shape: M=N=K=256 (2x2x2 tile grid — same as e256_full ping-pong version).
// A=B=1.0 (FP4 byte 0x22), scale=1.0 (E4M3 byte 0x38). Expected each
// C[i][j] = Σ_{k=0..255} 1.0*1.0 = 256.0 = 0x43800000. Success proves the
// wrapper produces identical output to e256_full's inline tile loop.

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define TILE_M       128
#define TILE_N       128
#define TILE_K       128                     // 1 RV @ CUTE_4Tops_128SCP
#define NVFP4_BLOCK  16
#define SCALE_ROW_BYTES  (TILE_K / NVFP4_BLOCK)   // 8

// -----------------------------------------------------------------------------
// The wrapper. Static-inline so the compiler can specialize constants per
// call site if the outer code passes dims as literals; also removes indirect
// call overhead in the Qwen hot path.
// -----------------------------------------------------------------------------
static void matmul_ame_nvfp4(
    size_t DIM_M, size_t DIM_N, size_t DIM_K,
    const void *A_fp4, const void *B_fp4, void *C_fp32,
    const void *A_scale, const void *B_scale,
    size_t stride_A_bytes, size_t stride_B_bytes,
    size_t stride_C_bytes)
{
    if ((DIM_M % TILE_M) || (DIM_N % TILE_N) || (DIM_K % TILE_K)) {
        printf("[matmul_ame_nvfp4] dims not multiple of 128\n");
        return;
    }

    const int TILES_M = DIM_M / TILE_M;
    const int TILES_N = DIM_N / TILE_N;
    const int TILES_K = DIM_K / TILE_K;

    // Tile-major scale slab strides
    const size_t A_scale_tile_bytes = DIM_M * SCALE_ROW_BYTES;
    const size_t B_scale_tile_bytes = DIM_N * SCALE_ROW_BYTES;

    // AME CSRs (single config for the whole call — all tiles have the same
    // 128x128x1RV shape internally).
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    // ACC0/ACC1 ping-pong across (M,N) tiles. AME operand registers must be
    // compile-time constants, so ACC0 and ACC1 branches are open-coded.
    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)C_fp32
                + (uint64_t)mt * TILE_M * stride_C_bytes
                + (uint64_t)nt * TILE_N * sizeof(uint32_t);

            if (acct == 0) {
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_scale_base = (uint64_t)A_scale
                        + (uint64_t)kt * A_scale_tile_bytes
                        + (uint64_t)mt * TILE_M * SCALE_ROW_BYTES;
                    uint64_t b_scale_base = (uint64_t)B_scale
                        + (uint64_t)kt * B_scale_tile_bytes
                        + (uint64_t)nt * TILE_N * SCALE_ROW_BYTES;
                    ame_mset_scalea(a_scale_base);
                    ame_mset_scaleb(b_scale_base);

                    uint64_t a_base = (uint64_t)A_fp4
                        + (uint64_t)mt * TILE_M * stride_A_bytes
                        + (uint64_t)kt * (TILE_K / 2);
                    uint64_t b_base = (uint64_t)B_fp4
                        + (uint64_t)nt * TILE_N * stride_B_bytes
                        + (uint64_t)kt * (TILE_K / 2);

                    if ((kt & 1) == 0) {
                        ame_mlae4(TR0, a_base, stride_A_bytes);
                        ame_mlbe4(TR2, b_base, stride_B_bytes);
                        ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);
                    } else {
                        ame_mlae4(TR1, a_base, stride_A_bytes);
                        ame_mlbe4(TR3, b_base, stride_B_bytes);
                        ame_mfmacc_s_nvfp4(ACC0, TR1, TR3);
                    }
                }
                ame_msce32(ACC0, c_base, stride_C_bytes);
                acct = 1;
            } else {
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_scale_base = (uint64_t)A_scale
                        + (uint64_t)kt * A_scale_tile_bytes
                        + (uint64_t)mt * TILE_M * SCALE_ROW_BYTES;
                    uint64_t b_scale_base = (uint64_t)B_scale
                        + (uint64_t)kt * B_scale_tile_bytes
                        + (uint64_t)nt * TILE_N * SCALE_ROW_BYTES;
                    ame_mset_scalea(a_scale_base);
                    ame_mset_scaleb(b_scale_base);

                    uint64_t a_base = (uint64_t)A_fp4
                        + (uint64_t)mt * TILE_M * stride_A_bytes
                        + (uint64_t)kt * (TILE_K / 2);
                    uint64_t b_base = (uint64_t)B_fp4
                        + (uint64_t)nt * TILE_N * stride_B_bytes
                        + (uint64_t)kt * (TILE_K / 2);

                    if ((kt & 1) == 0) {
                        ame_mlae4(TR0, a_base, stride_A_bytes);
                        ame_mlbe4(TR2, b_base, stride_B_bytes);
                        ame_mfmacc_s_nvfp4(ACC1, TR0, TR2);
                    } else {
                        ame_mlae4(TR1, a_base, stride_A_bytes);
                        ame_mlbe4(TR3, b_base, stride_B_bytes);
                        ame_mfmacc_s_nvfp4(ACC1, TR1, TR3);
                    }
                }
                ame_msce32(ACC1, c_base, stride_C_bytes);
                acct = 0;
            }
        }
    }
    ame_fence();
}

// -----------------------------------------------------------------------------
// Test driver
// -----------------------------------------------------------------------------
#define APP_M       256
#define APP_N       256
#define APP_K       256
#define TILES_M     (APP_M / TILE_M)
#define TILES_N     (APP_N / TILE_N)
#define TILES_K     (APP_K / TILE_K)

#define A_ROW_BYTES         (APP_K / 2)
#define SCALE_TILE_BYTES    (APP_M * SCALE_ROW_BYTES)          // 2048 per K-tile
#define SCALE_TOTAL_BYTES   (TILES_K * SCALE_TILE_BYTES)       // 4096

#define FP4_ONE_BYTE     0x22u
#define E4M3_ONE_BYTE    0x38u
#define EXPECTED_C_BITS  0x43800000u

static const uint8_t A_fp4_dram[APP_M][A_ROW_BYTES] __attribute__((aligned(64))) = {
    [0 ... APP_M - 1] = { [0 ... A_ROW_BYTES - 1] = FP4_ONE_BYTE }
};
static const uint8_t B_fp4_dram[APP_N][A_ROW_BYTES] __attribute__((aligned(64))) = {
    [0 ... APP_N - 1] = { [0 ... A_ROW_BYTES - 1] = FP4_ONE_BYTE }
};
static const uint8_t A_scale_dram[SCALE_TOTAL_BYTES] __attribute__((aligned(64))) = {
    [0 ... SCALE_TOTAL_BYTES - 1] = E4M3_ONE_BYTE
};
static const uint8_t B_scale_dram[SCALE_TOTAL_BYTES] __attribute__((aligned(64))) = {
    [0 ... SCALE_TOTAL_BYTES - 1] = E4M3_ONE_BYTE
};

#define TCM_A_ADDR         0x81000000UL
#define TCM_B_ADDR         0x81008000UL
#define TCM_A_SCALE_ADDR   0x81010000UL
#define TCM_B_SCALE_ADDR   0x81011000UL
#define TCM_C_ADDR         0x81012000UL

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[G1] start: matmul_ame_nvfp4 wrapper test, %dx%dx%d\n",
           APP_M, APP_N, APP_K);

    if (ame_tcm_config(4) != 0) {
        printf("[G1] FAIL: TCM config\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    ame_dma_load((uint64_t)A_fp4_dram,   TCM_A_ADDR,       sizeof(A_fp4_dram));
    ame_dma_load((uint64_t)B_fp4_dram,   TCM_B_ADDR,       sizeof(B_fp4_dram));
    ame_dma_load((uint64_t)A_scale_dram, TCM_A_SCALE_ADDR, sizeof(A_scale_dram));
    ame_dma_load((uint64_t)B_scale_dram, TCM_B_SCALE_ADDR, sizeof(B_scale_dram));
    asm volatile ("fence rw, rw" ::: "memory");

    if (*(volatile uint8_t *)TCM_A_ADDR != FP4_ONE_BYTE) {
        printf("[G1] FAIL: TCM stage\n");
        return 1;
    }

    // Call the wrapper. All buffers in TCM.
    uint64_t t0 = rd_cycle();
    matmul_ame_nvfp4(
        APP_M, APP_N, APP_K,
        (void *)TCM_A_ADDR,       (void *)TCM_B_ADDR,       (void *)TCM_C_ADDR,
        (void *)TCM_A_SCALE_ADDR, (void *)TCM_B_SCALE_ADDR,
        /* stride_A_bytes */ A_ROW_BYTES,
        /* stride_B_bytes */ A_ROW_BYTES,
        /* stride_C_bytes */ APP_N * sizeof(uint32_t));
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t t1 = rd_cycle();
    printf("[G1] wrapper cycles = %lu\n", (unsigned long)(t1 - t0));

    volatile uint32_t (*C_tcm)[APP_N] = (volatile uint32_t (*)[APP_N])TCM_C_ADDR;

    printf("[G1] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)C_tcm[0][0],
           APP_N - 1, (unsigned)C_tcm[0][APP_N - 1],
           APP_M - 1, (unsigned)C_tcm[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)C_tcm[APP_M - 1][APP_N - 1]);
    printf("[G1] seams:   C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           TILE_N, (unsigned)C_tcm[0][TILE_N],
           TILE_M, (unsigned)C_tcm[TILE_M][0],
           TILE_M, TILE_N, (unsigned)C_tcm[TILE_M][TILE_N]);

    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < APP_M; i++) {
        for (int j = 0; j < 10; j++) {
            uint32_t v = C_tcm[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) {
                    first_wrong = v; first_i = i; first_j = j;
                }
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("[G1] PASS: all %d elements == 0x%08x\n",
               APP_M * APP_N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[G1] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, APP_M * APP_N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[G1] done\n");
    return errors == 0 ? 0 : 1;
}
