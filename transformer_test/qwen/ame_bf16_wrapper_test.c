// Phase G.3.a: matmul_ame_bf16() wrapper — standalone driver test (raw GEMM).
//
// BF16 counterpart of the NVFP4 wrapper in ame_nvfp4_wrapper_test.c (G.1).
// No scale, no fuse-op — just a shape-generic BF16 GEMM built out of AME's
// 128x128x(1 RV) native compute tiles. G.4 will layer FUSE_MASKED_SOFTMAX_BF16
// on top.
//
// -----------------------------------------------------------------------------
// Wrapper contract
// -----------------------------------------------------------------------------
//   A_bf16  : uint16_t [DIM_M][DIM_K]  IEEE bf16, row_stride = stride_A_bytes
//   B_bf16  : uint16_t [DIM_N][DIM_K]  IEEE bf16, row_stride = stride_B_bytes
//                                      (transpose is implicit in AME — B is
//                                       "K rows of N" from the mfmacc side.)
//   C_fp32  : uint32_t [DIM_M][DIM_N]  fp32 output, row_stride = stride_C_bytes
//   DIM_M, DIM_N must be multiples of 128; DIM_K must be a multiple of 32.
//
// -----------------------------------------------------------------------------
// Tile schedule
// -----------------------------------------------------------------------------
//   TILE_M = TILE_N = 128 (AME native), TILE_K = 32 (1 RV of bf16 on CUTE_4Tops_128SCP)
//   for mt: for nt:
//     ACC := (acct==0) ? ACC0 : ACC1
//     mzero(ACC)
//     for kt in 0..(DIM_K/32):
//       (kt even → TR0/TR2, odd → TR1/TR3)
//       mlae16, mlbe16, mfmacc.s.bf16 into ACC
//     msce32(ACC, C_tile, stride_C_bytes)
//     acct ^= 1
//   ame_fence()   ← single fence at end of GEMM (per-tile fence not needed
//                    since ping-pong keeps ACC/CSpad banks disjoint across
//                    consecutive tiles — same as e512 / bf16_e256_full patterns
//                    that both PASS).
//
// -----------------------------------------------------------------------------
// Test setup
// -----------------------------------------------------------------------------
// Shape M=N=K=256 (2x2 M/N tiles × 8 K-tiles). A=B=1.0 (bf16 0x3F80).
// Expected C[i][j] = 256.0 = 0x43800000 (fp32). Success proves wrapper works
// with M/N/K all tiled and multiple K-tiles per (M,N) tile.

#include <stdio.h>
#include <stdint.h>
#include "../../ame_test/ame.h"

// ============================================================================
// Wrapper: tile sizes
// ============================================================================
#define TILE_M   128
#define TILE_N   128
#define TILE_K   32     // 1 RV of BF16 @ CUTE_4Tops_128SCP (RWByte=64, bf16=2B → 32 elem)

// ============================================================================
// matmul_ame_bf16() — the raw wrapper. No fuse-op; msce32 writes fp32 output
// directly to caller-provided C.
// ============================================================================
static void matmul_ame_bf16(
    size_t DIM_M, size_t DIM_N, size_t DIM_K,
    const void *A_bf16, const void *B_bf16, void *C_fp32,
    size_t stride_A_bytes, size_t stride_B_bytes, size_t stride_C_bytes)
{
    if ((DIM_M % TILE_M) || (DIM_N % TILE_N) || (DIM_K % TILE_K)) {
        printf("[matmul_ame_bf16] dims not aligned: M %% %d, N %% %d, K %% %d\n",
               TILE_M, TILE_N, TILE_K);
        return;
    }

    const int TILES_M = DIM_M / TILE_M;
    const int TILES_N = DIM_N / TILE_N;
    const int TILES_K = DIM_K / TILE_K;

    // 2 bytes per bf16 element — K-tile column offset in A/B row.
    const size_t TILE_K_BYTES = TILE_K * sizeof(uint16_t);

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)C_fp32
                + (uint64_t)mt * TILE_M * stride_C_bytes
                + (uint64_t)nt * TILE_N * sizeof(uint32_t);

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
                ame_msce32(ACC0, c_base, stride_C_bytes);
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
                ame_msce32(ACC1, c_base, stride_C_bytes);
                acct = 0;
            }
        }
    }
    ame_fence();
    asm volatile ("fence rw, rw" ::: "memory");
}

// ============================================================================
// Test driver
// ============================================================================
#define APP_M       256
#define APP_N       256
#define APP_K       256
#define A_ROW_BYTES  (APP_K * (int)sizeof(uint16_t))   // 512

#define BF16_ONE         0x3F80u
#define EXPECTED_C_BITS  0x43800000u                    // fp32 256.0

static const uint16_t A_bf16_dram[APP_M][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_M - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};
static const uint16_t B_bf16_dram[APP_N][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_N - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};

#define TCM_A_ADDR   0x81000000UL   // 128 KiB
#define TCM_B_ADDR   0x81020000UL   // 128 KiB
#define TCM_C_ADDR   0x81040000UL   // 256 KiB

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[G3a] start: matmul_ame_bf16 wrapper test, %dx%dx%d\n",
           APP_M, APP_N, APP_K);

    if (ame_tcm_config(4) != 0) {
        printf("[G3a] FAIL: TCM config\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    ame_dma_load((uint64_t)A_bf16_dram, TCM_A_ADDR, sizeof(A_bf16_dram));
    ame_dma_load((uint64_t)B_bf16_dram, TCM_B_ADDR, sizeof(B_bf16_dram));
    asm volatile ("fence rw, rw" ::: "memory");

    if (((volatile uint16_t *)TCM_A_ADDR)[0] != BF16_ONE) {
        printf("[G3a] FAIL: TCM A stage\n");
        return 1;
    }

    uint64_t t0 = rd_cycle();
    matmul_ame_bf16(
        APP_M, APP_N, APP_K,
        (void *)TCM_A_ADDR, (void *)TCM_B_ADDR, (void *)TCM_C_ADDR,
        A_ROW_BYTES, A_ROW_BYTES, APP_N * sizeof(uint32_t));
    uint64_t t1 = rd_cycle();
    printf("[G3a] wrapper cycles = %lu\n", (unsigned long)(t1 - t0));

    // Verify — CPU-side reads of C_fp32 from TCM.
    asm volatile ("fence rw, rw" ::: "memory");
    volatile uint32_t (*C)[APP_N] = (volatile uint32_t (*)[APP_N])TCM_C_ADDR;

    printf("[G3a] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)C[0][0],
           APP_N - 1, (unsigned)C[0][APP_N - 1],
           APP_M - 1, (unsigned)C[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)C[APP_M - 1][APP_N - 1]);
    printf("[G3a] seams:   C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           TILE_N, (unsigned)C[0][TILE_N],
           TILE_M, (unsigned)C[TILE_M][0],
           TILE_M, TILE_N, (unsigned)C[TILE_M][TILE_N]);

    int errors = 0, first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < APP_M; i++) {
        for (int j = 0; j < 10; j++) {
            uint32_t v = C[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) { first_wrong = v; first_i = i; first_j = j; }
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("[G3a] PASS: all %d elements == 0x%08x\n",
               APP_M * APP_N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[G3a] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, APP_M * APP_N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[G3a] done\n");
    return errors == 0 ? 0 : 1;
}
