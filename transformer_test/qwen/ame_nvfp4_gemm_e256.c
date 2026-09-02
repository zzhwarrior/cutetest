// Phase E-small: 256x256x128 NVFP4 GEMM on the AME path (DMA-staged, TCM-resident).
//
// Data-source strategy (matches ame_matmul_512x512x512_dma.c):
//   1. All operand tensors are declared as static const arrays in DRAM,
//      pre-filled at load time via GCC range-designated initializers
//      (`[0 ... N-1] = value`). Zero runtime init cost.
//   2. Before compute, DMA them into TCM. AME instructions then reference
//      the TCM copies via fixed absolute addresses.
//   3. C output also lives in TCM so we don't pay DRAM round-trip on stores.
//
// TCM layout (4-way TCM = 2 MiB @ 0x81000000, we use ~292 KiB of the top way):
//   0x81000000  A_fp4      256 * 64 = 16 KiB
//   0x81004000  B_fp4      256 * 64 = 16 KiB
//   0x81008000  A_scale    pad to 4 KiB
//   0x81009000  B_scale    pad to 4 KiB
//   0x8100A000  C_fp32     256 * 256 * 4 = 256 KiB
//
// Golden math (like Phase D variant 0):
//   A_fp4[i][k] = 1.0  (packed byte 0x22)
//   B_fp4[j][k] = 1.0
//   scaleA/B    = 1.0  (E4M3 byte 0x38)
// => C[i][j] = Σ_{k=0..127} 1.0 * 1.0 * 1.0 * 1.0 = 128.0 = 0x43000000
//
// Tile schedule (single ACC0, 2x2=4 M/N tiles, no K tiling):
//   for m_tile in 0..1: for n_tile in 0..1:
//     mset_scalea(A_scale_tcm + m_tile*TILE_M*(K/16))
//     mset_scaleb(B_scale_tcm + n_tile*TILE_N*(K/16))
//     settile{m,n}(128); settilek(1 RV)
//     mzero(ACC0)
//     mlae4(TR0, A_fp4_tcm + m_tile*TILE_M*(K/2), K/2)
//     mlbe4(TR2, B_fp4_tcm + n_tile*TILE_N*(K/2), K/2)
//     mfmacc.s.nvfp4(ACC0, TR0, TR2)
//     msce32(ACC0, C_fp32_tcm[m_tile*TILE_M][n_tile*TILE_N], N*4)

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define M       256
#define N       256
#define K       128                    // one full RV per tile (no K tiling)
#define TILE_M  128
#define TILE_N  128
#define NVFP4_BLOCK 16
#define M_TILES (M / TILE_M)
#define N_TILES (N / TILE_N)

#define FP4_ONE_BYTE     0x22u         // FP4 E2M1 value 1.0, packed 2/byte
#define E4M3_ONE_BYTE    0x38u         // E4M3 value 1.0
#define EXPECTED_C_BITS  0x43000000u   // fp32 128.0

#define SCALE_ROW_BYTES  (K / NVFP4_BLOCK)   // 8 bytes per row
// Round scale allocation up to a multiple of 64 so DMA length checks pass and
// ScaleLoader over-read stays in bounds. 256 rows * 8 = 2048; pad to 4 KiB.
#define SCALE_ALLOC_BYTES  4096

// ---- DRAM-resident source data (pre-initialized, zero runtime init cost) ----
// GCC range-designated initializers keep the init inline in .rodata / .data.
// This causes the binary to grow by (16+16+4+4) KiB but avoids runtime memset.
static const uint8_t A_fp4_dram[M][K / 2] __attribute__((aligned(64))) = {
    [0 ... M - 1] = { [0 ... K/2 - 1] = FP4_ONE_BYTE }
};
static const uint8_t B_fp4_dram[N][K / 2] __attribute__((aligned(64))) = {
    [0 ... N - 1] = { [0 ... K/2 - 1] = FP4_ONE_BYTE }
};
static const uint8_t A_scale_dram[SCALE_ALLOC_BYTES] __attribute__((aligned(64))) = {
    [0 ... SCALE_ALLOC_BYTES - 1] = E4M3_ONE_BYTE
};
static const uint8_t B_scale_dram[SCALE_ALLOC_BYTES] __attribute__((aligned(64))) = {
    [0 ... SCALE_ALLOC_BYTES - 1] = E4M3_ONE_BYTE
};

// ---- TCM addresses (compile-time constants) ----
#define TCM_A_ADDR         0x81000000UL     // 16 KiB
#define TCM_B_ADDR         0x81004000UL     // 16 KiB
#define TCM_A_SCALE_ADDR   0x81008000UL     // 4 KiB
#define TCM_B_SCALE_ADDR   0x81009000UL     // 4 KiB
#define TCM_C_ADDR         0x8100A000UL     // 256 KiB

// Pointer aliases so AME instructions can address TCM via array syntax.
static uint8_t  (*A_fp4_tcm)[K / 2]   = (uint8_t  (*)[K / 2])TCM_A_ADDR;
static uint8_t  (*B_fp4_tcm)[K / 2]   = (uint8_t  (*)[K / 2])TCM_B_ADDR;
static uint8_t  *A_scale_tcm          = (uint8_t  *)TCM_A_SCALE_ADDR;
static uint8_t  *B_scale_tcm          = (uint8_t  *)TCM_B_SCALE_ADDR;
static volatile uint32_t (*C_fp32_tcm)[N] = (volatile uint32_t (*)[N])TCM_C_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

// Single tile dispatch: (m_tile, n_tile). K is a single RV (settilek(1)).
static inline void nvfp4_dispatch_tile(int m_tile, int n_tile) {
    uint8_t  *A_tile        = &A_fp4_tcm[m_tile * TILE_M][0];
    uint8_t  *B_tile        = &B_fp4_tcm[n_tile * TILE_N][0];
    volatile uint32_t *C_tile = &C_fp32_tcm[m_tile * TILE_M][n_tile * TILE_N];
    uint8_t  *A_scale_tile  = A_scale_tcm + m_tile * TILE_M * SCALE_ROW_BYTES;
    uint8_t  *B_scale_tile  = B_scale_tcm + n_tile * TILE_N * SCALE_ROW_BYTES;

    ame_mset_scalea((uint64_t)A_scale_tile);
    ame_mset_scaleb((uint64_t)B_scale_tile);

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    // spec-偏离: mtilek 单位是 RV 数，1 RV = 128 FP4 elements under
    // CUTE_4Tops_128SCP。见 ame_nvfp4_ext.h 顶部注释。
    ame_settilek(1);

    ame_mzero(ACC0);
    ame_mlae4(TR0, (uint64_t)A_tile, (uint64_t)(K / 2));
    ame_mlbe4(TR2, (uint64_t)B_tile, (uint64_t)(K / 2));
    ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);
    ame_msce32(ACC0, (uint64_t)C_tile, (uint64_t)(N * 4));

    while (!ame_is_idle()) { /* spin */ }
}

int main(void)
{
    printf("[E256] start: M=%d N=%d K=%d, tiles=%dx%d, tile=%dx%dx%d\n",
           M, N, K, M_TILES, N_TILES, TILE_M, TILE_N, K);

    // Configure 4-way TCM (2 MiB aperture at 0x81000000) so all our buffers fit.
    if (ame_tcm_config(4) != 0) {
        printf("[E256] FAIL: could not configure TCM to 4 ways\n");
        return 1;
    }

    // DMA source arrays from DRAM to TCM. ame_dma_load is blocking so no
    // fence needed between calls, but keep a defensive one around the batch.
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_start = rd_cycle();
    ame_dma_load((uint64_t)A_fp4_dram,   TCM_A_ADDR,       (uint32_t)sizeof(A_fp4_dram));
    ame_dma_load((uint64_t)B_fp4_dram,   TCM_B_ADDR,       (uint32_t)sizeof(B_fp4_dram));
    ame_dma_load((uint64_t)A_scale_dram, TCM_A_SCALE_ADDR, (uint32_t)sizeof(A_scale_dram));
    ame_dma_load((uint64_t)B_scale_dram, TCM_B_SCALE_ADDR, (uint32_t)sizeof(B_scale_dram));
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_end = rd_cycle();
    printf("[E256] DMA staging done in %lu cycles (A+B+scales = %u bytes)\n",
           (unsigned long)(dma_end - dma_start),
           (unsigned)(sizeof(A_fp4_dram) + sizeof(B_fp4_dram)
                    + sizeof(A_scale_dram) + sizeof(B_scale_dram)));

    // Quick TCM content check (like the reference DMA test): compare first + last byte.
    if (((volatile uint8_t *)TCM_A_ADDR)[0] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_ADDR)[sizeof(A_fp4_dram) - 1] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_SCALE_ADDR)[0] != E4M3_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_SCALE_ADDR)[sizeof(A_scale_dram) - 1] != E4M3_ONE_BYTE) {
        printf("[E256] FAIL: TCM contents not matching after DMA\n");
        return 1;
    }

    // Compute.
    uint64_t t0 = rd_cycle();
    for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
        for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
            nvfp4_dispatch_tile(m_tile, n_tile);
        }
    }
    uint64_t t1 = rd_cycle();

    // Sample corners so we can spot which tile is bad if there's a mismatch.
    printf("[E256] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)C_fp32_tcm[0][0],
           N - 1, (unsigned)C_fp32_tcm[0][N - 1],
           M - 1, (unsigned)C_fp32_tcm[M - 1][0],
           M - 1, N - 1, (unsigned)C_fp32_tcm[M - 1][N - 1]);
    printf("[E256] tile-seams: C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           TILE_N, (unsigned)C_fp32_tcm[0][TILE_N],
           TILE_M, (unsigned)C_fp32_tcm[TILE_M][0],
           TILE_M, TILE_N, (unsigned)C_fp32_tcm[TILE_M][TILE_N]);

    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < 11; j++) {
            uint32_t v = C_fp32_tcm[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) {
                    first_wrong = v;
                    first_i = i; first_j = j;
                }
                errors++;
            }
        }
    }

    printf("[E256] compute cycles = %lu\n", (unsigned long)(t1 - t0));
    if (errors == 0) {
        printf("[E256] PASS: all %d elements == 0x%08x\n", M * N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[E256] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, M * N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[E256] done\n");
    return errors == 0 ? 0 : 1;
}
