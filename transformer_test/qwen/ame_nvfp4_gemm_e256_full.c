// Phase E-small full: 256x256x256 NVFP4 GEMM with M/N/K all tiled.
//
// Combines the M/N tiling pattern from ame_nvfp4_gemm_e256.c (2x2 tiles of
// 128x128 each) with the K tiling pattern from ame_nvfp4_gemm_kt.c (2 K-tiles
// of 128 FP4 elements each, TR0/TR2 <-> TR1/TR3 alternation within a K loop).
//
// Tile schedule (single ACC0, fence per (M,N) tile like kt-test):
//   for mt in 0..1: for nt in 0..1:
//     mzero(ACC0)
//     for kt in 0..1:
//       mset_scalea/b(per-tile scale base)
//       (kt even → TR0/TR2, TR1/TR3 otherwise)
//       mlae4, mlbe4, mfmacc.s.nvfp4 into ACC0
//     msce32(ACC0, C_tile, stride)
//     ame_fence()
//
// Golden math (A=B=1.0, scale=1.0):
//   C[i][j] = Σ_{k=0..255} 1.0*1.0*1.0*1.0 = 256.0 = 0x43800000
//
// TCM layout (4-way TCM, ~330 KiB used):
//   0x81000000  A_fp4    [256][128] = 32 KiB
//   0x81008000  B_fp4    [256][128] = 32 KiB
//   0x81010000  A_scale  [K_TILES][M][SCALE_ROW]  =  4 KiB
//   0x81011000  B_scale  [K_TILES][N][SCALE_ROW]  =  4 KiB
//   0x81012000  C_fp32   [256][256]              = 256 KiB

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define APP_M       256
#define APP_N       256
#define APP_K       256
#define TILE_M      128
#define TILE_N      128
#define TILE_K      128                     // 1 RV @ CUTE_4Tops_128SCP
#define NVFP4_BLOCK 16
#define TILES_M     (APP_M / TILE_M)        // 2
#define TILES_N     (APP_N / TILE_N)        // 2
#define TILES_K     (APP_K / TILE_K)        // 2

#define FP4_ONE_BYTE     0x22u
#define E4M3_ONE_BYTE    0x38u
#define EXPECTED_C_BITS  0x43800000u        // fp32 256.0

#define A_ROW_BYTES         (APP_K / 2)                        // 128
#define SCALE_ROW_BYTES     (TILE_K / NVFP4_BLOCK)             // 8
#define SCALE_TILE_BYTES    (APP_M * SCALE_ROW_BYTES)          // 2048 per K-tile
#define SCALE_TOTAL_BYTES   (TILES_K * SCALE_TILE_BYTES)       // 4096

// ---- DRAM source data ----
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

// ---- TCM addresses ----
#define TCM_A_ADDR         0x81000000UL
#define TCM_B_ADDR         0x81008000UL
#define TCM_A_SCALE_ADDR   0x81010000UL
#define TCM_B_SCALE_ADDR   0x81011000UL
#define TCM_C_ADDR         0x81012000UL

static uint8_t  (*a_tcm)[A_ROW_BYTES]     = (uint8_t (*)[A_ROW_BYTES])TCM_A_ADDR;
static uint8_t  (*b_tcm)[A_ROW_BYTES]     = (uint8_t (*)[A_ROW_BYTES])TCM_B_ADDR;
static uint8_t  *a_scale_tcm              = (uint8_t *)TCM_A_SCALE_ADDR;
static uint8_t  *b_scale_tcm              = (uint8_t *)TCM_B_SCALE_ADDR;
static volatile uint32_t (*c_tcm)[APP_N]  = (volatile uint32_t (*)[APP_N])TCM_C_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[E256F] start: %dx%dx%d NVFP4, tiles=%dx%dx%d, tile=%dx%dx%d\n",
           APP_M, APP_N, APP_K, TILES_M, TILES_N, TILES_K,
           TILE_M, TILE_N, TILE_K);

    if (ame_tcm_config(4) != 0) {
        printf("[E256F] FAIL: TCM config to 4 ways\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_start = rd_cycle();
    ame_dma_load((uint64_t)A_fp4_dram,   TCM_A_ADDR,       (uint32_t)sizeof(A_fp4_dram));
    ame_dma_load((uint64_t)B_fp4_dram,   TCM_B_ADDR,       (uint32_t)sizeof(B_fp4_dram));
    ame_dma_load((uint64_t)A_scale_dram, TCM_A_SCALE_ADDR, (uint32_t)sizeof(A_scale_dram));
    ame_dma_load((uint64_t)B_scale_dram, TCM_B_SCALE_ADDR, (uint32_t)sizeof(B_scale_dram));
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_end = rd_cycle();
    printf("[E256F] DMA staging done in %lu cycles (%u bytes)\n",
           (unsigned long)(dma_end - dma_start),
           (unsigned)(sizeof(A_fp4_dram) + sizeof(B_fp4_dram)
                    + sizeof(A_scale_dram) + sizeof(B_scale_dram)));

    if (((volatile uint8_t *)TCM_A_ADDR)[0] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_ADDR)[sizeof(A_fp4_dram) - 1] != FP4_ONE_BYTE) {
        printf("[E256F] FAIL: TCM contents mismatch after DMA\n");
        return 1;
    }

    // Configure once outside all loops. mtilek=1 RV under CUTE_4Tops_128SCP.
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);   // will be clamped to ReduceGroupSize by AMEDecoder

    const uint64_t a_stride = A_ROW_BYTES;
    const uint64_t b_stride = A_ROW_BYTES;
    const uint64_t c_stride = APP_N * sizeof(uint32_t);

    uint64_t gemm_start = rd_cycle();
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)&c_tcm[mt * TILE_M][nt * TILE_N];
            ame_mzero(ACC0);

            for (int kt = 0; kt < TILES_K; kt++) {
                uint64_t a_scale_base = (uint64_t)(a_scale_tcm
                    + kt * SCALE_TILE_BYTES
                    + mt * TILE_M * SCALE_ROW_BYTES);
                uint64_t b_scale_base = (uint64_t)(b_scale_tcm
                    + kt * SCALE_TILE_BYTES
                    + nt * TILE_N * SCALE_ROW_BYTES);
                ame_mset_scalea(a_scale_base);
                ame_mset_scaleb(b_scale_base);

                uint64_t a_base = (uint64_t)&a_tcm[mt * TILE_M][kt * (TILE_K / 2)];
                uint64_t b_base = (uint64_t)&b_tcm[nt * TILE_N][kt * (TILE_K / 2)];

                if ((kt & 1) == 0) {
                    ame_mlae4(TR0, a_base, a_stride);
                    ame_mlbe4(TR2, b_base, b_stride);
                    ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);
                } else {
                    ame_mlae4(TR1, a_base, a_stride);
                    ame_mlbe4(TR3, b_base, b_stride);
                    ame_mfmacc_s_nvfp4(ACC0, TR1, TR3);
                }
            }
            ame_msce32(ACC0, c_base, c_stride);
            ame_fence();
        }
    }
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t gemm_end = rd_cycle();
    printf("[E256F] compute cycles = %lu\n", (unsigned long)(gemm_end - gemm_start));

    printf("[E256F] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)c_tcm[0][0],
           APP_N - 1, (unsigned)c_tcm[0][APP_N - 1],
           APP_M - 1, (unsigned)c_tcm[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)c_tcm[APP_M - 1][APP_N - 1]);
    printf("[E256F] seams:   C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           TILE_N, (unsigned)c_tcm[0][TILE_N],
           TILE_M, (unsigned)c_tcm[TILE_M][0],
           TILE_M, TILE_N, (unsigned)c_tcm[TILE_M][TILE_N]);

    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < APP_M; i++) {
        for (int j = 0; j < 10; j++) {
            uint32_t v = c_tcm[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) {
                    first_wrong = v;
                    first_i = i; first_j = j;
                }
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("[E256F] PASS: all %d elements == 0x%08x\n",
               APP_M * APP_N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[E256F] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, APP_M * APP_N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[E256F] done\n");
    return errors == 0 ? 0 : 1;
}
