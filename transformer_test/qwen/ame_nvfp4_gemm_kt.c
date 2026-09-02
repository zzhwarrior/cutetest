// Phase E-medium: NVFP4 GEMM single (M,N) tile with K-tile TR0/TR2 → TR1/TR3
// unrolled. Shape: 128x128x256, single (M,N) tile, 2 K-tiles, both into ACC0.
//
//   ame_mzero(ACC0)
//   kt=0: ame_mset_scalea/b, ame_mlae4(TR0)/ame_mlbe4(TR2), ame_mfmacc.s.nvfp4(ACC0, TR0, TR2)
//   kt=1: ame_mset_scalea/b, ame_mlae4(TR1)/ame_mlbe4(TR3), ame_mfmacc.s.nvfp4(ACC0, TR1, TR3)
//   ame_msce32(ACC0, ...)
//   ame_fence()
//
// Golden: A_fp4 = B_fp4 = 1.0 (0x22), scale = 1.0 (0x38)
//   C[i][j] = Σ_{k=0..255} 1.0 = 256.0 = 0x43800000

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define APP_M       128
#define APP_N       128
#define APP_K       256
#define TILE_M      128
#define TILE_N      128
#define TILE_K      128                     // FP4 elements = 1 RV
#define NVFP4_BLOCK 16
#define TILES_M     (APP_M / TILE_M)        // 1
#define TILES_N     (APP_N / TILE_N)        // 1
#define TILES_K     (APP_K / TILE_K)        // 2

#define FP4_ONE_BYTE     0x22u
#define E4M3_ONE_BYTE    0x38u
#define EXPECTED_C_BITS  0x43800000u        // fp32 256.0

#define A_ROW_BYTES         (APP_K / 2)                        // 128
#define SCALE_ROW_BYTES     (TILE_K / NVFP4_BLOCK)             // 8
#define SCALE_TILE_BYTES    (APP_M * SCALE_ROW_BYTES)          // 1024 per K-tile
#define SCALE_TOTAL_BYTES   (TILES_K * SCALE_TILE_BYTES)       // 2048

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
    printf("[E-KT] start: %dx%dx%d NVFP4, tiles=%dx%dx%d\n",
           APP_M, APP_N, APP_K, TILES_M, TILES_N, TILES_K);

    if (ame_tcm_config(4) != 0) {
        printf("[E-KT] FAIL: TCM config to 4 ways\n");
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
    printf("[E-KT] DMA staging done in %lu cycles\n", (unsigned long)(dma_end - dma_start));

    if (((volatile uint8_t *)TCM_A_ADDR)[0] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_ADDR)[sizeof(A_fp4_dram) - 1] != FP4_ONE_BYTE) {
        printf("[E-KT] FAIL: TCM contents mismatch after DMA\n");
        return 1;
    }

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    const uint64_t a_stride = A_ROW_BYTES;
    const uint64_t b_stride = A_ROW_BYTES;
    const uint64_t c_stride = APP_N * sizeof(uint32_t);

    uint64_t c_base = (uint64_t)&c_tcm[0][0];

    uint64_t gemm_start = rd_cycle();

    ame_mzero(ACC0);

    // K-tile 0: TR0/TR2 → ACC0
    {
        uint64_t a_scale_base = (uint64_t)(a_scale_tcm + 0 * SCALE_TILE_BYTES);
        uint64_t b_scale_base = (uint64_t)(b_scale_tcm + 0 * SCALE_TILE_BYTES);
        ame_mset_scalea(a_scale_base);
        ame_mset_scaleb(b_scale_base);

        uint64_t a_base = (uint64_t)&a_tcm[0][0 * (TILE_K / 2)];
        uint64_t b_base = (uint64_t)&b_tcm[0][0 * (TILE_K / 2)];

        ame_mlae4(TR0, a_base, a_stride);
        ame_mlbe4(TR2, b_base, b_stride);
        ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);
    }

    // K-tile 1: TR1/TR3 → ACC0
    {
        uint64_t a_scale_base = (uint64_t)(a_scale_tcm + 1 * SCALE_TILE_BYTES);
        uint64_t b_scale_base = (uint64_t)(b_scale_tcm + 1 * SCALE_TILE_BYTES);
        ame_mset_scalea(a_scale_base);
        ame_mset_scaleb(b_scale_base);

        uint64_t a_base = (uint64_t)&a_tcm[0][1 * (TILE_K / 2)];
        uint64_t b_base = (uint64_t)&b_tcm[0][1 * (TILE_K / 2)];

        ame_mlae4(TR1, a_base, a_stride);
        ame_mlbe4(TR3, b_base, b_stride);
        ame_mfmacc_s_nvfp4(ACC0, TR1, TR3);
    }

    ame_msce32(ACC0, c_base, c_stride);
    asm volatile ("fence rw, rw" ::: "memory");

    uint64_t gemm_end = rd_cycle();
    printf("[E-KT] compute cycles = %lu\n", (unsigned long)(gemm_end - gemm_start));

    printf("[E-KT] corners:  C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)c_tcm[0][0],
           APP_N - 1, (unsigned)c_tcm[0][APP_N - 1],
           APP_M - 1, (unsigned)c_tcm[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)c_tcm[APP_M - 1][APP_N - 1]);

    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < APP_M; i++) {
        for (int j = 0; j < 11; j++) {
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
        printf("[E-KT] PASS: all %d elements == 0x%08x\n",
               APP_M * APP_N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[E-KT] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, APP_M * APP_N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[E-KT] done\n");
    return errors == 0 ? 0 : 1;
}
