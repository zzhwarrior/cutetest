// Phase F multi-tile: 256x256x256 BF16 GEMM on AME, M/N/K all tiled.
//
// Structural mirror of ame_nvfp4_gemm_e256_full.c, swapping:
//   - fp4 packed data (2 elem/byte) → bf16 (2 bytes/elem)
//   - mlae4/mlbe4 → mlae16/mlbe16
//   - mfmacc.s.nvfp4 → mfmacc.s.bf16
//   - scale bases: BF16 doesn't use scales; drop mset_scalea/b
//
// Tile K (elements) per RV: RWByte=64, bf16=2 bytes/elem → 32 elem/RV.
// So TILE_K = 32 (single RV), TILES_K = APP_K / 32 = 8 for APP_K=256.
//
// TCM layout (all resident):
//   0x81000000  A_bf16   [256][256] * 2 = 128 KiB
//   0x81020000  B_bf16   [256][256] * 2 = 128 KiB
//   0x81040000  C_fp32   [256][256] * 4 = 256 KiB
//   Total = 512 KiB (of 2 MiB)
//
// Golden math (A=B=1.0):
//   C[i][j] = Σ_{k=0..255} 1.0 * 1.0 = 256.0 = 0x43800000

#include <stdio.h>
#include <stdint.h>
#include "../../ame_test/ame.h"

#define APP_M       256
#define APP_N       256
#define APP_K       256
#define TILE_M      128
#define TILE_N      128
#define TILE_K      32                       // 1 RV in BF16 @ CUTE_4Tops_128SCP
#define TILES_M     (APP_M / TILE_M)         // 2
#define TILES_N     (APP_N / TILE_N)         // 2
#define TILES_K     (APP_K / TILE_K)         // 8

#define BF16_ONE         0x3F80u              // bf16 1.0
#define EXPECTED_C_BITS  0x43800000u          // fp32 256.0

#define A_ROW_ELEMS      APP_K                // bf16, 1 elem/2byte, stride in bytes
#define A_ROW_BYTES      (A_ROW_ELEMS * 2)    // 512 bytes/row

static const uint16_t A_bf16_dram[APP_M][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_M - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};
static const uint16_t B_bf16_dram[APP_N][APP_K] __attribute__((aligned(64))) = {
    [0 ... APP_N - 1] = { [0 ... APP_K - 1] = BF16_ONE }
};

#define TCM_A_ADDR   0x81000000UL
#define TCM_B_ADDR   0x81020000UL
#define TCM_C_ADDR   0x81040000UL

static uint16_t (*a_tcm)[APP_K]           = (uint16_t (*)[APP_K])TCM_A_ADDR;
static uint16_t (*b_tcm)[APP_K]           = (uint16_t (*)[APP_K])TCM_B_ADDR;
static volatile uint32_t (*c_tcm)[APP_N]  = (volatile uint32_t (*)[APP_N])TCM_C_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[BF16-F] start: %dx%dx%d BF16, tiles=%dx%dx%d, tile=%dx%dx%d\n",
           APP_M, APP_N, APP_K, TILES_M, TILES_N, TILES_K,
           TILE_M, TILE_N, TILE_K);

    if (ame_tcm_config(4) != 0) {
        printf("[BF16-F] FAIL: TCM config\n");
        return 1;
    }

    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_start = rd_cycle();
    ame_dma_load((uint64_t)A_bf16_dram, TCM_A_ADDR, (uint32_t)sizeof(A_bf16_dram));
    ame_dma_load((uint64_t)B_bf16_dram, TCM_B_ADDR, (uint32_t)sizeof(B_bf16_dram));
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_end = rd_cycle();
    printf("[BF16-F] DMA done in %lu cycles\n", (unsigned long)(dma_end - dma_start));

    if (((volatile uint16_t *)TCM_A_ADDR)[0] != BF16_ONE ||
        ((volatile uint16_t *)TCM_A_ADDR)[APP_M * APP_K - 1] != BF16_ONE) {
        printf("[BF16-F] FAIL: TCM A mismatch\n");
        return 1;
    }

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    const uint64_t a_stride = A_ROW_BYTES;
    const uint64_t b_stride = A_ROW_BYTES;
    const uint64_t c_stride = APP_N * sizeof(uint32_t);

    // ACC0/ACC1 ping-pong across (M,N) tiles — same pattern as e512 / the
    // int8 512x512x512 reference. Different ACC → different C SCP bank, so
    // previous tile's msce32 store overlaps with current tile's mfmacc on
    // the other CSpad bank without port conflict. One ame_fence() at end.
    uint64_t gemm_start = rd_cycle();
    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)&c_tcm[mt * TILE_M][nt * TILE_N];

            if (acct == 0) {
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_base = (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K];
                    uint64_t b_base = (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K];

                    if ((kt & 1) == 0) {
                        ame_mlae16(TR0, a_base, a_stride);
                        ame_mlbe16(TR2, b_base, b_stride);
                        ame_mfmacc_s_bf16(ACC0, TR0, TR2);
                    } else {
                        ame_mlae16(TR1, a_base, a_stride);
                        ame_mlbe16(TR3, b_base, b_stride);
                        ame_mfmacc_s_bf16(ACC0, TR1, TR3);
                    }
                }
                ame_msce32(ACC0, c_base, c_stride);
                acct = 1;
            } else {
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    uint64_t a_base = (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K];
                    uint64_t b_base = (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K];

                    if ((kt & 1) == 0) {
                        ame_mlae16(TR0, a_base, a_stride);
                        ame_mlbe16(TR2, b_base, b_stride);
                        ame_mfmacc_s_bf16(ACC1, TR0, TR2);
                    } else {
                        ame_mlae16(TR1, a_base, a_stride);
                        ame_mlbe16(TR3, b_base, b_stride);
                        ame_mfmacc_s_bf16(ACC1, TR1, TR3);
                    }
                }
                ame_msce32(ACC1, c_base, c_stride);
                acct = 0;
            }
        }
    }
    ame_fence();
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t gemm_end = rd_cycle();
    printf("[BF16-F] compute cycles = %lu\n", (unsigned long)(gemm_end - gemm_start));

    printf("[BF16-F] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)c_tcm[0][0],
           APP_N - 1, (unsigned)c_tcm[0][APP_N - 1],
           APP_M - 1, (unsigned)c_tcm[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)c_tcm[APP_M - 1][APP_N - 1]);
    printf("[BF16-F] seams:   C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           TILE_N, (unsigned)c_tcm[0][TILE_N],
           TILE_M, (unsigned)c_tcm[TILE_M][0],
           TILE_M, TILE_N, (unsigned)c_tcm[TILE_M][TILE_N]);

    int errors = 0, first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    for (int i = 0; i < APP_M; i++) {
        for (int j = 0; j < APP_N; j++) {
            uint32_t v = c_tcm[i][j];
            if (v != EXPECTED_C_BITS) {
                if (errors == 0) { first_wrong = v; first_i = i; first_j = j; }
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("[BF16-F] PASS: all %d elements == 0x%08x\n",
               APP_M * APP_N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[BF16-F] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, APP_M * APP_N, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[BF16-F] done\n");
    return errors == 0 ? 0 : 1;
}
