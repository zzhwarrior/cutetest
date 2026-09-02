// Phase E-full: 512x512x512 NVFP4 GEMM, all TCM-resident.
//
// Extends e256_full to 4x4x4 tile grid (16 (M,N) tiles x 4 K-tiles = 64 macc).
// TCM budget (2 MiB total for 4-way config) used:
//   0x81000000  A_fp4    [512][256]         = 128 KiB
//   0x81020000  B_fp4    [512][256]         = 128 KiB
//   0x81040000  A_scale  [K_TILES][M][8]    =  16 KiB
//   0x81044000  B_scale  [K_TILES][N][8]    =  16 KiB
//   0x81048000  C_fp32   [512][512]         =   1 MiB
//   Total                                   = 1.28 MiB   (of 2 MiB)
//
// Tile schedule (ACC0/ACC1 ping-pong across (M,N) tiles, K double-buffered
// TR0/TR2 <-> TR1/TR3 within each tile — mirrors ame_matmul_512x512x512_dma.c
// pattern to hide msce32 latency behind the next tile's compute):
//   acct=0
//   for mt in 0..3: for nt in 0..3:
//     ACC := (acct==0) ? ACC0 : ACC1
//     mzero(ACC)
//     for kt in 0..3:
//       mset_scalea/b(per-tile scale base)
//       (kt even → TR0/TR2, odd → TR1/TR3)
//       mlae4, mlbe4, mfmacc.s.nvfp4 into ACC
//     msce32(ACC, C_tile, stride)
//     acct ^= 1
//   ame_fence()   ← single fence at end, not per-tile
//
// Golden math (A=B=1.0, scale=1.0):
//   C[i][j] = Σ_{k=0..511} 1.0 * 1.0 * 1.0 * 1.0 = 512.0 = 0x44000000

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define APP_M       512
#define APP_N       512
#define APP_K       512
#define TILE_M      128
#define TILE_N      128
#define TILE_K      128                      // 1 RV @ CUTE_4Tops_128SCP
#define NVFP4_BLOCK 16
#define TILES_M     (APP_M / TILE_M)         // 4
#define TILES_N     (APP_N / TILE_N)         // 4
#define TILES_K     (APP_K / TILE_K)         // 4

#define FP4_ONE_BYTE     0x22u
#define E4M3_ONE_BYTE    0x38u
#define EXPECTED_C_BITS  0x44000000u         // fp32 512.0

#define A_ROW_BYTES         (APP_K / 2)                        // 256
#define SCALE_ROW_BYTES     (TILE_K / NVFP4_BLOCK)             // 8
#define SCALE_TILE_BYTES    (APP_M * SCALE_ROW_BYTES)          // 4096  per K-tile
#define SCALE_TOTAL_BYTES   (TILES_K * SCALE_TILE_BYTES)       // 16384

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
#define TCM_B_ADDR         0x81020000UL
#define TCM_A_SCALE_ADDR   0x81040000UL
#define TCM_B_SCALE_ADDR   0x81044000UL
#define TCM_C_ADDR         0x81048000UL

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
    printf("[E512] start: %dx%dx%d NVFP4, tiles=%dx%dx%d, tile=%dx%dx%d\n",
           APP_M, APP_N, APP_K, TILES_M, TILES_N, TILES_K,
           TILE_M, TILE_N, TILE_K);

    // 4-way TCM = 2 MiB. Our footprint = 1.28 MiB, well within budget.
    if (ame_tcm_config(4) != 0) {
        printf("[E512] FAIL: TCM config to 4 ways\n");
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
    printf("[E512] DMA staging done in %lu cycles (%u bytes)\n",
           (unsigned long)(dma_end - dma_start),
           (unsigned)(sizeof(A_fp4_dram) + sizeof(B_fp4_dram)
                    + sizeof(A_scale_dram) + sizeof(B_scale_dram)));

    if (((volatile uint8_t *)TCM_A_ADDR)[0] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_ADDR)[sizeof(A_fp4_dram) - 1] != FP4_ONE_BYTE ||
        ((volatile uint8_t *)TCM_A_SCALE_ADDR)[0] != E4M3_ONE_BYTE) {
        printf("[E512] FAIL: TCM contents mismatch after DMA\n");
        return 1;
    }

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);   // clamped to ReduceGroupSize by AMEDecoder

    const uint64_t a_stride = A_ROW_BYTES;
    const uint64_t b_stride = A_ROW_BYTES;
    const uint64_t c_stride = APP_N * sizeof(uint32_t);

    // ACC0/ACC1 ping-pong across (M,N) tiles — mirrors ame_matmul_512x512x512_dma.c:127-163.
    // Different ACC → different C SCP bank (c_scp_bank = inst_md(0) — ACC0→bank 0,
    // ACC1→bank 1). Previous tile's msce32 (CML reading CSpad(prev)) overlaps with
    // current tile's mzero/mfmacc (CDC on the other bank) with no CSpad port conflict.
    // No per-tile ame_fence() needed — only one at end of GEMM. AME operand registers
    // must be compile-time constants so the two branches are explicit.
    uint64_t gemm_start = rd_cycle();
    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)&c_tcm[mt * TILE_M][nt * TILE_N];

            if (acct == 0) {
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
                acct = 1;
            } else {
                ame_mzero(ACC1);
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
                        ame_mfmacc_s_nvfp4(ACC1, TR0, TR2);
                    } else {
                        ame_mlae4(TR1, a_base, a_stride);
                        ame_mlbe4(TR3, b_base, b_stride);
                        ame_mfmacc_s_nvfp4(ACC1, TR1, TR3);
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
    printf("[E512] compute cycles = %lu\n", (unsigned long)(gemm_end - gemm_start));

    // Sample corners + all tile seams. Full-scan (262144 elements) would add
    // ~1M cycles to sim time; targeted sampling covers every (M,N) tile boundary.
    printf("[E512] corners: C[0][0]=%08x C[0][%d]=%08x C[%d][0]=%08x C[%d][%d]=%08x\n",
           (unsigned)c_tcm[0][0],
           APP_N - 1, (unsigned)c_tcm[0][APP_N - 1],
           APP_M - 1, (unsigned)c_tcm[APP_M - 1][0],
           APP_M - 1, APP_N - 1, (unsigned)c_tcm[APP_M - 1][APP_N - 1]);
    printf("[E512] N-seams row 0:   C[0][128]=%08x C[0][256]=%08x C[0][384]=%08x\n",
           (unsigned)c_tcm[0][128], (unsigned)c_tcm[0][256], (unsigned)c_tcm[0][384]);
    printf("[E512] M-seams col 0:   C[128][0]=%08x C[256][0]=%08x C[384][0]=%08x\n",
           (unsigned)c_tcm[128][0], (unsigned)c_tcm[256][0], (unsigned)c_tcm[384][0]);
    printf("[E512] MN-seams:        C[128][128]=%08x C[256][256]=%08x C[384][384]=%08x\n",
           (unsigned)c_tcm[128][128], (unsigned)c_tcm[256][256], (unsigned)c_tcm[384][384]);

    // Sampled verification: every 16th row, all N-tile boundary columns +/-1.
    int errors = 0;
    int first_i = -1, first_j = -1;
    uint32_t first_wrong = 0;
    const int check_cols[] = {
        0, 1, 127, 128, 129,
        255, 256, 257,
        383, 384, 385,
        510, 511
    };
    const int n_check_cols = sizeof(check_cols) / sizeof(check_cols[0]);
    for (int i = 0; i < APP_M; i += 16) {
        for (int c = 0; c < n_check_cols; c++) {
            int j = check_cols[c];
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
    int total_checked = (APP_M / 16) * n_check_cols;

    if (errors == 0) {
        printf("[E512] PASS: sampled %d elements all == 0x%08x\n",
               total_checked, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[E512] FAIL: %d/%d wrong; first C[%d][%d]=0x%08x expected 0x%08x\n",
               errors, total_checked, first_i, first_j,
               (unsigned)first_wrong, (unsigned)EXPECTED_C_BITS);
    }
    printf("[E512] done\n");
    return errors == 0 ? 0 : 1;
}
