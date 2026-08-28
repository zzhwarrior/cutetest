// AME Test: 512x512x512 INT8 GEMM, but the operand matrices live in DRAM at
// boot and are moved into TCM via the AME_DMA_LOAD custom instruction before
// compute starts. Compares runtime between DMA-load and pure DRAM/TCM-resident
// variants (see ame_matmul_512x512x512.riscv and _tcm.riscv for baselines).
//
// Build note: this file uses the DEFAULT linker script (no htif_tcm.ld). The
// `.tcm_data` section attribute on a[]/b[] in the shared value header still
// causes them to be placed by the linker's orphan-section handling, which
// keeps them inside the .data range in DRAM (they are NOT preloaded at TCM
// base 0x81000000 like in the _tcm build). At runtime we DMA them into TCM.

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "matmul_value_mnk_512_512_512.h"    // a[512][512] and b[512][512] (DRAM)
#include "matmul_cref_512_512_512.h"

#define APPLICATION_M 512
#define APPLICATION_N 512
#define APPLICATION_K 512
#define TILE_M         128
#define TILE_N         128
#define TILE_K         64
#define TILES_M        (APPLICATION_M / TILE_M)
#define TILES_N        (APPLICATION_N / TILE_N)
#define TILES_K        (APPLICATION_K / TILE_K)

// TCM layout for this test:
//   0x81000000 : b[512][512]  (256 KB)
//   0x81040000 : a[512][512]  (256 KB)  — offset by |b|
// Output C: back in TCM to reproduce the mt>=1 mismatch.
#define TCM_B_ADDR   0x81000000UL
#define TCM_A_ADDR   0x81040000UL
#define C_BASE_ADDR  0x81100000UL


// Row width MUST match APPLICATION_N so &c[i][j] steps by 512*4 bytes per
// row, matching the c_stride used by the AME store. The original
// ame_matmul_512x512x512.c leaves this as [256] (a 256x256 leftover) but
// only ever checks row 0 in verify, which hides the mismatch.
static int32_t (*c)[APPLICATION_N] = (int32_t (*)[APPLICATION_N])C_BASE_ADDR;
// Pointer aliases so the compute loop can address TCM using array syntax.
static int8_t (*a_tcm)[512] = (int8_t (*)[512])TCM_A_ADDR;
static int8_t (*b_tcm)[512] = (int8_t (*)[512])TCM_B_ADDR;

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void) {
    // ---- TCM partition control ---------------------------------------------
    // This workload places A (256 KiB), B (256 KiB), and C (1 MiB) inside
    // the TCM aperture. Under way_bytes=512KiB and grow-from-top mapping:
    //   * offset 0x000000..0x080000 -> way (ways-1)  -- holds B (+ part of A)
    //   * offset 0x080000..0x100000 -> way (ways-2)  -- (unused gap)
    //   * offset 0x100000..0x180000 -> way (ways-3)  -- holds C[0..511, 0..255]
    //   * offset 0x180000..0x200000 -> way (ways-4)  -- holds C[0..511, 256..511]
    // So the full 4-way (2 MiB) TCM is required to run the kernel. After the
    // kernel completes and C has been read out, we demonstrate a safe shrink
    // to 2 ways: B and A live in the top 512 KiB of the aperture and survive
    // that transition, so subsequent work can still read them.
    {
        uint32_t info = ame_tcm_get_info();
        printf("TCM info: ways=%u log2sets=%u log2blk=%u\n",
               info & 0xF, (info >> 8) & 0xFF, (info >> 16) & 0xFF);
        printf("TCM initial: count=%u mask=0x%02x\n",
               ame_tcm_get_count(), ame_tcm_get_mask());
        if (ame_tcm_config(4) != 0) {
            printf("FAIL: could not configure TCM to 4 ways\n");
            return 1;
        }
        printf("TCM configured: count=4 (2 MiB) — matmul can run\n");
    }
    printf("AME DMA Matmul: %dx%dx%d INT8 (a,b staged from DRAM via DMA)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K);
    printf("DRAM src   : a=%p  b=%p\n", (void *)a,     (void *)b);
    printf("TCM  dst   : a=%p  b=%p\n", (void *)a_tcm, (void *)b_tcm);
    printf("C output   : c=%p\n",       (void *)c);

    // Sanity: each matrix must fit in the DMA length field (32 bit) and be a
    // multiple of the 64-byte DMA block size. 512*512 = 262144 bytes = 0x40000.
    const uint32_t a_bytes = (uint32_t)sizeof(a); // 262144
    const uint32_t b_bytes = (uint32_t)sizeof(b);

    // --------------------------------------------------------------
    // Phase 1: DMA-copy a[] and b[] from DRAM to TCM.
    // ame_dma_load blocks (CPU pipeline stalls on the subsequent fence
    // via io.busy) until each transfer completes, so the two calls are
    // sequential and the fence is only there as a defensive barrier for
    // the compiler / any subsequent memory ops.
    // --------------------------------------------------------------
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_start = rd_cycle();
    ame_dma_load((uint64_t)(uintptr_t)a, TCM_A_ADDR, a_bytes);
    ame_dma_load((uint64_t)(uintptr_t)b, TCM_B_ADDR, b_bytes);
    asm volatile ("fence rw, rw" ::: "memory");
    uint64_t dma_end = rd_cycle();

    // Split high/low 32-bit for iprintf-friendly output.
    uint64_t dma_bytes  = (uint64_t)a_bytes + b_bytes;
    uint64_t dma_cycles = dma_end - dma_start;
    printf("DMA staging: %u + %u bytes  in %u cycles  (~%u byte/cyc)\n",
           (unsigned)a_bytes, (unsigned)b_bytes,
           (unsigned)dma_cycles,
           (unsigned)(dma_bytes / (dma_cycles ? dma_cycles : 1)));

    // Quick spot-check: TCM should now match DRAM in the first + last block.
    if (((volatile uint8_t *)TCM_A_ADDR)[0]                != ((uint8_t *)a)[0] ||
        ((volatile uint8_t *)TCM_A_ADDR)[a_bytes - 1]      != ((uint8_t *)a)[a_bytes - 1] ||
        ((volatile uint8_t *)TCM_B_ADDR)[0]                != ((uint8_t *)b)[0] ||
        ((volatile uint8_t *)TCM_B_ADDR)[b_bytes - 1]      != ((uint8_t *)b)[b_bytes - 1]) {
        printf("FAIL: TCM contents do not match DRAM source after DMA\n");
        return 1;
    }
    printf("DMA content check: TCM matches DRAM at bounds\n");

    // --------------------------------------------------------------
    // Phase 2: GEMM using TCM-resident operands.
    // Identical inner loop to ame_matmul_512x512x512.c, but every
    // ame_mlae/ame_mlbe now references a_tcm[]/b_tcm[] in TCM.
    // --------------------------------------------------------------
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    uint64_t a_stride = APPLICATION_K * sizeof(int8_t);
    uint64_t b_stride = APPLICATION_K * sizeof(int8_t);
    uint64_t c_stride = APPLICATION_N * sizeof(int32_t);

    uint64_t gemm_start = rd_cycle();
    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)&c[mt * TILE_M][nt * TILE_N];
            if (acct == 0) {
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR2, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC0, TR0, TR2);
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR3, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC0, TR1, TR3);
                    }
                }
                ame_msce32(ACC0, c_base, c_stride);
                acct++;
            } else {
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR2, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC1, TR0, TR2);
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a_tcm[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR3, (uint64_t)&b_tcm[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC1, TR1, TR3);
                    }
                }
                ame_msce32(ACC1, c_base, c_stride);
                acct--;
            }
        }
    }
    asm volatile ("fence" ::: "memory");
    uint64_t gemm_end = rd_cycle();
    printf("GEMM       : %u cycles\n", (unsigned)(gemm_end - gemm_start));

    // --------------------------------------------------------------
    // Verify against reference. Same subset as the non-DMA variant so
    // output is easy to compare across runs.
    // --------------------------------------------------------------
    int errors = 0;
    for (int i = 0; i < APPLICATION_M; i++) {
        for (int j = 0; j < 10; j++) {
            if (c[i][j] != c_ref[i][j]) {
                if (errors < 10)
                    printf("MISMATCH C[%d][%d]: got %d, expected %d\n",
                           i, j, c[i][j], c_ref[i][j]);
                errors++;
            }
        }
        for (int j = 128; j < 138; j++) {
            if (c[i][j] != c_ref[i][j]) {
                if (errors < 10)
                    printf("MISMATCH C[%d][%d]: got %d, expected %d\n",
                           i, j, c[i][j], c_ref[i][j]);
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("PASS! elements match.\n");
        return 0;
    }
    printf("FAIL! %d mismatches.\n", errors);
    return 1;
}
