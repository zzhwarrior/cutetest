// AME Test: 256x256x128 INT8 Matrix Multiplication
// Tiled: M=256/128=2, N=256/128=2, K=128/64=2 -> 8 tiles total
// Double-buffered: tr0/tr2 (bank0) and tr1/tr3 (bank1) alternate each K step
// so Load(next) overlaps with Compute(current).

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "matmul_value_mnk_512_512_512.h"
#include "matmul_cref_512_512_512.h"
#define APPLICATION_M  512
#define APPLICATION_N  512
#define APPLICATION_K  512
#define TILE_M         128
#define TILE_N         128
#define TILE_K         64
#define TILES_M        (APPLICATION_M / TILE_M)   // 2
#define TILES_N        (APPLICATION_N / TILE_N)   // 2
#define TILES_K        (APPLICATION_K / TILE_K)   // 2
#define C_BASE_ADDR 0x81100000UL
// Row width MUST equal APPLICATION_N so &c[i][j] steps by APPLICATION_N*4
// bytes per row, matching the c_stride used by ame_msce32. The old [256]
// value here was a 256x256 leftover — it silently miscomputed &c[i][*] for
// i > 0 because pointer arithmetic used 1024 B/row instead of 2048 B/row.
static int32_t (*c)[APPLICATION_N] = (int32_t (*)[APPLICATION_N])C_BASE_ADDR;
// Input matrices (int8), row-major - provided by matmul_value header
// Output matrix (int32) - provided by matmul_value header


int main(void) {
    printf("AME Test: %dx%dx%d INT8 GEMM (tiled %dx%dx%d, double-buffered)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K,
           TILE_M, TILE_N, TILE_K);


    printf("a base: %p (expected 0x81040000 for TCM build)\n", (void *)a);
    printf("b base: %p (expected 0x81000000 for TCM build)\n", (void *)b);
    printf("c base: %p (expected 0x81100000)\n", (void *)c);
    printf("init done\n");

    // ---- TCM partition control ---------------------------------------------
    // A (256 KiB) and B (256 KiB) live inside the TCM aperture. C (1 MiB) may
    // or may not — depends on the config's way_bytes:
    //   * testL2   (sets=4096, way_bytes=256 KiB): 4-way TCM = 1 MiB. C at
    //     0x81100000 falls OUTSIDE the aperture and is served by cache/DRAM.
    //   * testL2Dma (sets=8192, way_bytes=512 KiB): 4-way TCM = 2 MiB. C at
    //     0x81100000 falls INSIDE the aperture, in ways ways-3 and ways-4.
    // Either way, A and B live in the top way (way 7 under grow-from-top).
    // Ensuring count=4 up front satisfies both configs.
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

    uint64_t cycle_start, cycle_end;
    

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    uint64_t a_stride = APPLICATION_K * sizeof(int8_t);   // 128 bytes per row
    uint64_t b_stride = APPLICATION_K * sizeof(int8_t);   // 128 bytes per row
    uint64_t c_stride = APPLICATION_N * sizeof(int32_t);  // 1024 bytes per row
    asm volatile("rdcycle %0" : "=r"(cycle_start));
    
    int acct = 0; 
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            uint64_t c_base = (uint64_t)&c[mt * TILE_M][nt * TILE_N];
            if(acct == 0){
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    // A tile: row block mt, col block kt -> &a[mt*TILE_M][kt*TILE_K]
                    // B tile: row block nt, col block kt -> &b[nt*TILE_N][kt*TILE_K]
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR2, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC0, TR0, TR2);
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR3, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC0, TR1, TR3);
                    }
                    //ame_fence();
                }
                ame_msce32(ACC0, c_base, c_stride);
                //ame_fence();
                acct++;
            }
            else{
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR2, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC1, TR0, TR2);
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        ame_mlbe8(TR3, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        ame_mmacc_w_b(ACC1, TR1, TR3);
                    }
                    //ame_fence();
                }
                ame_msce32(ACC1, c_base, c_stride);
                //ame_fence();
                acct--;
            }
        }
    }
      // Wait for all AME operations to complete
    asm volatile("fence" ::: "memory");
    //while (!ame_is_idle()) {}

    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);
    // Verify c[i][0..9] against c_ref for every row. Prior version compared
    // c[0][j] on every iteration, so it never actually checked rows 1..M-1
    // (the whole mt=1 half of the tile grid went unverified).
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
    }
    if (errors == 0) {
        printf("PASS! elements match.\n");
    } else {
        printf("FAIL! %d mismatches.\n", errors);
    }

    // ---- Demonstrate a safe TCM shrink after the workload ----------------
    // Under grow-from-top mapping, B (offset 0x00000..0x40000) and A
    // (0x40000..0x80000) both live in the top TCM way (way ways-1). Any
    // shrink to count >= 1 preserves them. We snapshot 8 bytes of B, shrink
    // to count=2, and confirm the sample survives.
    // (C may or may not have been in TCM depending on the config; either
    // way we've already read it back for verification, so we don't need it
    // preserved across the shrink.)
    {
        // Snapshot a few B bytes BEFORE the transition so we can compare.
        volatile int8_t *bp = (volatile int8_t *)0x81000000UL;
        int8_t b_sample[8];
        for (int i = 0; i < 8; i++) b_sample[i] = bp[i];

        if (ame_tcm_config(2) != 0) {
            printf("FAIL: could not shrink TCM to 2 ways\n");
            return 1;
        }
        printf("TCM shrunk: count=%u mask=0x%02x\n",
               ame_tcm_get_count(), ame_tcm_get_mask());

        // Verify B survived. Under grow-from-top, offset 0..512K stays in
        // way 7 regardless of count (as long as count >= 1).
        int b_errors = 0;
        for (int i = 0; i < 8; i++) {
            if (bp[i] != b_sample[i]) {
                if (b_errors < 4) {
                    printf("  B mismatch @ +%d: expected 0x%02x, got 0x%02x\n",
                           i, (uint8_t)b_sample[i], (uint8_t)bp[i]);
                }
                b_errors++;
            }
        }
        if (b_errors == 0) {
            printf("TCM shrink preserved B (spot-checked 8 bytes)\n");
        } else {
            printf("FAIL: TCM shrink lost B data\n");
            return 1;
        }

        // Restore the pre-workload partition for a clean exit.
        (void)ame_tcm_config(4);
    }

    return 0;
}
