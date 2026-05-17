// AME Test: 128x128x64 INT8 Matrix Multiplication
// Single tile operation: the entire GEMM fits in one tile (no tiling loop needed)
// C[128][128] += A[128][64] * B[128][64]^T, int8 -> int32 accumulator

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "marchid.h"
#include "matmul_value_mnk_128_128_64.h"
#include "matmul_cref_128_128_64.h"
#define APPLICATION_M 128
#define APPLICATION_N 128
#define APPLICATION_K 64
#define ARRAY_BASE_ADDR  0x81000000

int main(void) {
    printf("AME Test: %dx%dx%d INT8 GEMM (single tile)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K);
    // 访问数组（通过指针）
    volatile int32_t (*c)[APPLICATION_N] = (volatile int32_t (*)[APPLICATION_N])ARRAY_BASE_ADDR;
    for (int i = 0; i < APPLICATION_M; i++)
        for (int j = 0; j < APPLICATION_N; j++)
            c[i][j] =0;

    int errors = 0;
    printf("init done\n");

    // --- AME single-tile GEMM ---
    uint64_t a_stride = APPLICATION_K * sizeof(int8_t);   // 64 bytes per row
    uint64_t b_stride = APPLICATION_K * sizeof(int8_t);   // 64 bytes per row
    uint64_t c_stride = APPLICATION_N * sizeof(int32_t);  // 512 bytes per row
    
    // Configure tile dimensions (entire matrix fits in one tile)
    ame_settilem(APPLICATION_M);
    ame_settilen(APPLICATION_N);
    ame_settilek(APPLICATION_K);

    uint64_t cycle_start, cycle_end;
    asm volatile("rdcycle %0" : "=r"(cycle_start));

    // 1. Zero accumulator acc0 (bank 0)
    ame_mzero(ACC0);

    // 2. Load A[128][64] into tr0 (A bank 0)
    ame_mlae8(TR0, (uint64_t)a, a_stride);

    // 3. Load B[128][64] into tr2 (B bank 0)
    ame_mlbe8(TR2, (uint64_t)b, b_stride);

    // 4. Compute: acc0 += tr0 * tr2^T  (A-bank0 * B-bank0 -> C-bank0)
    ame_mmacc_w_b(ACC0, TR0, TR2);

    // 5. Store acc0 -> C[128][128]
    ame_msce32(ACC0, (uint64_t)c, c_stride);

    uint64_t res1 = 1;  
    res1 = ame_is_idle();
    while(!res1)
    {
        res1 = ame_is_idle();
    }
    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);

    
    for (int i = 0; i < APPLICATION_M; i++) {
        for (int j = 0; j < APPLICATION_N; j++) {
            if (c[i][j] != c_ref[i][j]) {
                if (errors < 10)
                    printf("MISMATCH C[%d][%d]: got %d, expected %d\n",
                           i, j, c[i][j], c_ref[i][j]);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("PASS! All %d elements match.\n", APPLICATION_M * APPLICATION_N);
    } else {
        printf("FAIL! %d mismatches out of %d elements.\n",
               errors, APPLICATION_M * APPLICATION_N);
    }
    return 0;
}
