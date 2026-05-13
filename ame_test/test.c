// AME Test: 128x128x64 INT8 Matrix Multiplication
// Single tile operation: the entire GEMM fits in one tile (no tiling loop needed)
// C[128][128] += A[128][64] * B[128][64]^T, int8 -> int32 accumulator

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "cuteMarcoinstHelper.h"
#include "marchid.h"
#include "matmul_value_mnk_128_128_64.h"
#define APPLICATION_M 128
#define APPLICATION_N 128
#define APPLICATION_K 64


int main(void) {
    //printf("AME Test: %dx%dx%d INT8 GEMM (single tile)\n",
    //       APPLICATION_M, APPLICATION_N, APPLICATION_K);

    //init_matrices();
    //memset(C, 0, sizeof(C));
    //compute_reference();
    printf("init\n");
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

    // 1. Zero accumulator (acc0)
    //ame_mzero();

    // 2. Load A[128][64] into tr0
    ame_mlae8((uint64_t)a, a_stride);

    // 3. Load B[128][64] into tr2 (B is stored transposed: N x K)
    ame_mlbe8((uint64_t)b, b_stride);

    // 4. Compute: acc0 += tr0 * tr2^T (mmacc.w.b)
    ame_mmacc_w_b();

    // 5. Store acc0 -> C[128][128] 指定C矩阵的地址
    ame_msce32(0x80020000, c_stride);

    uint64_t res1 = 1;  
    res1 = ame_is_idle();
    while(!res1)
    {
        res1 = ame_is_idle();
    }
    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);

    return 0;
}
