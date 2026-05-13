// AME Test: 128x128x64 INT8 Matrix Multiplication
// Single tile operation: the entire GEMM fits in one tile (no tiling loop needed)
// C[128][128] += A[128][64] * B[128][64]^T, int8 -> int32 accumulator

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"

#define APPLICATION_M 128
#define APPLICATION_N 128
#define APPLICATION_K 64

// Input matrices (int8)
static int8_t A[APPLICATION_M][APPLICATION_K] __attribute__((aligned(64)));
static int8_t B[APPLICATION_N][APPLICATION_K] __attribute__((aligned(64)));
// Output matrix (int32 accumulator)
static int32_t C[APPLICATION_M][APPLICATION_N] __attribute__((aligned(64)));
// Reference output
static int32_t C_ref[APPLICATION_M][APPLICATION_N];

static void init_matrices(void) {
    for (int i = 0; i < APPLICATION_M; i++)
        for (int j = 0; j < APPLICATION_K; j++)
            //A[i][j] = (int8_t)((i * 3 + j * 7 + 1) % 127 - 63);
            A[i][j] =0;

    for (int i = 0; i < APPLICATION_N; i++)
        for (int j = 0; j < APPLICATION_K; j++)
            B[i][j] = 0;//(int8_t)((i * 5 + j * 11 + 3) % 127 - 63);
}

static void compute_reference(void) {
    memset(C_ref, 0, sizeof(C_ref));
    for (int i = 0; i < APPLICATION_M; i++)
        for (int j = 0; j < APPLICATION_N; j++)
            for (int k = 0; k < APPLICATION_K; k++)
                C_ref[i][j] += (int32_t)A[i][k] * (int32_t)B[j][k];
}

static int verify_result(void) {
    int errors = 0;
    for (int i = 0; i < APPLICATION_M; i++) {
        for (int j = 0; j < APPLICATION_N; j++) {
            if (C[i][j] != C_ref[i][j]) {
                if (errors < 10)
                    printf("MISMATCH C[%d][%d]: got %d, expected %d\n",
                           i, j, C[i][j], C_ref[i][j]);
                errors++;
            }
        }
    }
    return errors;
}

int main(void) {
    printf("AME Test: %dx%dx%d INT8 GEMM (single tile)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K);

    init_matrices();
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
    ame_mlae8((uint64_t)A, a_stride);

    // 3. Load B[128][64] into tr2 (B is stored transposed: N x K)
    ame_mlbe8((uint64_t)B, b_stride);

    // 4. Compute: acc0 += tr0 * tr2^T (mmacc.w.b)
    ame_mmacc_w_b();

    // 5. Store acc0 -> C[128][128]
    ame_msce32((uint64_t)C, c_stride);

    uint64_t res1 = 1;
    res1 = cute_inst_fifo_finish_search();
    while(!res1)
    {
        //printf("Waiting for finish\n");
        res1 = cute_inst_fifo_finish_search();
    }

    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);

    // Verify
    /*int errors = verify_result();
    if (errors == 0) {
        printf("PASS! All %d elements match.\n", APPLICATION_M * APPLICATION_N);
    } else {
        printf("FAIL! %d mismatches out of %d elements.\n",
               errors, APPLICATION_M * APPLICATION_N);
    }

    return (errors == 0) ? 0 : 1;*/
    return 0;
}
