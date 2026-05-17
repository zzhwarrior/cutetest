// AME Test: 256x256x128 INT8 Matrix Multiplication
// Tiled: M=256/128=2, N=256/128=2, K=128/64=2 -> 8 tiles total
// Double-buffered: tr0/tr2 (bank0) and tr1/tr3 (bank1) alternate each K step
// so Load(next) overlaps with Compute(current).

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ame.h"
#include "matmul_value_mnk_256_256_128.h"
#include "matmul_cref_256_256_128.h"
#define APPLICATION_M  256
#define APPLICATION_N  256
#define APPLICATION_K  128
#define TILE_M         128
#define TILE_N         128
#define TILE_K         64
#define TILES_M        (APPLICATION_M / TILE_M)   // 2
#define TILES_N        (APPLICATION_N / TILE_N)   // 2
#define TILES_K        (APPLICATION_K / TILE_K)   // 2

// Input matrices (int8), row-major
static int8_t  A[APPLICATION_M][APPLICATION_K] __attribute__((aligned(64)));
static int8_t  B[APPLICATION_N][APPLICATION_K] __attribute__((aligned(64)));
// Output matrix (int32), placed at fixed address to avoid tohost overlap


int main(void) {
    printf("AME Test: %dx%dx%d INT8 GEMM (tiled %dx%dx%d, double-buffered)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K,
           TILE_M, TILE_N, TILE_K);

    //for (int i = 0; i < APPLICATION_M; i++)
        //for (int j = 0; j < APPLICATION_N; j++)
            //c[i][j] =0;
    printf("init done\n");

    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(TILE_K);

    uint64_t a_stride = APPLICATION_K * sizeof(int8_t);   // 128 bytes per row
    uint64_t b_stride = APPLICATION_K * sizeof(int8_t);   // 128 bytes per row
    uint64_t c_stride = APPLICATION_N * sizeof(int32_t);  // 1024 bytes per row

    uint64_t cycle_start, cycle_end;
    asm volatile("rdcycle %0" : "=r"(cycle_start));
    int acct = 0;
    for (int mt = 0; mt < TILES_M; mt++) {
        for (int nt = 0; nt < TILES_N; nt++) {
            // Zero accumulator (C bank 0, ACC0 used throughout)
            if(acct == 0){
                ame_mzero(ACC0);
            }
            else{
                ame_mzero(ACC1);
            }
            for (int kt = 0; kt < TILES_K; kt++) {
                int cur = kt & 1;      
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a[mt * TILE_M][kt], a_stride);
                        ame_mlbe8(TR2, (uint64_t)&b[nt * TILE_N][kt], b_stride);
                        ame_mmacc_w_b(ACC0, TR0, TR2);
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a[mt * TILE_M][kt], a_stride);
                        ame_mlbe8(TR3, (uint64_t)&b[nt * TILE_N][kt], b_stride);
                        ame_mmacc_w_b(ACC1, TR1, TR3);
                    }
            }
            uint64_t c_base = (uint64_t)&c[mt * TILE_M][nt * TILE_N];
            if(acct == 0){
                ame_msce32(ACC0, c_base, c_stride);
                acct++;
            }
            else{
                ame_msce32(ACC1, c_base, c_stride);
                acct--;
            }
            
        }
    }
    printf("issue all!\n");
      // Wait for all AME operations to complete
    while (!ame_is_idle()) {}

    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);
    int errors = 0;
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
    return (errors == 0) ? 0 : 1;
}
