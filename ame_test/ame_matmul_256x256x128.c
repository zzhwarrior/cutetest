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

// Input matrices (int8), row-major - provided by matmul_value header
// Output matrix (int32) - provided by matmul_value header


int main(void) {
    printf("AME Test: %dx%dx%d INT8 GEMM (tiled %dx%dx%d, double-buffered)\n",
           APPLICATION_M, APPLICATION_N, APPLICATION_K,
           TILE_M, TILE_N, TILE_K);

    //for (int i = 0; i < APPLICATION_M; i++)
    //for (int j = 0; j < APPLICATION_N; j++)
        //c[0][j] =0;
    printf("a base: %p (expected 0x81008000 for TCM build)\n", (void *)a);
    printf("b base: %p (expected 0x81000000 for TCM build)\n", (void *)b);
    printf("c base: %p (expected 0x82000000)\n", (void *)c);
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
            uint64_t c_base = (uint64_t)&c[mt * TILE_M][nt * TILE_N];
            if(acct == 0){
                ame_mzero(ACC0);
                for (int kt = 0; kt < TILES_K; kt++) {
                    // A tile: row block mt, col block kt -> &a[mt*TILE_M][kt*TILE_K]
                    // B tile: row block nt, col block kt -> &b[nt*TILE_N][kt*TILE_K]
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mlbe8(TR2, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mmacc_w_b(ACC0, TR0, TR2);
                        //asm volatile("fence" ::: "memory");
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mlbe8(TR3, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mmacc_w_b(ACC0, TR1, TR3);
                        //asm volatile("fence" ::: "memory");
                    }
                    //ame_fence();
                }
                ame_msce32(ACC0, c_base, c_stride);
                //asm volatile("fence" ::: "memory");
                acct++;
            }
            else{
                ame_mzero(ACC1);
                for (int kt = 0; kt < TILES_K; kt++) {
                    int cur = kt & 1;
                    if (cur == 0) {
                        ame_mlae8(TR0, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mlbe8(TR2, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mmacc_w_b(ACC1, TR0, TR2);
                        //asm volatile("fence" ::: "memory");
                    } else {
                        ame_mlae8(TR1, (uint64_t)&a[mt * TILE_M][kt * TILE_K], a_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mlbe8(TR3, (uint64_t)&b[nt * TILE_N][kt * TILE_K], b_stride);
                        //asm volatile("fence" ::: "memory");
                        ame_mmacc_w_b(ACC1, TR1, TR3);
                        //asm volatile("fence" ::: "memory");
                    }
                }
                ame_msce32(ACC1, c_base, c_stride);
                //asm volatile("fence" ::: "memory");
                acct--;
            }
        }
    }
    asm volatile("fence" ::: "memory");
    // Wait for all AME operations to complete
    //while (!ame_is_idle()) {}

    asm volatile("rdcycle %0" : "=r"(cycle_end));
    printf("AME GEMM done in %lu cycles\n", cycle_end - cycle_start);
    int errors = 0;
     for (int i = 0; i < APPLICATION_M; i++) {
        for (int j = 0; j < 10; j++) {
            if (c[0][j] != c_ref[0][j]) {
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
        printf("FAIL! mismatches.\n");
    }
    return 0;
}
