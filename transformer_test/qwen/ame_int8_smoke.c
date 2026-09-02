// AME + INT8 smoke test — single 64x64x64 INT8 (W8A8) GEMM on the AME path.
//
// Purpose: establish a working baseline on the AME instruction path (CUSTOM1,
// opcode 0x2B) before we extend AME with NVFP4 support. The current AME
// decoder does not know about NVFP4 (ElementDataType code 9) and forcibly
// disables the Scale subsystem (AMEDecoder.scala:195-223), so we cannot run
// NVFP4 on AME yet. This smoke gives us a "known-good" AME reference that
// exercises tile load / matmul / store / cycle counting through the AME
// pipeline; the hardware NVFP4 work then targets making a matching NVFP4
// smoke pass on this same path.
//
// Data: zero-initialized in BSS. INT8 x INT8 -> INT32 accumulator, all zeros
// expected in the output. Cycles are still meaningful (data-independent).

#include <stdio.h>
#include <stdint.h>
#include "../../ame_test/ame.h"

#define M 64
#define N 64
#define K 64

static int8_t  A_i8[M][K] __attribute__((aligned(64))) = {0};
static int8_t  B_i8[N][K] __attribute__((aligned(64))) = {0};
static int32_t C_i32[M][N] __attribute__((aligned(64))) = {0};

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[AME-INT8-SMOKE] start: MxNxK = %dx%dx%d\n", M, N, K);

    ame_settilem(M);
    ame_settilen(N);
    ame_settilek(K);

    uint64_t t0 = rd_cycle();

    ame_mzero(ACC0);
    ame_mlae8(TR0, (uint64_t)A_i8, (uint64_t)K);              // A stride = K bytes/row
    ame_mlbe8(TR2, (uint64_t)B_i8, (uint64_t)K);              // B stride = K bytes/row
    ame_mmacc_w_b(ACC0, TR0, TR2);                             // ACC0 += A * B^T
    ame_msce32(ACC0, (uint64_t)C_i32, (uint64_t)(N * 4));      // store INT32

    while (!ame_is_idle()) { /* spin */ }

    uint64_t t1 = rd_cycle();

    int nonzero = 0;
    for (int i = 0; i < M && !nonzero; i++)
        for (int j = 0; j < N; j++)
            if (C_i32[i][j] != 0) { nonzero = 1; break; }

    printf("[AME-INT8-SMOKE] cycles = %lu\n", (unsigned long)(t1 - t0));
    printf("[AME-INT8-SMOKE] output %s\n", nonzero ? "NONZERO (unexpected)" : "zero (OK)");
    printf("[AME-INT8-SMOKE] done\n");
    return 0;
}
