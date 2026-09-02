// AME NVFP4 smoke — Phase A.
//
// Purpose: validate that the new msetscalea / msetscaleb instructions are
// (a) accepted by the AME cmd FIFO without hang, (b) decoded and dequeued,
// and (c) update AMEDecoder's scale sticky registers to the values we pass.
//
// Phase A does NOT wire scale into the load / matmul data path yet. So the
// matmul in this test is a plain INT8 GEMM (same body as ame_int8_smoke.c);
// its purpose is only to prove that config instructions don't stall or
// disturb subsequent compute. Look for these markers in the sim log:
//
//   [AME-DEC ...] MSET_SCALEA: rs1=<A_SCALE_MARKER> -> scale_a_base_reg
//   [AME-DEC ...] MSET_SCALEB: rs1=<B_SCALE_MARKER> -> scale_b_base_reg
//   [AME-DEC ...] CONFIG: funct=... scaleA=<A_SCALE_MARKER> scaleB=<B_SCALE_MARKER>
//
// Expected: [AME-INT8-SMOKE] output zero (OK); cycles printed.

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

// Tile size must match the target config's Tensor_M/N/K to avoid the
// CDataController assert `ScaratchpadWorkingTensor_N === Tensor_N.U` at
// CDataController.scala:171 (N must equal the full scratchpad width).
//   testL2Dma1core  → CUTE_4Tops_128SCP (Tensor_M=Tensor_N=128, Tensor_K=64)
//   CUTE2TopsSCP64Config → CUTE_2Tops_64SCP  (Tensor_M=Tensor_N=Tensor_K=64)
// Choose SMOKE_TILE_128 at compile time; default matches testL2Dma1core.
#ifndef SMOKE_TILE_64
#define M 128
#define N 128
#define K 64
#else
#define M 64
#define N 64
#define K 64
#endif

// Sentinel scale-base vaddrs. These are arbitrary — Phase A doesn't read
// from them, we only verify the value round-trips through AMEDecoder.
#define A_SCALE_MARKER 0xDEADBEEF00000010ULL
#define B_SCALE_MARKER 0xDEADBEEF00000020ULL

static int8_t  A_i8[M][K]  __attribute__((aligned(64))) = {0};
static int8_t  B_i8[N][K]  __attribute__((aligned(64))) = {0};
static int32_t C_i32[M][N] __attribute__((aligned(64))) = {0};

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[AME-NVFP4-PHASEA] start\n");

    // -- Fire the two new config instructions.
    // Phase A: AMEDecoder will accept, dequeue in 1 cycle, and write the
    // scale_a_base_reg / scale_b_base_reg. No downstream effect yet.
    ame_mset_scalea(A_SCALE_MARKER);
    ame_mset_scaleb(B_SCALE_MARKER);

    // -- Run a plain INT8 GEMM to confirm downstream is undisturbed.
    ame_settilem(M);
    ame_settilen(N);
    ame_settilek(K);

    uint64_t t0 = rd_cycle();

    ame_mzero(ACC0);
    ame_mlae8(TR0, (uint64_t)A_i8, (uint64_t)K);
    ame_mlbe8(TR2, (uint64_t)B_i8, (uint64_t)K);
    ame_mmacc_w_b(ACC0, TR0, TR2);
    ame_msce32(ACC0, (uint64_t)C_i32, (uint64_t)(N * 4));

    while (!ame_is_idle()) { /* spin */ }

    uint64_t t1 = rd_cycle();

    int nonzero = 0;
    for (int i = 0; i < M && !nonzero; i++)
        for (int j = 0; j < N; j++)
            if (C_i32[i][j] != 0) { nonzero = 1; break; }

    printf("[AME-NVFP4-PHASEA] cycles = %lu\n", (unsigned long)(t1 - t0));
    printf("[AME-NVFP4-PHASEA] output %s\n", nonzero ? "NONZERO (unexpected)" : "zero (OK)");
    printf("[AME-NVFP4-PHASEA] done\n");
    return 0;
}
