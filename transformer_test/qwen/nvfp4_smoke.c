// NVFP4 smoke test — single 64x64x64 NVFP4 matmul.
// Purpose: validate the software→hardware plumbing for element_type=9 (NVFP4)
// and the ScaleA/ScaleB config funct codes (7/8). No cutetest binary currently
// exercises this path, so run this before qwen_full_attn_nvfp4.
//
// Weights, scales, and inputs are all-zero (BSS), so the expected FP32 output
// is all zero. This is deliberate: the smoke test only validates that the
// hardware accepts the instruction sequence without hang / trap. Cycle count
// is reported for reference.

#include <stdio.h>
#include <stdint.h>
#include "cuteMarcoinstHelper.h"

#define M 64
#define N 64
#define K 64
#define NVFP4_BLOCK 16          // matches CUTEParameters.scala:1082 (NVFP4 group size)

// A / B are FP4 packed 2/byte. Layout: A is [M][K/2], B is [N][K/2].
// low nibble = element k, high nibble = element k+1 (matches FP4toint.scala:1146).
static uint8_t A_fp4[M][K / 2] __attribute__((aligned(64))) = {0};
static uint8_t B_fp4[N][K / 2] __attribute__((aligned(64))) = {0};

// A_scale / B_scale are E4M3 (8-bit FP), one per NVFP4_BLOCK along K.
// Layout: A_scale is [M][K/NVFP4_BLOCK], B_scale is [N][K/NVFP4_BLOCK].
static uint8_t A_scale[M][K / NVFP4_BLOCK] __attribute__((aligned(64))) = {0};
static uint8_t B_scale[N][K / NVFP4_BLOCK] __attribute__((aligned(64))) = {0};

// FP32 accumulator output.
static float C_fp32[M][N] __attribute__((aligned(64))) = {0};

int main(void)
{
    printf("[NVFP4-SMOKE] start: MxNxK = %dx%dx%d, block=%d\n", M, N, K, NVFP4_BLOCK);

    uint64_t t0 = mrdcycle();

    uint64_t task_id = issue_cute_matmul_marco_inst_nvfp4(
        (uint64_t)A_fp4, K / 2,          // A base + M-stride in bytes
        (uint64_t)B_fp4, K / 2,          // B base + M-stride in bytes
        0, 0,                             // no bias (D tensor)
        (uint64_t)C_fp32, N * 4,          // C base + M-stride (FP32)
        (uint64_t)A_scale, (uint64_t)B_scale,
        M, N, K,
        TaskTypeTensorZeroLoad,           // zero-init accumulator
        0,                                // no transpose
        0);                               // matmul m-index

    CUTE_TASK_END(task_id);

    uint64_t t1 = mrdcycle();

    // Sanity: with all-zero inputs, output must be zero. Any non-zero suggests
    // uninitialized memory or a decode bug.
    int nonzero = 0;
    for (int i = 0; i < M && nonzero == 0; i++) {
        for (int j = 0; j < N; j++) {
            if (C_fp32[i][j] != 0.0f) { nonzero = 1; break; }
        }
    }

    printf("[NVFP4-SMOKE] cycles = %lu\n", (unsigned long)(t1 - t0));
    printf("[NVFP4-SMOKE] output %s\n", nonzero ? "NONZERO (unexpected)" : "zero (OK)");
    printf("[NVFP4-SMOKE] done\n");
    return 0;
}
