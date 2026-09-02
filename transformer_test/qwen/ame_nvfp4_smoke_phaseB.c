// AME NVFP4 smoke — Phase B.
//
// Full NVFP4 GEMM through the AME path, with Scale scratchpads actually loaded:
//   1. ame_mset_scalea(A_scale_ptr) + ame_mset_scaleb(B_scale_ptr)
//        -> AMEDecoder writes scale_a_base_reg / scale_b_base_reg
//   2. ame_mlae4(TR0, A_ptr, K/2) + ame_mlbe4(TR2, B_ptr, K/2)
//        -> AMEDecoder emits Load micro-inst with Is_A/B_Scale_Work=true and
//           ApplicationScale_A/B populated. TaskController fires ASL + BSL,
//           which pull scale bytes from DRAM into ASSpad/BSSpad.
//   3. ame_mfmacc_s_nvfp4(ACC0, TR0, TR2)
//        -> MTE runs FP4 x FP4 -> FP32 using per-16 block E4M3 scales.
//   4. ame_msce32(ACC0, C_ptr, N*4)
//        -> Store fp32 tile out.
//
// All operand memory is zero-initialized BSS (weights, activations, and scales
// all zero). Expected output: all zeros. Cycles are still meaningful (schedule
// is data-independent).
//
// Tile shape matches CUTE_4Tops_128SCP (Tensor_M=Tensor_N=128, Tensor_K=64),
// which is what testL2Dma1core uses.

#include <stdio.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

#define M 128
#define N 128
#define K 64
#define NVFP4_BLOCK 16

// A / B: [rows][K/2] packed FP4 (2 elements per byte, low nibble first).
static uint8_t A_fp4[M][K / 2] __attribute__((aligned(64))) = {0};
static uint8_t B_fp4[N][K / 2] __attribute__((aligned(64))) = {0};

// Scales: nominal layout is [rows][K/16] E4M3 (M*K/16 = 128*4 = 512B).
// ScaleLoader's MaxRequestIter formula in AScaleLoader.scala:138 uses
// ScaratchpadTensor_K in ReduceVector units, not raw K, which yields more
// bursts than the tight per-element math would predict. Over-provision to
// 2 KiB per side (aligned 64B) so the loader can't wander past the end.
#define SCALE_PAD_BYTES 2048
static uint8_t A_scale[SCALE_PAD_BYTES] __attribute__((aligned(64))) = {0};
static uint8_t B_scale[SCALE_PAD_BYTES] __attribute__((aligned(64))) = {0};

// FP32 accumulator output.
static float C_fp32[M][N] __attribute__((aligned(64))) = {0};

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[AME-NVFP4-PHASEB] start: MxNxK = %dx%dx%d, block=%d\n",
           M, N, K, NVFP4_BLOCK);
    printf("[AME-NVFP4-PHASEB] A_fp4=%p A_scale=%p B_fp4=%p B_scale=%p C=%p\n",
           A_fp4, A_scale, B_fp4, B_scale, C_fp32);

    // 1. Set the Scale scratchpad base vaddrs. These land in AMEDecoder's
    //    scale_a_base_reg / scale_b_base_reg. Each subsequent mlae4/mlbe4
    //    picks them up as the ASL/BSL load address.
    ame_mset_scalea((uint64_t)A_scale);
    ame_mset_scaleb((uint64_t)B_scale);

    // 2. Configure tile shape. CUTE_4Tops_128SCP requires N == Tensor_N == 128
    //    (CDataController.scala:171 asserts full-N). M can be anything up to
    //    Tensor_M; we go full 128 too. K stays at Tensor_K=64.
    ame_settilem(M);
    ame_settilen(N);
    ame_settilek(K);

    uint64_t t0 = rd_cycle();

    // 3. Zero ACC0, then load A/B as packed FP4.
    //    Stride is bytes-per-row = K/2 for packed FP4 (2 elements per byte).
    ame_mzero(ACC0);
    ame_mlae4(TR0, (uint64_t)A_fp4, (uint64_t)(K / 2));
    ame_mlbe4(TR2, (uint64_t)B_fp4, (uint64_t)(K / 2));

    // 4. NVFP4 matmul. func4=0010 under uop=10 tells the decoder to set
    //    ame_datatype=DataTypenvfp4F32; the MTE will multiply FP4 lanes with
    //    the per-16 E4M3 block scales that ASL/BSL just staged.
    ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);

    // 5. Store FP32 tile back to DRAM (via TCM path).
    ame_msce32(ACC0, (uint64_t)C_fp32, (uint64_t)(N * 4));

    while (!ame_is_idle()) { /* spin */ }

    uint64_t t1 = rd_cycle();

    int nonzero = 0;
    for (int i = 0; i < M && !nonzero; i++)
        for (int j = 0; j < N; j++)
            if (C_fp32[i][j] != 0.0f) { nonzero = 1; break; }

    printf("[AME-NVFP4-PHASEB] cycles = %lu\n", (unsigned long)(t1 - t0));
    printf("[AME-NVFP4-PHASEB] output %s\n",
           nonzero ? "NONZERO (unexpected)" : "zero (OK)");
    printf("[AME-NVFP4-PHASEB] done\n");
    return 0;
}
