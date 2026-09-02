// AME NVFP4 smoke — Phase D (numerical golden check).
//
// Phase B proved the plumbing (ASL/BSL fire, no hang). Phase D confirms that
// CUTE's NVFP4 pipeline actually does the FP math correctly by running a
// tile with a known-answer input.
//
// Construction:
//   A_fp4[m][k]      = 1.0  (E2M1 nibble = 0b0010 = 0x2   → packed byte 0x22)
//   B_fp4[n][k]      = 1.0
//   scaleA[m][k/16]  = 1.0  (E4M3 = 0b0_0111_000 = 0x38)
//   scaleB[n][k/16]  = 1.0
//
// Per-element math:
//   C[i][j] = Σ_{k=0..K-1} (A[i,k]*scaleA[i,k/16]) * (B[j,k]*scaleB[j,k/16])
//           = Σ (1.0 * 1.0) * (1.0 * 1.0)
//           = K = 64.0
//
// IEEE-754 fp32 bit pattern for 64.0f = 0x42800000. Bit-exact.
//
// We print / compare via uint32 bit pattern rather than %f because htif_nano
// specs strip float printf support (only %d/%x/%s work). A first Phase D run
// showed empty fields where %.3f used to be — same code, integer prints
// worked, so the compare against EXPECTED_C_BITS is the only source of truth.
//
// Fill the entire scale buffer (2 KiB) with 0x38, not just the logical
// M*(K/16)=512B, because ScaleLoader's MaxRequestIter over-requests (see
// AScaleLoader.scala:138) — we'd rather have any stray reads still land on
// a "1.0" byte than on 0x00 or uninitialized memory.

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ame_nvfp4_ext.h"

// Tile shape: for testL2Dma1core (CUTE_4Tops_128SCP), ReduceWidthByte=64, so
// one ReduceVector holds 512 bits / 4 bits per NVFP4 = 128 FP4 elements per row.
// The AME K granularity is 1 RV, so K must be a multiple of 128 for this config.
// Using exactly one RV (K=128) gives the simplest golden math: C[i][j] = K = 128.
#define M 128
#define N 128
#define K 128
#define NVFP4_BLOCK 16

// Phase D test variants (select at compile time with -DPHASED_VARIANT=N).
// K = 128 (one full RV under CUTE_4Tops_128SCP with ReduceWidthByte=64).
// Each variant's expected C[i][j] = Σ_{k=0..127} A×scaleA × B×scaleB.
//   0 (default) = A/B=1.0, scale=1.0  → 128.0  (0x43000000)
//   1           = A/B=1.0, scale=0.5  →  32.0  (0x42000000)   (K × 0.25)
//   2           = A/B=2.0, scale=1.0  → 512.0  (0x44000000)   (K × 4)
//   3           = A/B=0.5, scale=1.0  →  32.0  (0x42000000)   (K × 0.25)
//
// FP4 E2M1 encoding (per FP4toint.scala: xxxx.x, 00001 == 0.5):
//   0x0=0.0, 0x1=0.5, 0x2=1.0, 0x3=1.5, 0x4=2.0, 0x5=3.0, 0x6=4.0, 0x7=6.0  (sign in bit 3)
// E4M3 encoding (per RawFloat.frome4m3):
//   1.0 = 0b0_0111_000 = 0x38  (exp=7-7=0, sig=1.000)
//   0.5 = 0b0_0110_000 = 0x30  (exp=6-7=-1)

#ifndef PHASED_VARIANT
#define PHASED_VARIANT 0
#endif

#if   PHASED_VARIANT == 0
#define FP4_BYTE         0x22u    // 1.0 packed twice
#define E4M3_BYTE        0x38u    // 1.0
#define EXPECTED_C_BITS  0x43000000u   // 128.0 = 1.0 * 1.0 * 1.0 * 1.0 * K(128)
#define VARIANT_NAME     "A=B=1.0 scale=1.0 -> 128.0"
#elif PHASED_VARIANT == 1
#define FP4_BYTE         0x22u    // 1.0
#define E4M3_BYTE        0x30u    // 0.5
#define EXPECTED_C_BITS  0x42000000u   // 32.0  = 1.0 * 1.0 * 0.5 * 0.5 * K
#define VARIANT_NAME     "A=B=1.0 scale=0.5 -> 32.0"
#elif PHASED_VARIANT == 2
#define FP4_BYTE         0x44u    // 2.0
#define E4M3_BYTE        0x38u    // 1.0
#define EXPECTED_C_BITS  0x44000000u   // 512.0 = 2.0 * 2.0 * 1.0 * 1.0 * K
#define VARIANT_NAME     "A=B=2.0 scale=1.0 -> 512.0"
#elif PHASED_VARIANT == 3
#define FP4_BYTE         0x11u    // 0.5
#define E4M3_BYTE        0x38u    // 1.0
#define EXPECTED_C_BITS  0x42000000u   // 32.0  = 0.5 * 0.5 * 1.0 * 1.0 * K
#define VARIANT_NAME     "A=B=0.5 scale=1.0 -> 32.0"
#else
#error "PHASED_VARIANT must be 0..3"
#endif

#define FP4_ONE_BYTE     FP4_BYTE
#define E4M3_ONE_BYTE    E4M3_BYTE

static uint8_t A_fp4[M][K / 2]  __attribute__((aligned(64)));
static uint8_t B_fp4[N][K / 2]  __attribute__((aligned(64)));
static uint32_t C_fp32[M][N]    __attribute__((aligned(64)));  // stored as raw bits

// Over-provisioned to guard against ScaleLoader over-read (see Phase B notes).
#define SCALE_PAD_BYTES 2048
static uint8_t A_scale[SCALE_PAD_BYTES] __attribute__((aligned(64)));
static uint8_t B_scale[SCALE_PAD_BYTES] __attribute__((aligned(64)));

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    asm volatile ("rdcycle %0" : "=r"(v));
    return v;
}

int main(void)
{
    printf("[AME-NVFP4-PHASED] start: MxNxK = %dx%dx%d, block=%d, variant=%d (%s)\n",
           M, N, K, NVFP4_BLOCK, PHASED_VARIANT, VARIANT_NAME);

    memset(A_fp4,   (int)FP4_ONE_BYTE,  sizeof(A_fp4));
    memset(B_fp4,   (int)FP4_ONE_BYTE,  sizeof(B_fp4));
    memset(A_scale, (int)E4M3_ONE_BYTE, sizeof(A_scale));
    memset(B_scale, (int)E4M3_ONE_BYTE, sizeof(B_scale));
    memset(C_fp32,  0,                  sizeof(C_fp32));
    printf("[AME-NVFP4-PHASED] filled: A/B=%02x scales=%02x expected_bits=0x%08x (fp32 64.0)\n",
           (unsigned)FP4_ONE_BYTE, (unsigned)E4M3_ONE_BYTE, (unsigned)EXPECTED_C_BITS);

    ame_mset_scalea((uint64_t)A_scale);
    ame_mset_scaleb((uint64_t)B_scale);

    ame_settilem(M);
    ame_settilen(N);
    // AMEDecoder clamps csr_mtilek to ReduceGroupSize = Tensor_K/ReduceWidthByte.
    // For CUTE_4Tops_128SCP, that's 1. HW then processes exactly one RV
    // (= 128 FP4 elements at ReduceWidthByte=64) regardless of the raw K
    // requested here. Passing K works because clamp handles the excess.
    ame_settilek(K);

    uint64_t t0 = rd_cycle();

    ame_mzero(ACC0);
    ame_mlae4(TR0, (uint64_t)A_fp4, (uint64_t)(K / 2));
    ame_mlbe4(TR2, (uint64_t)B_fp4, (uint64_t)(K / 2));
    ame_mfmacc_s_nvfp4(ACC0, TR0, TR2);
    ame_msce32(ACC0, (uint64_t)C_fp32, (uint64_t)(N * 4));

    while (!ame_is_idle()) { /* spin */ }

    uint64_t t1 = rd_cycle();

    // Print corners + middle as uint32 bit patterns so we can eyeball whether
    // the pipeline is producing anything sensible before running the full
    // compare. htif_nano's printf supports %x but not %f.
    printf("[AME-NVFP4-PHASED] samples (bits): C[0][0]=%08x C[0][127]=%08x C[64][64]=%08x C[127][127]=%08x\n",
           (unsigned)C_fp32[0][0], (unsigned)C_fp32[0][127],
           (unsigned)C_fp32[64][64], (unsigned)C_fp32[127][127]);

    int errors = 0;
    uint32_t first_wrong_bits = 0;
    int   first_i = -1, first_j = -1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < 10; j++) {
            if (C_fp32[i][j] != EXPECTED_C_BITS) {
                if (errors == 0) {
                    first_wrong_bits = C_fp32[i][j];
                    first_i = i;
                    first_j = j;
                }
                errors++;
            }
        }
    }

    printf("[AME-NVFP4-PHASED] cycles = %lu\n", (unsigned long)(t1 - t0));
    if (errors == 0) {
        printf("[AME-NVFP4-PHASED] PASS: all %d elements == 0x%08x\n",
               M * N, (unsigned)EXPECTED_C_BITS);
    } else {
        printf("[AME-NVFP4-PHASED] FAIL: %d/%d elements wrong. first wrong C[%d][%d]=0x%08x (expected 0x%08x)\n",
               errors, M * N, first_i, first_j,
               (unsigned)first_wrong_bits, (unsigned)EXPECTED_C_BITS);
    }
    printf("[AME-NVFP4-PHASED] done\n");
    return errors == 0 ? 0 : 1;
}
