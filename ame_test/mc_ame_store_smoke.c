// 4-core AME-store-only smoke test.
//
// Bracketed between two verified-good pieces:
//   * CPU load/store to TCM was proven OK by mc_tcm_rw_smoke.
//   * DMA to TCM was proven OK by the diagnostics from the mc GEMM run.
// So this test isolates the AME store path (Cute2TL → atlNode →
// tlMasterXbar → tcmNode) under 4-core concurrent access, with no other
// AME activity (no loads, no computes).
//
// Each hart:
//   1. Configures its private TCM to 4 ways
//   2. Uses CPU stores to pre-fill 0x81100000..+SIZE with 0xdeadbeef, then fence
//   3. barrier — all harts now have poisoned TCM
//   4. ame_mzero(ACC0) — zeros the AME accumulator (no memory traffic)
//   5. ame_msce32(ACC0, 0x81100000, stride) — writes TILE_M×TILE_N int32 zeros
//      through the AME store engine to its private TCM
//   6. ame_fence + fence — wait for all AME ops to drain
//   7. barrier — synchronize completion so we compare in a stable state
//   8. CPU reads back and counts how many words are still 0xdeadbeef (i.e.
//      not overwritten by the AME store)
//   9. serialised summary print
//
// A PASS means AME store to TCM works on all 4 cores concurrently.
// A HANG at step 5/6 means the AME store engine hasn't returned — matches
// the GEMM symptom and is the minimal reproducer.
// A "still 0xdeadbeef" mismatch would mean the AME store fired but wrote
// somewhere else (address routing wrong).

#include <stdio.h>
#include <stdint.h>
#include "ame.h"

#define N_CORES  4

// Match the tile geometry the existing matmul kernels use so we exercise
// the same store path. TILE_M=TILE_N=128, int32 -> 128*128*4 = 64 KiB
// per hart. If needed to shrink further for iteration speed, drop these.
#define TILE_M   128
#define TILE_N   128

#define C_BASE_ADDR  0x81100000UL
#define POISON       0xdeadbeefU

static inline uint64_t rd_mhartid(void) {
    uint64_t v;
    __asm__ __volatile__("csrr %0, mhartid" : "=r"(v));
    return v;
}

static inline uint64_t rd_cycle(void) {
    uint64_t v;
    __asm__ __volatile__("rdcycle %0" : "=r"(v));
    return v;
}

static void __attribute__((noinline)) barrier(void) {
    static volatile int sense;
    static volatile int count;
    static __thread int threadsense;

    __sync_synchronize();
    threadsense = !threadsense;
    if (__sync_fetch_and_add(&count, 1) == N_CORES - 1) {
        count = 0;
        sense = threadsense;
    } else {
        while (sense != threadsense) { /* spin */ }
    }
    __sync_synchronize();
}

void __main(void) {
    uint64_t hid = rd_mhartid();
    if (hid >= N_CORES) { while (1) { } }

    if (ame_tcm_config(4) != 0) {
        printf("[core %u] FAIL: TCM config\n", (unsigned)hid);
        return;
    }

    if (hid == 0) {
        printf("MC-AME-STORE: 4 cores, TILE=%dx%d int32 (=%d KiB per hart)\n",
               TILE_M, TILE_N, TILE_M * TILE_N * 4 / 1024);
    }

    volatile uint32_t *c = (volatile uint32_t *)C_BASE_ADDR;
    const int n_words = TILE_M * TILE_N;

    // --- 1. Poison target region with CPU stores (known-good path) --------
    uint64_t p_t0 = rd_cycle();
    for (int i = 0; i < n_words; i++) c[i] = POISON;
    __asm__ __volatile__("fence rw, rw" ::: "memory");
    uint64_t p_t1 = rd_cycle();

    barrier();

    // --- 2. Configure AME tile shape and zero the accumulator -------------
    ame_settilem(TILE_M);
    ame_settilen(TILE_N);
    ame_settilek(64);
    ame_mzero(ACC0);

    // --- 2b. Force at least one real compute microinst so the store's
    // Compute_Micro_Inst_FIFO_Index dependency points at a real slot that
    // will actually be marked Ready_GO. Without this, TaskController's
    // Store issue state machine hangs in IDLE forever waiting on a compute
    // slot that was never occupied.
    //
    // Load garbage into TR0/TR2 (contents don't matter — we only need the
    // subsequent mmacc to actually enter the compute FIFO). We reuse the
    // poisoned C region as a scratch load source; A/B are undefined here
    // so the *result* is nonsense, but the store is what we care about.
    ame_mlae8(TR0,  (uint64_t)C_BASE_ADDR, 512);
    ame_mlbe8(TR2,  (uint64_t)C_BASE_ADDR, 512);
    ame_mmacc_w_b(ACC0, TR0, TR2);   // ACC0 will be junk but non-zero;
                                     // that's fine — we're testing the store path

    // --- 3. THE thing we're testing: AME store to TCM ---------------------
    const uint64_t c_stride = TILE_N * sizeof(int32_t);
    uint64_t s_t0 = rd_cycle();
    ame_msce32(ACC0, (uint64_t)C_BASE_ADDR, c_stride);
    ame_fence();                                          // AME-side drain
    __asm__ __volatile__("fence rw, rw" ::: "memory");    // CPU-side fence
    uint64_t s_t1 = rd_cycle();

    barrier();

    // --- 4. Verify with CPU reads -----------------------------------------
    // We only care that the AME store *fired* and overwrote POISON. The
    // computed values (from mmacc on garbage inputs) are undefined, so
    // "not still POISON" is our success criterion.
    uint64_t v_t0 = rd_cycle();
    int not_written = 0;         // words still holding POISON
    int first_bad_i = -1;
    for (int i = 0; i < n_words; i++) {
        if (c[i] == POISON) {
            not_written++;
            if (first_bad_i < 0) { first_bad_i = i; }
        }
    }
    uint64_t v_t1 = rd_cycle();

    barrier();

    // --- 5. Serialised per-core summary -----------------------------------
    for (int i = 0; i < N_CORES; i++) {
        if ((int)hid == i) {
            if (not_written == 0) {
                printf("[core %u] PASS   poison=%u  msce=%u  verify=%u cyc  (%d words overwritten)\n",
                       (unsigned)hid,
                       (unsigned)(p_t1 - p_t0),
                       (unsigned)(s_t1 - s_t0),
                       (unsigned)(v_t1 - v_t0),
                       n_words);
            } else {
                printf("[core %u] FAIL   not_written=%d first_still_POISON @ i=%d  msce=%u cyc\n",
                       (unsigned)hid, not_written, first_bad_i,
                       (unsigned)(s_t1 - s_t0));
            }
        }
        barrier();
    }

    if (hid > 0) { while (1) { } }
}

int main(void) {
    __main();
    return 0;
}
