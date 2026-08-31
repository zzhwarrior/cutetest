// 4-core TCM read/write smoke test.
//
// Purpose: prove that plain CPU loads/stores against each tile's private TCM
// aperture at 0x81000000 work correctly when all 4 harts hammer it in
// parallel. This bypasses DMA and the AME store engine, so if it PASSes we
// know the L2 tcmNode path itself is fine under concurrency, and the
// remaining GEMM hang is specifically an AME-store issue.
//
// Each hart:
//   1. Configures its private TCM to 4 ways
//   2. Writes N_WORDS uint64_t entries starting at 0x81000000 with a
//      per-hart pattern (pattern = 0xA000_0000_0000_0000 | (hid<<56) | i)
//   3. Reads them back and verifies
//   4. Prints its own error count. Serialised via barrier so lines don't
//      interleave over HTIF.
//
// If PASS on all 4 cores: TCM path works under concurrent CPU access.
// If any hart hangs: something is wrong with plain TCM access, not AME.
// If PASS but cross-tile pattern check fails (unlikely): TCM isn't actually
// per-tile private and the config is wrong.

#include <stdio.h>
#include <stdint.h>
#include "ame.h"          // ame_tcm_config, ame_tcm_get_*

#define N_CORES  4
#define N_WORDS  1024     // 1024 * 8 = 8 KiB per hart; all in top TCM way

#define TCM_BASE 0x81000000UL

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

// Deterministic pattern each hart writes into its own TCM.
static inline uint64_t pattern(uint64_t hid, uint64_t i) {
    return (0xA000000000000000ULL) | (hid << 56) | (i & 0x00FFFFFFFFFFFFFFULL);
}

void __main(void) {
    uint64_t hid = rd_mhartid();
    if (hid >= N_CORES) { while (1) { } }

    // Each hart opens its own TCM (private per-tile).
    if (ame_tcm_config(4) != 0) {
        printf("[core %u] FAIL: could not configure TCM to 4 ways\n", (unsigned)hid);
        return;
    }

    if (hid == 0) {
        uint32_t info = ame_tcm_get_info();
        printf("MC-TCM RW: %d cores, %d uint64 per hart (%d KiB)\n",
               N_CORES, N_WORDS, (int)(N_WORDS * 8 / 1024));
        printf("TCM info: ways=%u log2sets=%u log2blk=%u\n",
               info & 0xF, (info >> 8) & 0xFF, (info >> 16) & 0xFF);
    }

    barrier();

    volatile uint64_t *tcm = (volatile uint64_t *)TCM_BASE;

    // ---- WRITE phase (parallel) --------------------------------------
    uint64_t w_t0 = rd_cycle();
    for (int i = 0; i < N_WORDS; i++) {
        tcm[i] = pattern(hid, (uint64_t)i);
    }
    __asm__ __volatile__("fence rw, rw" ::: "memory");
    uint64_t w_t1 = rd_cycle();
    uint64_t w_cyc = w_t1 - w_t0;

    barrier();

    // ---- READ + VERIFY phase (parallel) ------------------------------
    uint64_t r_t0 = rd_cycle();
    int errors = 0;
    uint64_t first_bad_i = 0, first_bad_got = 0, first_bad_want = 0;
    for (int i = 0; i < N_WORDS; i++) {
        uint64_t got  = tcm[i];
        uint64_t want = pattern(hid, (uint64_t)i);
        if (got != want) {
            if (errors == 0) {
                first_bad_i    = (uint64_t)i;
                first_bad_got  = got;
                first_bad_want = want;
            }
            errors++;
        }
    }
    __asm__ __volatile__("fence rw, rw" ::: "memory");
    uint64_t r_t1 = rd_cycle();
    uint64_t r_cyc = r_t1 - r_t0;

    barrier();

    // ---- Serialised summary print ------------------------------------
    for (int i = 0; i < N_CORES; i++) {
        if ((int)hid == i) {
            if (errors == 0) {
                printf("[core %u] PASS   write=%u cyc  read=%u cyc  (bytes=%u)\n",
                       (unsigned)hid,
                       (unsigned)w_cyc, (unsigned)r_cyc,
                       (unsigned)(N_WORDS * 8));
            } else {
                printf("[core %u] FAIL   errors=%d  first bad @ i=%u got=%016lx want=%016lx  w=%u r=%u cyc\n",
                       (unsigned)hid, errors,
                       (unsigned)first_bad_i,
                       (unsigned long)first_bad_got,
                       (unsigned long)first_bad_want,
                       (unsigned)w_cyc, (unsigned)r_cyc);
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
