// tcm_partition_smoke.c — Step-2A smoke test for the TcmCtrl MMIO regmap.
//
// What this test verifies:
//   * The regmap layout is what we expect (TCM_INFO reads back ways/sets/block).
//   * The initial partition matches the compile-time default (tcm_way_count=4).
//   * A shrink write (tcm_way_count 4 → 2) commits the new mask; the busy bit
//     drops before the poll runs out; TCM writes into the surviving low half
//     of the aperture still round-trip; the top half is now out-of-range and
//     the hardware asserts on it (checked in sim, not here — we only stay
//     inside the safe range).
//   * A grow write (2 → 4) also commits (Step 2A: no HW flush — we're
//     running fresh cache so no line has been populated in ways 4-5 by the
//     time we grow; a real workload would need a wbinvd first).
//   * Illegal writes (e.g., 3) set the illegal_last_write flag and leave the
//     partition unchanged.
//
// This test does NOT verify the way-flush semantics — Step 2B will.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TCM_CTRL_BASE  0x22000000ULL
#define TCM_MODE       (TCM_CTRL_BASE + 0x00)
#define TCM_STATUS     (TCM_CTRL_BASE + 0x04)
#define TCM_MASK       (TCM_CTRL_BASE + 0x08)
#define TCM_INFO       (TCM_CTRL_BASE + 0x0C)

#define STATUS_BUSY    0x1
#define STATUS_ILLEGAL 0x2

#define TCM_BASE       0x81000000ULL

static inline uint32_t mmio_r32(uint64_t addr) {
    return *(volatile uint32_t *)(uintptr_t)addr;
}
static inline void mmio_w32(uint64_t addr, uint32_t val) {
    *(volatile uint32_t *)(uintptr_t)addr = val;
}

#define POLL_LIMIT 10000

// Block until busy clears, or POLL_LIMIT expires. Returns 0 on success.
static int wait_ready(void) {
    int t = POLL_LIMIT;
    while ((mmio_r32(TCM_STATUS) & STATUS_BUSY) && t-- > 0) { /* spin */ }
    if (t <= 0) {
        printf("  [ERR] timeout waiting for TCM_STATUS.busy to clear\n");
        return 1;
    }
    return 0;
}

// Poke a handful of strategic 8-byte words in each way's aperture and
// verify round-trip. We touch: first block, one middle block, last block —
// enough to prove the (way, set) mapping is stable, cheap enough for
// simulation (a full-way sweep would be ~500K bytes and take forever).
static int tcm_roundtrip(uint32_t count, uint32_t way_bytes) {
    volatile uint64_t *tcm = (volatile uint64_t *)(uintptr_t)TCM_BASE;
    const uint32_t probe_offsets[3] = {
        0,                          // first block
        (way_bytes / 2) & ~63u,     // middle block (64B-aligned)
        way_bytes - 64              // last block
    };
    // 8-byte pattern that encodes (way, block_id, word_id) so a mismatch
    // pinpoints exactly which cell is wrong.
    #define PAT(w, boff, wi) \
        (((uint64_t)(w)  << 56) | \
         ((uint64_t)(boff) << 16) | \
         ((uint64_t)(wi)  <<  0) | \
         0xC0DEC0DE00000000ULL)

    // Write phase
    for (uint32_t w = 0; w < count; w++) {
        for (int pi = 0; pi < 3; pi++) {
            uint32_t boff = probe_offsets[pi];
            uint64_t off  = (uint64_t)w * way_bytes + boff;
            for (int wi = 0; wi < 8; wi++) {         // 8 words = 64B block
                tcm[(off >> 3) + wi] = PAT(w, boff, wi);
            }
        }
    }
    asm volatile("fence rw, rw" ::: "memory");

    // Read-back phase
    int errors = 0;
    for (uint32_t w = 0; w < count; w++) {
        for (int pi = 0; pi < 3; pi++) {
            uint32_t boff = probe_offsets[pi];
            uint64_t off  = (uint64_t)w * way_bytes + boff;
            for (int wi = 0; wi < 8; wi++) {
                uint64_t exp = PAT(w, boff, wi);
                uint64_t got = tcm[(off >> 3) + wi];
                if (got != exp) {
                    if (errors < 4) {
                        printf("  mismatch way=%u block_off=0x%x word=%d: exp=0x%016lx got=0x%016lx\n",
                               w, boff, wi, exp, got);
                    }
                    errors++;
                }
            }
        }
    }
    #undef PAT
    return errors;
}

int main(void) {
    printf("=== TCM partition smoke test (Step 2A) ===\n");

    // ---- Discover geometry --------------------------------------------------
    uint32_t info      = mmio_r32(TCM_INFO);
    uint32_t ways      =  info        & 0xF;
    uint32_t log2sets  = (info >>  8) & 0xFF;
    uint32_t log2blk   = (info >> 16) & 0xFF;
    uint32_t sets      = 1u << log2sets;
    uint32_t blockB    = 1u << log2blk;
    uint32_t wayBytes  = sets * blockB;
    printf("L2 info: ways=%u sets=%u block=%uB way_bytes=%uB\n",
           ways, sets, blockB, wayBytes);

    // ---- Sanity: initial state --------------------------------------------
    uint32_t init_count = mmio_r32(TCM_MODE) & 0xF;
    uint32_t init_mask  = mmio_r32(TCM_MASK) & 0xFF;
    printf("Initial: tcm_way_count=%u mask=0x%02x\n", init_count, init_mask);
    if (init_count != 4) {
        printf("  [WARN] expected initial count=4 (matches compile-time default), got %u\n",
               init_count);
    }

    // ---- Round-trip within the current TCM range ---------------------------
    printf("[phase1] Round-trip within count=%u\n", init_count);
    if (tcm_roundtrip(init_count, wayBytes) != 0) {
        printf("  FAIL: initial round-trip mismatched\n");
        return 1;
    }
    printf("  PASS\n");

    // ---- Sweep through every legal partition value -------------------------
    // Legal values (Step 2A): {0, 1, 2, 4}. Sequence covers shrink AND grow
    // in both directions. Under "grow-from-top" convention, data at TCM
    // offsets < new_count*way_bytes survives every transition, so we can
    // always round-trip within min(before, after).
    const uint32_t sweep[] = { 4, 2, 1, 2, 4, 1, 0, 1, 4, 0, 4 };
    const int nSweep = sizeof(sweep) / sizeof(sweep[0]);
    for (int step = 0; step < nSweep; step++) {
        uint32_t target = sweep[step];
        printf("[phase2.%d] Transition to count=%u\n", step, target);
        mmio_w32(TCM_MODE, target);
        if (wait_ready()) return 1;
        uint32_t got_count = mmio_r32(TCM_MODE) & 0xF;
        uint32_t got_mask  = mmio_r32(TCM_MASK) & 0xFF;
        if (got_count != target) {
            printf("  FAIL: expected count=%u, got %u (mask=0x%02x)\n",
                   target, got_count, got_mask);
            return 1;
        }
        printf("  count=%u mask=0x%02x\n", got_count, got_mask);
        // Round-trip only if TCM is actually enabled. count=0 disables TCM
        // entirely — any access to the aperture is out-of-range (assertion
        // in TcmSinkA), so we skip.
        if (target > 0) {
            if (tcm_roundtrip(target, wayBytes) != 0) {
                printf("  FAIL: post-transition round-trip mismatched\n");
                return 1;
            }
        }
        printf("  PASS\n");
    }

    // ---- Illegal write -----------------------------------------------------
    printf("[phase3] Illegal write (count=3)\n");
    // Land on a known-legal state first so we can verify the illegal write
    // does NOT change it.
    mmio_w32(TCM_MODE, 4);
    if (wait_ready()) return 1;
    mmio_w32(TCM_MODE, 3);
    // Illegal writes latch a flag but don't set busy. Give one cycle for the
    // MMIO write to propagate before reading status.
    asm volatile("fence rw, rw" ::: "memory");
    uint32_t st = mmio_r32(TCM_STATUS);
    uint32_t ct = mmio_r32(TCM_MODE) & 0xF;
    printf("  after illegal write: status=0x%x count=%u\n", st, ct);
    if (!(st & STATUS_ILLEGAL)) {
        printf("  FAIL: illegal_last_write not set\n");
        return 1;
    }
    if (ct != 4) {
        printf("  FAIL: illegal write must NOT change count, got %u\n", ct);
        return 1;
    }
    // A subsequent legal write must clear the illegal flag.
    mmio_w32(TCM_MODE, 4);
    if (wait_ready()) return 1;
    st = mmio_r32(TCM_STATUS);
    if (st & STATUS_ILLEGAL) {
        printf("  FAIL: legal write did not clear illegal_last_write (status=0x%x)\n", st);
        return 1;
    }
    // Also try count=8 (all-TCM, disallowed in Step 2A) — should be illegal.
    mmio_w32(TCM_MODE, 8);
    asm volatile("fence rw, rw" ::: "memory");
    st = mmio_r32(TCM_STATUS);
    ct = mmio_r32(TCM_MODE) & 0xF;
    if (!(st & STATUS_ILLEGAL) || ct != 4) {
        printf("  FAIL: count=8 should be illegal in Step 2A (status=0x%x count=%u)\n", st, ct);
        return 1;
    }
    // Reset illegal flag with a legal write for a clean exit.
    mmio_w32(TCM_MODE, 4);
    if (wait_ready()) return 1;
    printf("  PASS\n");

    printf("\n=== OVERALL: PASS ===\n");
    return 0;
}
