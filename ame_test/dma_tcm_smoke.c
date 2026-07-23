// dma_tcm_smoke.c — Baremetal smoke test for TcmDmaEngine.
//
// Flow:
//   1. Fill src[] in DRAM with a known pattern.
//   2. Pre-poison the TCM destination window with 0xAA so we can prove the DMA
//      actually overwrote it (and not just that TCM was zero at reset).
//   3. Program the DMA control registers via MMIO.
//   4. Kick CTRL[0]=1 and poll STATUS.done.
//   5. Byte-compare TCM against src[]. Print PASS/FAIL.
//
// Run three sizes to exercise both the single-block path and the FSM loop:
//   - 64B  (one iteration)
//   - 128B (loop once)
//   - 512B (loop 7 times)
//
// Register layout: see section 3.4 of L2_TCM_DMA_Refactor_Plan.md and the
// regmap in src/main/scala/TcmDmaEngine.scala.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define DMA_BASE     0x20000000ULL
#define DMA_SRC_LO   (DMA_BASE + 0x00)
#define DMA_SRC_HI   (DMA_BASE + 0x04)
#define DMA_DST_LO   (DMA_BASE + 0x08)
#define DMA_DST_HI   (DMA_BASE + 0x0C)
#define DMA_LENGTH   (DMA_BASE + 0x10)
#define DMA_CTRL     (DMA_BASE + 0x14)
#define DMA_STATUS   (DMA_BASE + 0x18)
#define DMA_IRQ_CLR  (DMA_BASE + 0x1C)

#define STATUS_BUSY  0x1
#define STATUS_DONE  0x2
#define STATUS_ERR   0x4

#define TCM_BASE     0x81000000ULL

static inline uint32_t mmio_r32(uint64_t addr) {
    return *(volatile uint32_t *)(uintptr_t)addr;
}
static inline void mmio_w32(uint64_t addr, uint32_t val) {
    *(volatile uint32_t *)(uintptr_t)addr = val;
}

#define POLL_LIMIT 200000

// Source pattern in DRAM. Sized for the largest transfer we run.
static uint8_t __attribute__((aligned(64))) src[512];

// Fill src[i] with (i*p + q) mod 256 — a distinctive per-byte pattern.
static void fill_pattern(int len, uint8_t p, uint8_t q) {
    for (int i = 0; i < len; i++) src[i] = (uint8_t)(i * p + q);
}

// Reset TCM window to 0xAA so any surviving 0xAA byte flags a DMA miss.
static void poison_tcm(int len) {
    volatile uint8_t *tcm = (volatile uint8_t *)(uintptr_t)TCM_BASE;
    for (int i = 0; i < len; i++) tcm[i] = 0xAA;
    asm volatile("fence rw, rw" ::: "memory");
}

// Issue one DMA transfer and poll to completion. Returns 0 on success.
static int run_dma(uint64_t src_addr, uint64_t dst_addr, uint32_t length) {
    // Wait for engine idle.
    while (mmio_r32(DMA_STATUS) & STATUS_BUSY) { /* spin */ }

    mmio_w32(DMA_SRC_LO, (uint32_t)(src_addr));
    mmio_w32(DMA_SRC_HI, (uint32_t)(src_addr >> 32));
    mmio_w32(DMA_DST_LO, (uint32_t)(dst_addr));
    mmio_w32(DMA_DST_HI, (uint32_t)(dst_addr >> 32));
    mmio_w32(DMA_LENGTH, length);
    asm volatile("fence rw, rw" ::: "memory");

    mmio_w32(DMA_CTRL, 0x1);

    int t = POLL_LIMIT;
    uint32_t status;
    do {
        status = mmio_r32(DMA_STATUS);
        if (status & STATUS_ERR) {
            printf("  [ERR] STATUS.err set (status=0x%x)\n", status);
            return 1;
        }
        t--;
    } while (!(status & STATUS_DONE) && t > 0);

    if (t <= 0) {
        printf("  [ERR] DMA timeout after %d polls (status=0x%x)\n", POLL_LIMIT, status);
        return 1;
    }

    // Clear done latch + reset CTRL so the next kick sees a rising edge.
    mmio_w32(DMA_IRQ_CLR, 0x1);
    mmio_w32(DMA_CTRL, 0x0);
    return 0;
}

static int verify(int len) {
    volatile uint8_t *tcm = (volatile uint8_t *)(uintptr_t)TCM_BASE;
    asm volatile("fence rw, rw" ::: "memory");
    int errors = 0;
    for (int i = 0; i < len; i++) {
        uint8_t got = tcm[i];
        uint8_t exp = src[i];
        if (got != exp) {
            if (errors < 8) {
                printf("  mismatch @ %3d: expected 0x%02x, got 0x%02x\n", i, exp, got);
            }
            errors++;
        }
    }
    return errors;
}

static int one_test(const char *tag, int len, uint8_t p, uint8_t q) {
    printf("[%s] len=%d bytes\n", tag, len);
    fill_pattern(len, p, q);
    poison_tcm(len);

    if (run_dma((uint64_t)(uintptr_t)src, TCM_BASE, (uint32_t)len)) {
        printf("[%s] FAIL (DMA did not complete cleanly)\n", tag);
        return 1;
    }

    int errors = verify(len);
    if (errors == 0) {
        printf("[%s] PASS\n", tag);
        return 0;
    }
    printf("[%s] FAIL: %d/%d mismatched bytes\n", tag, errors, len);
    return 1;
}

int main(void) {
    printf("=== TCM DMA smoke test ===\n");
    // htif_nano's iprintf doesn't support %llx; split the 64-bit values into
    // two 32-bit halves so the console output stays readable.
    printf("MMIO base = 0x%08x%08x, TCM base = 0x%08x%08x\n",
           (uint32_t)(DMA_BASE >> 32), (uint32_t)(DMA_BASE & 0xFFFFFFFFu),
           (uint32_t)(TCM_BASE >> 32), (uint32_t)(TCM_BASE & 0xFFFFFFFFu));
    printf("src[] @ %p\n", src);

    int fail = 0;
    fail |= one_test("test1_64B",   64,  0x11, 0x55);
    fail |= one_test("test2_128B", 128,  0x07, 0x03);
    fail |= one_test("test3_512B", 512,  0x1D, 0x2A);

    if (fail) {
        printf("\n=== OVERALL: FAIL ===\n");
        return 1;
    }
    printf("\n=== OVERALL: PASS ===\n");
    return 0;
}
