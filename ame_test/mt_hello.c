// Self-contained mt-hello: same idea as chipyard/tests/mt-hello.c but with
// no dependency on riscv-pk/encoding.h. Used to sanity-check that a
// 4-core config (e.g. testL2Dma) boots all harts, they can printf, and
// the mt-hello barrier synchronises them correctly.

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#define N_CORES 4

static inline uint64_t rd_mhartid(void) {
    uint64_t v;
    __asm__ __volatile__("csrr %0, mhartid" : "=r"(v));
    return v;
}

static inline uint64_t rd_marchid(void) {
    uint64_t v;
    __asm__ __volatile__("csrr %0, marchid" : "=r"(v));
    return v;
}

static const char *march_name(uint64_t marchid) {
    switch (marchid) {
    case 1:  return "rocket";
    case 2:  return "sonicboom";
    case 5:  return "spike";
    default: return "unknown";
    }
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

    // Park any hart the config gave us beyond our expected count.
    if (hid >= N_CORES) { while (1) { } }

    const char *march = march_name(rd_marchid());

    for (uint64_t i = 0; i < N_CORES; i++) {
        if (hid == i) {
            printf("Hello world from core %lu, a %s\n",
                   (unsigned long)hid, march);
        }
        barrier();
    }

    if (hid > 0) { while (1) { /* only hart 0 exits cleanly via HTIF */ } }
}

int main(void) {
    __main();
    return 0;
}
