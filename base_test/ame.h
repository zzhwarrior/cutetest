#ifndef AME_H
#define AME_H

#include <stdint.h>

// ============================================================
// AME (Attached Matrix Extension) Instruction Encoding
// Based on RISC-V Matrix Specification Proposal v0.6.0
// Major opcode: custom-1 = 0101011 (0x2B)
// ============================================================

#define AME_OPCODE 0x2B  // custom-1: 0101011

// uop field [27:26]
#define UOP_CONFIG  0  // 00: configuration
#define UOP_LDST    1  // 01: load/store
#define UOP_MATMUL  2  // 10: matrix multiplication
#define UOP_MISC    3  // 11: misc

// d_size / s_size encoding [1:0]
#define ESIZE_8   0  // 00: 8-bit
#define ESIZE_16  1  // 01: 16-bit
#define ESIZE_32  2  // 10: 32-bit
#define ESIZE_64  3  // 11: 64-bit

// Matrix register indices (md/ms1/ms2) [2:0]
#define TR0   0  // 000
#define TR1   1  // 001
#define TR2   2  // 010
#define TR3   3  // 011
#define ACC0  4  // 100
#define ACC1  5  // 101
#define ACC2  6  // 110
#define ACC3  7  // 111

// Load/Store func4 [31:28]
#define FUNC4_MLA   0x0  // left matrix A load/store
#define FUNC4_MLB   0x1  // right matrix B load/store
#define FUNC4_MLC   0x2  // output matrix C load/store
#define FUNC4_MLAT  0x4  // left matrix A transposed
#define FUNC4_MLBT  0x5  // right matrix B transposed
#define FUNC4_MLCT  0x6  // output matrix C transposed
#define FUNC4_MLE   0x8  // whole register load/store

// MatMul func4 [31:28]
#define FUNC4_FLOAT   0x0  // float matrix multiplication
#define FUNC4_INT     0x1  // integer matrix multiplication
#define FUNC4_INTHYB  0x2  // integer hybrid-precision

// MISC func4 [31:28]
#define FUNC4_MZERO   0x0  // mzero
#define FUNC4_MMOV    0x1  // mmov.mm

// ============================================================
// Instruction encoding macros
// ============================================================

// --- Configuration Instructions ---
// Format: func4[31:28] | uop=00[27:26] | imm_sel[25] | imm10/rs1[24:15] | func3=000[14:12] | rd=0[11:7] | opcode[6:0]
// Register variant (inst[25]=1): func4 | 00 | 1 | 0[24:20] | rs1[19:15] | 000 | 00000 | 0101011

#define AME_CONFIG_REG(func4_val, rs1_reg) \
    ( AME_OPCODE \
    | (0       << 7)   /* rd = x0 */ \
    | (0       << 12)  /* func3 = 000 */ \
    | ((rs1_reg) << 15) /* rs1 */ \
    | (0       << 20)  /* unused */ \
    | (1       << 25)  /* imm_sel = 1 (register) */ \
    | (UOP_CONFIG << 26) /* uop = 00 */ \
    | ((func4_val) << 28) /* func4 */ \
    )

// msettilem rs1: func4=0000
#define AME_MSETTILEM(rs1_reg)  AME_CONFIG_REG(0x0, rs1_reg)
// msettilek rs1: func4=0001
#define AME_MSETTILEK(rs1_reg)  AME_CONFIG_REG(0x1, rs1_reg)
// msettilen rs1: func4=0010
#define AME_MSETTILEN(rs1_reg)  AME_CONFIG_REG(0x2, rs1_reg)
// mrelease: all fields zero except opcode
#define AME_MRELEASE()          (AME_OPCODE)

// --- Load/Store Instructions ---
// Format: func4[31:28] | uop=01[27:26] | ls[25] | rs2[24:20] | rs1[19:15] | func3=000[14:12] | d_size[11:10] | md[9:7] | opcode[6:0]

#define AME_LDST(func4_val, ls_val, rs2_reg, rs1_reg, dsize, md_val) \
    ( AME_OPCODE \
    | ((md_val)    << 7)  /* md/ms3 [9:7] */ \
    | ((dsize)     << 10) /* d_size [11:10] */ \
    | (0           << 12) /* func3 = 000 */ \
    | ((rs1_reg)   << 15) /* rs1 [19:15] */ \
    | ((rs2_reg)   << 20) /* rs2 [24:20] */ \
    | ((ls_val)    << 25) /* ls [25] */ \
    | (UOP_LDST    << 26) /* uop = 01 */ \
    | ((func4_val) << 28) /* func4 [31:28] */ \
    )

// Load macros (ls=0)
#define AME_MLAE(dsize, md, rs1, rs2)   AME_LDST(FUNC4_MLA, 0, rs2, rs1, dsize, md)
#define AME_MLBE(dsize, md, rs1, rs2)   AME_LDST(FUNC4_MLB, 0, rs2, rs1, dsize, md)
#define AME_MLCE(dsize, md, rs1, rs2)   AME_LDST(FUNC4_MLC, 0, rs2, rs1, dsize, md)
#define AME_MLATE(dsize, md, rs1, rs2)  AME_LDST(FUNC4_MLAT, 0, rs2, rs1, dsize, md)
#define AME_MLBTE(dsize, md, rs1, rs2)  AME_LDST(FUNC4_MLBT, 0, rs2, rs1, dsize, md)
#define AME_MLCTE(dsize, md, rs1, rs2)  AME_LDST(FUNC4_MLCT, 0, rs2, rs1, dsize, md)

// Store macros (ls=1)
#define AME_MSAE(dsize, ms3, rs1, rs2)  AME_LDST(FUNC4_MLA, 1, rs2, rs1, dsize, ms3)
#define AME_MSBE(dsize, ms3, rs1, rs2)  AME_LDST(FUNC4_MLB, 1, rs2, rs1, dsize, ms3)
#define AME_MSCE(dsize, ms3, rs1, rs2)  AME_LDST(FUNC4_MLC, 1, rs2, rs1, dsize, ms3)
#define AME_MSATE(dsize, ms3, rs1, rs2) AME_LDST(FUNC4_MLAT, 1, rs2, rs1, dsize, ms3)
#define AME_MSBTE(dsize, ms3, rs1, rs2) AME_LDST(FUNC4_MLBT, 1, rs2, rs1, dsize, ms3)
#define AME_MSCTE(dsize, ms3, rs1, rs2) AME_LDST(FUNC4_MLCT, 1, rs2, rs1, dsize, ms3)

// --- Matrix Multiplication Instructions ---
// Format: func4[31:28] | uop=10[27:26] | size_sup[25:23] | ms2[22:20] | s_size[19:18] | ms1[17:15] | func3=000[14:12] | d_size[11:10] | md[9:7] | opcode[6:0]

#define AME_MATMUL(func4_val, size_sup_val, ms2_val, ssize, ms1_val, dsize, md_val) \
    ( AME_OPCODE \
    | ((md_val)       << 7)  /* md [9:7] */ \
    | ((dsize)        << 10) /* d_size [11:10] */ \
    | (0              << 12) /* func3 = 000 */ \
    | ((ms1_val)      << 15) /* ms1 [17:15] */ \
    | ((ssize)        << 18) /* s_size [19:18] */ \
    | ((ms2_val)      << 20) /* ms2 [22:20] */ \
    | ((size_sup_val) << 23) /* size_sup [25:23] */ \
    | (UOP_MATMUL     << 26) /* uop = 10 */ \
    | ((func4_val)    << 28) /* func4 [31:28] */ \
    )

// Integer matmul: func4=0001
// size_sup: inst[25]=int4_sel, inst[24]=ms1_signed, inst[23]=ms2_signed
// mmacc.w.b (signed*signed):   size_sup = 0b011 (inst[25]=0, inst[24]=1, inst[23]=1)
// mmaccu.w.b (unsigned*unsigned): size_sup = 0b000
// mmaccsu.w.b (signed*unsigned):  size_sup = 0b010 (inst[24]=1, inst[23]=0)
// mmaccus.w.b (unsigned*signed):  size_sup = 0b001 (inst[24]=0, inst[23]=1)
#define AME_MMACC_W_B(md, ms1, ms2)    AME_MATMUL(FUNC4_INT, 0b011, ms2, ESIZE_8, ms1, ESIZE_32, md)
#define AME_MMACCU_W_B(md, ms1, ms2)   AME_MATMUL(FUNC4_INT, 0b000, ms2, ESIZE_8, ms1, ESIZE_32, md)
#define AME_MMACCSU_W_B(md, ms1, ms2)  AME_MATMUL(FUNC4_INT, 0b010, ms2, ESIZE_8, ms1, ESIZE_32, md)
#define AME_MMACCUS_W_B(md, ms1, ms2)  AME_MATMUL(FUNC4_INT, 0b001, ms2, ESIZE_8, ms1, ESIZE_32, md)

// Float matmul non-widen: func4=0000, inst[24]=0
// mfmacc.h (fp16->fp16): s_size=01, d_size=01, size_sup=0b000
#define AME_MFMACC_H(md, ms1, ms2)     AME_MATMUL(FUNC4_FLOAT, 0b000, ms2, ESIZE_16, ms1, ESIZE_16, md)
// mfmacc.s (fp32->fp32): s_size=10, d_size=10, size_sup=0b000
#define AME_MFMACC_S(md, ms1, ms2)     AME_MATMUL(FUNC4_FLOAT, 0b000, ms2, ESIZE_32, ms1, ESIZE_32, md)

// Float matmul double-widen: func4=0000, inst[24]=1
// mfmacc.s.h (fp16->fp32): s_size=01, d_size=10, size_sup=0b010 (inst[25]=0,inst[24]=1,inst[23]=0)
#define AME_MFMACC_S_H(md, ms1, ms2)   AME_MATMUL(FUNC4_FLOAT, 0b010, ms2, ESIZE_16, ms1, ESIZE_32, md)
// mfmacc.s.bf16 (bf16->fp32): s_size=01, d_size=10, size_sup=0b110 (inst[25]=1,inst[24]=1,inst[23]=0)
#define AME_MFMACC_S_BF16(md, ms1, ms2) AME_MATMUL(FUNC4_FLOAT, 0b110, ms2, ESIZE_16, ms1, ESIZE_32, md)

// Float matmul quad-widen: func4=0000
// mfmacc.s.e4 (fp8e4m3->fp32): s_size=00, d_size=10, size_sup=0b011 (inst[25]=0,inst[24]=1,inst[23]=1=E4M3)
#define AME_MFMACC_S_E4(md, ms1, ms2)  AME_MATMUL(FUNC4_FLOAT, 0b011, ms2, ESIZE_8, ms1, ESIZE_32, md)
// mfmacc.s.e5 (fp8e5m2->fp32): s_size=00, d_size=10, size_sup=0b010 (inst[25]=0,inst[24]=1,inst[23]=0=E5M2)
#define AME_MFMACC_S_E5(md, ms1, ms2)  AME_MATMUL(FUNC4_FLOAT, 0b010, ms2, ESIZE_8, ms1, ESIZE_32, md)

// --- MISC Instructions ---
// Format: func4[31:28] | uop=11[27:26] | imm3[25:23] | ms2[22:20] | s_size[19:18] | ms1[17:15] | func3=000[14:12] | d_size[11:10] | md[9:7] | opcode[6:0]

#define AME_MISC(func4_val, imm3, ms2_val, ssize, ms1_val, dsize, md_val) \
    ( AME_OPCODE \
    | ((md_val)    << 7)  \
    | ((dsize)     << 10) \
    | (0           << 12) \
    | ((ms1_val)   << 15) \
    | ((ssize)     << 18) \
    | ((ms2_val)   << 20) \
    | ((imm3)      << 23) \
    | (UOP_MISC    << 26) \
    | ((func4_val) << 28) \
    )

// mzero: zero 1/2/4/8 registers starting from md
// imm3[2:0] encodes count: 000=1reg, 001=2reg, 010=4reg, 011=8reg
#define AME_MZERO(md)       AME_MISC(FUNC4_MZERO, 0, 0, 0, 0, 0, md)
#define AME_MZERO2(md)      AME_MISC(FUNC4_MZERO, 1, 0, 0, 0, 0, md)
#define AME_MZERO4(md)      AME_MISC(FUNC4_MZERO, 2, 0, 0, 0, 0, md)
#define AME_MZERO8(md)      AME_MISC(FUNC4_MZERO, 3, 0, 0, 0, 0, md)

// ============================================================
// Inline assembly helpers
// ============================================================

#define GET_VALUE1(x) #x
#define GET_VALUE(x) GET_VALUE1(x)

// Issue a single AME instruction (no GPR input/output needed for matmul/misc)
#define AME_ISSUE(encoding) \
    __asm__ __volatile__(".word " GET_VALUE(encoding) "\n\t" ::: "memory")

// Issue AME instruction that reads rs1 and rs2 from GPR (for load/store/config)
// We put base_addr in t1 (x6) and stride in t2 (x7), matching rs1=6, rs2=7 in encoding
#define AME_ISSUE_WITH_GPR(encoding, val_rs1, val_rs2) \
    do { \
        register uint64_t _rs1 __asm__("t1") = (uint64_t)(val_rs1); \
        register uint64_t _rs2 __asm__("t2") = (uint64_t)(val_rs2); \
        __asm__ __volatile__( \
            ".word " GET_VALUE(encoding) "\n\t" \
            : \
            : "r"(_rs1), "r"(_rs2) \
            : "memory" \
        ); \
    } while(0)

// GPR register numbers used in instruction encoding
#define GPR_T1  6   // x6 = t1
#define GPR_T2  7   // x7 = t2
#define GPR_T0  5   // x5 = t0

// RoCC instruction encoding (for fence.m and mstatus which use RoCC format)
// RoCC format: funct7[31:25] | rs2[24:20] | rs1[19:15] | xd[14] | xs1[13] | xs2[12] | rd[11:7] | opcode[6:0]
// Note: RoCC uses different bit layout than standard RISC-V R-type
#define ROCC_BIT(opcode, rd, xs2, xs1, xd, rs1, rs2, funct7) \
    ( (opcode) \
    | ((rd)     << 7) \
    | ((xs2)    << 12) \
    | ((xs1)    << 13) \
    | ((xd)     << 14) \
    | ((rs1)    << 15) \
    | ((rs2)    << 20) \
    | ((funct7) << 25) \
    )

// fence.m: RoCC CUSTOM1, funct=0x70, no GPR operands, no writeback
#define AME_FENCE_M_ENC  ROCC_BIT(0x2B, 0, 0, 0, 0, 0, 0, 0x70)
// mstatus: RoCC CUSTOM1, funct=0x71, xd=1 (writes rd), rd=t0(5)
#define AME_MSTATUS_ENC  ROCC_BIT(0x2B, GPR_T0, 1, 1, 1, 6, 7, 0x71)

// ============================================================
// High-level API functions
// ============================================================

// Configuration (value passed via t1/rs1)
static inline void ame_settilem(uint64_t m) {
    AME_ISSUE_WITH_GPR(AME_MSETTILEM(GPR_T1), m, 0);
}

static inline void ame_settilen(uint64_t n) {
    AME_ISSUE_WITH_GPR(AME_MSETTILEN(GPR_T1), n, 0);
}

static inline void ame_settilek(uint64_t k) {
    AME_ISSUE_WITH_GPR(AME_MSETTILEK(GPR_T1), k, 0);
}

static inline void ame_release(void) {
    AME_ISSUE(AME_MRELEASE());
}

// Load A (8-bit) into tr0: base in t1, stride in t2
static inline void ame_mlae8(uint64_t base_addr, uint64_t stride) {
    AME_ISSUE_WITH_GPR(AME_MLAE(ESIZE_8, TR0, GPR_T1, GPR_T2), base_addr, stride);
}

// Load B (8-bit) into tr2: base in t1, stride in t2
static inline void ame_mlbe8(uint64_t base_addr, uint64_t stride) {
    AME_ISSUE_WITH_GPR(AME_MLBE(ESIZE_8, TR2, GPR_T1, GPR_T2), base_addr, stride);
}

// Zero accumulator acc0
static inline void ame_mzero(void) {
    AME_ISSUE(AME_MZERO(ACC0));
}

// Matrix multiply-accumulate: acc0 += tr0 * tr2^T (int8 signed)
static inline void ame_mmacc_w_b(void) {
    AME_ISSUE(AME_MMACC_W_B(ACC0, TR0, TR2));
}

// Store acc0 as 32-bit to memory: base in t1, stride in t2
static inline void ame_msce32(uint64_t base_addr, uint64_t stride) {
    AME_ISSUE_WITH_GPR(AME_MSCE(ESIZE_32, ACC0, GPR_T1, GPR_T2), base_addr, stride);
}

// Fence: hardware-blocking wait for all AME operations to complete.
// The RoCC interface stalls the CPU pipeline until all micro-instruction
// FIFOs are empty and all operations have finished.
static inline void ame_fence(void) {
    __asm__ __volatile__(".word " GET_VALUE(AME_FENCE_M_ENC) "\n\t" ::: "memory");
}

// Status query (non-blocking): returns pipeline status word.
//   bit[0]: load_fifo_full
//   bit[1]: compute_fifo_full
//   bit[2]: store_fifo_full
//   bit[3]: busy (any operation in flight)
//   bit[4]: all_idle (1 = all complete, safe to read results)
// Software can poll (ame_status() >> 4) & 1 to check completion without blocking.
static inline uint64_t ame_status(void) {
    uint64_t res;
    __asm__ __volatile__(
        "sd t0, -8(sp)\n\t"
        ".word " GET_VALUE(AME_MSTATUS_ENC) "\n\t"
        "add %0, zero, t0\n\t"
        "ld t0, -8(sp)\n\t"
        : "=r"(res)
        :
        : "t0", "memory"
    );
    return res;
}
#define YGJK_INS_RRR(rd, rs1, rs2)                      \ 
{                                                           \
    __asm__ __volatile__ (                                  \
        "sd t0, -24(sp)\n\t"                                \
        "sd t1, -16(sp)\n\t"                                    \
        "sd t2,  -8(sp)\n\t"                                    \
        "add t1, zero, %1\n\t"                                  \
        "add t2, zero, %2\n\t"                                  \
        ".word " GET_VALUE(AME_MSTATUS_ENC) "\n\t"              \
        "add %0, zero, t0\n\t"                                  \
        "ld t0, -24(sp)\n\t"                                    \
        "ld t1, -16(sp)\n\t"                                    \
        "ld t2,  -8(sp)\n\t"                                    \
        :"=r"(rd)                                           \
        :"r" (rs1) , "r" (rs2)                              \
        :"t0","t1","t2","memory"                                     \
        );                                                  \
}
// Non-blocking completion check
static inline int ame_is_idle(void) {
    return (ame_status() >> 4) & 1;
}

uint64_t cute_inst_fifo_finish_search()
{
    uint64_t res1=1;
    YGJK_INS_RRR(res1, 0, 0);
    return res1;
}
#endif // AME_H
