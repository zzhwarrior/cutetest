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

// --- Additional float matmul encodings (matches AMEInstConfigs FUNCT_MFMACC_*) ---
// mfmacc.d (fp64 -> fp64), non-widen: s_size=11, d_size=11, size_sup=0b000
#define AME_MFMACC_D(md, ms1, ms2)      AME_MATMUL(FUNC4_FLOAT, 0b000, ms2, ESIZE_64, ms1, ESIZE_64, md)
// mfmacc.d.s (fp32 -> fp64), double-widen: s_size=10, d_size=11, size_sup=0b010
#define AME_MFMACC_D_S(md, ms1, ms2)    AME_MATMUL(FUNC4_FLOAT, 0b010, ms2, ESIZE_32, ms1, ESIZE_64, md)
// mfmacc.h.e4 (fp8e4m3 -> fp16), double-widen: s_size=00, d_size=01, size_sup=0b011
#define AME_MFMACC_H_E4(md, ms1, ms2)   AME_MATMUL(FUNC4_FLOAT, 0b011, ms2, ESIZE_8, ms1, ESIZE_16, md)
// mfmacc.h.e5 (fp8e5m2 -> fp16), double-widen: s_size=00, d_size=01, size_sup=0b010
#define AME_MFMACC_H_E5(md, ms1, ms2)   AME_MATMUL(FUNC4_FLOAT, 0b010, ms2, ESIZE_8, ms1, ESIZE_16, md)
// mfmacc.bf16.e4 (fp8e4m3 -> bf16), double-widen: s_size=00, d_size=01, size_sup=0b111
// (inst[25]=1 for bf16 nuance, inst[24]=1 widen, inst[23]=1 = E4M3)
#define AME_MFMACC_BF16_E4(md, ms1, ms2) AME_MATMUL(FUNC4_FLOAT, 0b111, ms2, ESIZE_8, ms1, ESIZE_16, md)
// mfmacc.bf16.e5 (fp8e5m2 -> bf16), double-widen: size_sup=0b110
#define AME_MFMACC_BF16_E5(md, ms1, ms2) AME_MATMUL(FUNC4_FLOAT, 0b110, ms2, ESIZE_8, ms1, ESIZE_16, md)

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
// Workaround for a Shuttle-side RoCC dispatch bug: when the compiler emits
// LUI + ALU + ALU directly into t1 (or t2) as the operand computation for a
// RoCC instruction, Shuttle's fusion/rename path reads a stale rs1 that skips
// the middle ALU. Symptom: mlbe4 kt=1 dispatched with rs1=0x10201040 (= LUI +
// ADDI, missing SLLI) instead of 0x81008040, causing BML to fetch an invalid
// address and hang.
//
// Fix: let the compiler compute val_rs1/val_rs2 into any scratch register
// (via the "r" input constraint), then explicitly `mv` into t1/t2 as the
// last writer before the RoCC dispatch. This guarantees t1's most recent
// writer is a single `mv` — no LUI+X fusion pattern for Shuttle to trip on.
#define AME_ISSUE_WITH_GPR(encoding, val_rs1, val_rs2) \
    do { \
        __asm__ __volatile__( \
            "mv t1, %0\n\t" \
            "mv t2, %1\n\t" \
            ".word " GET_VALUE(encoding) "\n\t" \
            : \
            : "r"((uint64_t)(val_rs1)), "r"((uint64_t)(val_rs2)) \
            : "t1", "t2", "memory" \
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
#define AME_FENCE_M_ENC  ROCC_BIT(0x2B, GPR_T0, 0, 0, 1, 0, 0, 0x70)
// mstatus: RoCC CUSTOM1, funct=0x71, xd=1 (writes rd), rd=t0(5)
#define AME_MSTATUS_ENC  ROCC_BIT(0x2B, GPR_T0, 0, 0, 1, 0, 0, 0x71)
// ame_dma_load: RoCC CUSTOM1, funct=0x72, xs1=xs2=1, xd=0. rs1=src(64b),
// rs2 = (dst << 32) | length. CPU stalls in io.cmd.ready until DMA completes.
// rs1=t1(6), rs2=t2(7) so the inline-asm scaffolding in AME_ISSUE_WITH_GPR
// works verbatim.
#define AME_DMA_LOAD_ENC ROCC_BIT(0x2B, 0, 1, 1, 0, GPR_T1, GPR_T2, 0x72)

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

// ============================================================
// High-level API: register-parameterized macros
// tr_reg: TR0/TR1/TR2/TR3  acc_reg: ACC0/ACC1/ACC2/ACC3
// All register arguments must be compile-time constants (#define values).
// ============================================================

// ------------------------------------------------------------
// Load helpers (uop=01, ls=0). base_addr in t1, stride in t2.
// Element size is baked into the macro name (8/16/32/64).
// ------------------------------------------------------------

// mlae — Load A (non-transposed) into tr_reg
#define ame_mlae8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLAE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlae16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLAE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlae32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLAE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlae64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLAE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// mlbe — Load B (non-transposed) into tr_reg
#define ame_mlbe8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbe16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbe32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbe64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// mlce — Load C (accumulator) into acc_reg (md field carries acc index)
#define ame_mlce8(acc_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCE(ESIZE_8,  acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlce16(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCE(ESIZE_16, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlce32(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCE(ESIZE_32, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlce64(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCE(ESIZE_64, acc_reg, GPR_T1, GPR_T2), base_addr, stride)

// mlate — Load A transposed
#define ame_mlate8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLATE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlate16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLATE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlate32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLATE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlate64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLATE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// mlbte — Load B transposed
#define ame_mlbte8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBTE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbte16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBTE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbte32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBTE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbte64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLBTE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// mlcte — Load C transposed
#define ame_mlcte8(acc_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCTE(ESIZE_8,  acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlcte16(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCTE(ESIZE_16, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlcte32(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCTE(ESIZE_32, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlcte64(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MLCTE(ESIZE_64, acc_reg, GPR_T1, GPR_T2), base_addr, stride)

// ------------------------------------------------------------
// Store helpers (uop=01, ls=1). base_addr in t1, stride in t2.
// ------------------------------------------------------------

// msae — Store A
#define ame_msae8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSAE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msae16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSAE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msae32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSAE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msae64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSAE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// msbe — Store B
#define ame_msbe8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbe16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbe32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbe64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// msce — Store C (accumulator) at all element widths
#define ame_msce8(acc_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCE(ESIZE_8,  acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msce16(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCE(ESIZE_16, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msce32(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCE(ESIZE_32, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msce64(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCE(ESIZE_64, acc_reg, GPR_T1, GPR_T2), base_addr, stride)

// msate — Store A transposed
#define ame_msate8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSATE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msate16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSATE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msate32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSATE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msate64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSATE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// msbte — Store B transposed
#define ame_msbte8(tr_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBTE(ESIZE_8,  tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbte16(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBTE(ESIZE_16, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbte32(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBTE(ESIZE_32, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_msbte64(tr_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSBTE(ESIZE_64, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

// mscte — Store C transposed
#define ame_mscte8(acc_reg,  base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCTE(ESIZE_8,  acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mscte16(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCTE(ESIZE_16, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mscte32(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCTE(ESIZE_32, acc_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mscte64(acc_reg, base_addr, stride) AME_ISSUE_WITH_GPR(AME_MSCTE(ESIZE_64, acc_reg, GPR_T1, GPR_T2), base_addr, stride)

// ------------------------------------------------------------
// Misc: mzero (batch-count 1/2/4/8 accumulators cleared).
// ------------------------------------------------------------
#define ame_mzero(acc_reg)  AME_ISSUE(AME_MZERO(acc_reg))
#define ame_mzero2(acc_reg) AME_ISSUE(AME_MZERO2(acc_reg))
#define ame_mzero4(acc_reg) AME_ISSUE(AME_MZERO4(acc_reg))
#define ame_mzero8(acc_reg) AME_ISSUE(AME_MZERO8(acc_reg))

// ------------------------------------------------------------
// Integer matmul: acc_reg += tr_a * tr_b^T (all int8 → int32).
// Signedness suffix indicates {A_signed, B_signed}:
//   w_b   = signed  * signed
//   u_w_b = unsigned * unsigned
//   su_w_b= signed  * unsigned
//   us_w_b= unsigned * signed
// ------------------------------------------------------------
#define ame_mmacc_w_b(acc_reg, tr_a, tr_b)   AME_ISSUE(AME_MMACC_W_B(acc_reg, tr_a, tr_b))
#define ame_mmaccu_w_b(acc_reg, tr_a, tr_b)  AME_ISSUE(AME_MMACCU_W_B(acc_reg, tr_a, tr_b))
#define ame_mmaccsu_w_b(acc_reg, tr_a, tr_b) AME_ISSUE(AME_MMACCSU_W_B(acc_reg, tr_a, tr_b))
#define ame_mmaccus_w_b(acc_reg, tr_a, tr_b) AME_ISSUE(AME_MMACCUS_W_B(acc_reg, tr_a, tr_b))

// ------------------------------------------------------------
// Float matmul family. Names match AMEInstConfigs.FUNCT_MFMACC_*.
// Non-widen: src and dst have identical element size.
//   ame_mfmacc_h   : fp16   -> fp16
//   ame_mfmacc_s   : fp32   -> fp32
//   ame_mfmacc_d   : fp64   -> fp64
// Double-widen: dst = 2x src.
//   ame_mfmacc_s_h    : fp16    -> fp32
//   ame_mfmacc_s_bf16 : bf16    -> fp32
//   ame_mfmacc_d_s    : fp32    -> fp64
//   ame_mfmacc_h_e4   : fp8e4m3 -> fp16
//   ame_mfmacc_h_e5   : fp8e5m2 -> fp16
//   ame_mfmacc_bf16_e4: fp8e4m3 -> bf16
//   ame_mfmacc_bf16_e5: fp8e5m2 -> bf16
// Quad-widen: dst = 4x src.
//   ame_mfmacc_s_e4 : fp8e4m3 -> fp32
//   ame_mfmacc_s_e5 : fp8e5m2 -> fp32
// ------------------------------------------------------------
#define ame_mfmacc_h(acc_reg, tr_a, tr_b)       AME_ISSUE(AME_MFMACC_H(acc_reg, tr_a, tr_b))
#define ame_mfmacc_s(acc_reg, tr_a, tr_b)       AME_ISSUE(AME_MFMACC_S(acc_reg, tr_a, tr_b))
#define ame_mfmacc_d(acc_reg, tr_a, tr_b)       AME_ISSUE(AME_MFMACC_D(acc_reg, tr_a, tr_b))
#define ame_mfmacc_s_h(acc_reg, tr_a, tr_b)     AME_ISSUE(AME_MFMACC_S_H(acc_reg, tr_a, tr_b))
#define ame_mfmacc_s_bf16(acc_reg, tr_a, tr_b)  AME_ISSUE(AME_MFMACC_S_BF16(acc_reg, tr_a, tr_b))
#define ame_mfmacc_d_s(acc_reg, tr_a, tr_b)     AME_ISSUE(AME_MFMACC_D_S(acc_reg, tr_a, tr_b))
#define ame_mfmacc_h_e4(acc_reg, tr_a, tr_b)    AME_ISSUE(AME_MFMACC_H_E4(acc_reg, tr_a, tr_b))
#define ame_mfmacc_h_e5(acc_reg, tr_a, tr_b)    AME_ISSUE(AME_MFMACC_H_E5(acc_reg, tr_a, tr_b))
#define ame_mfmacc_bf16_e4(acc_reg, tr_a, tr_b) AME_ISSUE(AME_MFMACC_BF16_E4(acc_reg, tr_a, tr_b))
#define ame_mfmacc_bf16_e5(acc_reg, tr_a, tr_b) AME_ISSUE(AME_MFMACC_BF16_E5(acc_reg, tr_a, tr_b))
#define ame_mfmacc_s_e4(acc_reg, tr_a, tr_b)    AME_ISSUE(AME_MFMACC_S_E4(acc_reg, tr_a, tr_b))
#define ame_mfmacc_s_e5(acc_reg, tr_a, tr_b)    AME_ISSUE(AME_MFMACC_S_E5(acc_reg, tr_a, tr_b))

// Fence: hardware-blocking wait for all AME operations to complete.
// The RoCC interface stalls the CPU pipeline until all micro-instruction
// FIFOs are empty and all operations have finished.
//
// AME_FENCE_M_ENC has rd=t0/xd=1 in its RoCC encoding, so the fence returns
// a value into t0 (the return is effectively unused, always 0). We must list
// `t0` in the clobber set — otherwise the compiler treats t0 as preserved
// across ame_fence() and can pre-compute long-lived values into t0 that
// silently become 0 after the fence retires. Symptom seen in Phase E-small:
// only tiles where the compiler happened to allocate scale-base to t0 got
// MSET_SCALEB rs1=0 after crossing a fence.
static inline void ame_fence(void) {
    __asm__ __volatile__(".word " GET_VALUE(AME_FENCE_M_ENC) "\n\t" ::: "t0", "memory");
}

static inline uint64_t ame_status(void) {
    uint64_t res;
    __asm__ __volatile__(
        ".word " GET_VALUE(AME_MSTATUS_ENC) "\n\t"
        "mv %0, t0\n\t"
        : "=r"(res)
        :
        : "t0", "memory"
    );
    return res;
}


uint64_t ame_is_idle()
{
    return ame_status();
}

// Bulk-copy `length` bytes from DRAM at `src` into TCM at `dst` via the
// TcmDmaEngine. `length` must be a multiple of 64 (the DMA block size).
// Blocking: the RoCC holds io.cmd.ready low until the transfer finishes.
static inline void ame_dma_load(uint64_t src, uint64_t dst, uint32_t length) {
    uint64_t rs2 = (dst << 32) | (uint64_t)length;
    AME_ISSUE_WITH_GPR(AME_DMA_LOAD_ENC, src, rs2);
}

// ============================================================
// TCM partition control (Step 2A)
//
// Software runtime-configurable split between L2 cache and TCM scratchpad.
// Under grow-from-top convention, TCM offset 0 always maps to the highest
// physical way; larger offsets descend way indices. Shrinking TCM cuts
// the top of the aperture and preserves data at lower offsets. Growing
// TCM (fewer cache ways) requires the caller to have flushed any dirty
// cache lines in the target ways beforehand (fence + wbinvd equivalent).
// Legal way counts on Step 2A: {0, 1, 2, 4}. Value 8 (all-TCM) is
// currently rejected in HW.
//
// Per-tile layout: each tile's private L2 exposes its TcmCtrl regmap in a
// distinct 4 KB PBUS window (WithHuanCunL2 offsets tile N's base by
// N * 0x1000). Helper functions read mhartid at call time to hit the
// window owned by the current hart. Single-tile configs (hartid=0) are
// unaffected.
// ============================================================
#define TCM_CTRL_BASE_TILE0  0x22000000ULL
#define TCM_CTRL_WINDOW      0x1000ULL
#define TCM_MODE_OFFSET      0x00
#define TCM_STATUS_OFFSET    0x04
#define TCM_MASK_OFFSET      0x08
#define TCM_INFO_OFFSET      0x0C

#define TCM_STATUS_BUSY    0x1u
#define TCM_STATUS_ILLEGAL 0x2u

static inline uint64_t ame_tcm_ctrl_base(void) {
    uint64_t hid;
    __asm__ __volatile__("csrr %0, mhartid" : "=r"(hid));
    return TCM_CTRL_BASE_TILE0 + hid * TCM_CTRL_WINDOW;
}

static inline uint32_t ame_tcm_get_count(void) {
    return *(volatile uint32_t*)(ame_tcm_ctrl_base() + TCM_MODE_OFFSET) & 0xFu;
}
static inline uint32_t ame_tcm_get_mask(void) {
    return *(volatile uint32_t*)(ame_tcm_ctrl_base() + TCM_MASK_OFFSET) & 0xFFu;
}
static inline uint32_t ame_tcm_get_status(void) {
    return *(volatile uint32_t*)(ame_tcm_ctrl_base() + TCM_STATUS_OFFSET);
}
static inline uint32_t ame_tcm_get_info(void) {
    return *(volatile uint32_t*)(ame_tcm_ctrl_base() + TCM_INFO_OFFSET);
}

// Block until any in-flight partition change finishes.
static inline void ame_tcm_wait_idle(void) {
    while (ame_tcm_get_status() & TCM_STATUS_BUSY) { /* spin */ }
}

// Atomically switch the TCM partition. `count` must be one of {0, 1, 2, 4}.
// Returns 0 on success, negative on illegal value.
// Blocks until the transition is committed.
static inline int ame_tcm_config(uint32_t count) {
    uint64_t base = ame_tcm_ctrl_base();
    ame_tcm_wait_idle();
    *(volatile uint32_t*)(base + TCM_MODE_OFFSET) = count;
    asm volatile("fence rw, rw" ::: "memory");
    ame_tcm_wait_idle();
    uint32_t st = ame_tcm_get_status();
    if (st & TCM_STATUS_ILLEGAL) return -1;
    if (ame_tcm_get_count() != count) return -2;
    return 0;
}
#endif // AME_H
