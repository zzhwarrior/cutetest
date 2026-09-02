// AME NVFP4 extension — Phase A wire-up.
//
// Adds three instructions on top of the base AME ISA (ame_test/ame.h):
//   1. msetscalea rs1 : store rs1 (64-bit vaddr) into AMEDecoder's A-scale sticky reg
//   2. msetscaleb rs1 : store rs1 into AMEDecoder's B-scale sticky reg
//   3. mfmacc.s.nvfp4 md, ms1, ms2 : FP4×FP4 → FP32 matmul (data path not wired in Phase A)
//
// Native AME encoding (matches AMEInstConfigs.scala Phase A additions):
//   msetscalea : uop=00 (config), func4=0011, imm_sel=1, rs1=t1  → inst[31:25] = 0x19
//   msetscaleb : uop=00 (config), func4=0100, imm_sel=1, rs1=t1  → inst[31:25] = 0x21
//   mfmacc.s.nvfp4 : uop=10 (matmul), func4=0010, size_sup=011   → inst[31:25] = 0x14
//
// In Phase A, only the two config instructions have observable HW effect
// (they write scale_a_base_reg / scale_b_base_reg inside AMEDecoder). The
// matmul instruction is decodable but its data path is not yet connected —
// don't call ame_mfmacc_s_nvfp4() from a Phase A test.

#ifndef AME_NVFP4_EXT_H
#define AME_NVFP4_EXT_H

#include "../../ame_test/ame.h"

// -----------------------------------------------------------------------------
// KNOWN DEVIATION FROM RISC-V MATRIX SPEC (v0.6.0)
// -----------------------------------------------------------------------------
// Per spec 5.2.4, mtilek is the K dimension in ELEMENT COUNT, bounded by
// TRLEN/element_bit_width. So for INT8 (8-bit), settilek(64) means 64 int8
// elements; for NVFP4 (4-bit), settilek(64) should mean 64 fp4 elements
// (=32 bytes = half a 512-bit RV on CUTE_4Tops_128SCP).
//
// CUTE's current AMEDecoder (AMEDecoder.scala:139-141) instead treats mtilek
// as ReduceVector COUNT, clamped to Tensor_K/ReduceWidthByte (an INT8-centric
// value). For NVFP4 this over-processes by (element density factor) since
// one RV holds 2× the elements it does for INT8.
//
// PRACTICAL WORKAROUND while HW isn't spec-aligned yet: when running NVFP4
// kernels, sizing K to a multiple of the config's per-RV NVFP4 element count
// (2 * ReduceWidthByte). For CUTE_4Tops_128SCP (RWByte=64), that's 128 FP4
// elements per RV. Passing K=128 gives one full RV of accumulation.
//
// Fixing HW is tracked as a future item (spec-compliant datatype-aware
// K conversion in AMEDecoder before dispatching Load/Compute micro-insts).
// -----------------------------------------------------------------------------

// --- Encoding builders (follow AME_CONFIG_REG / AME_MATMUL patterns from ame.h) ---

// msetscalea rs1  (func4=0011, uop=00, imm_sel=1, rs1=GPR_T1)
#define AME_MSETSCALEA(rs1_reg)  AME_CONFIG_REG(0x3, rs1_reg)
// msetscaleb rs1  (func4=0100, uop=00, imm_sel=1, rs1=GPR_T1)
#define AME_MSETSCALEB(rs1_reg)  AME_CONFIG_REG(0x4, rs1_reg)

// mfmacc.s.nvfp4 md, ms1, ms2
//   func4=0010 (was reserved as "integer hybrid", now NVFP4)
//   uop=10 (matmul), s_size=00 (4-bit source, byte-packed), d_size=10 (fp32 dest),
//   size_sup=0b011 (same discriminator as fp8-widen paths)
#define AME_MFMACC_S_NVFP4(md, ms1, ms2) \
    AME_MATMUL(0x2, 0b011, ms2, ESIZE_8, ms1, ESIZE_32, md)

// mlae4 md, rs1, rs2 (Phase B) — NVFP4 tile load into TR{0..3}.
//   func4=1001 (unused slot), uop=01, ls=0. d_size ignored by decoder for this func4;
//   we still emit 00 to keep the encoding canonical.
//   rs1 = base vaddr of the packed-FP4 tile (2 elements/byte)
//   rs2 = row stride in BYTES (typically K/2 for a plain [M][K/2] layout)
// Scale base for A must have been set beforehand via ame_mset_scalea().
#define AME_MLAE4(dsize, md, rs1, rs2)  AME_LDST(0x9, 0, rs2, rs1, dsize, md)
// mlbe4 md, rs1, rs2 (Phase B) — NVFP4 tile load into TR{0..3}, B side.
//   func4=1010, otherwise identical to mlae4.
#define AME_MLBE4(dsize, md, rs1, rs2)  AME_LDST(0xA, 0, rs2, rs1, dsize, md)

// --- Inline wrappers ---

static inline void ame_mset_scalea(uint64_t vaddr) {
    // rs2 unused, pass 0. Encoding puts rs1 in t1 via AME_ISSUE_WITH_GPR.
    AME_ISSUE_WITH_GPR(AME_MSETSCALEA(GPR_T1), vaddr, 0);
}

static inline void ame_mset_scaleb(uint64_t vaddr) {
    AME_ISSUE_WITH_GPR(AME_MSETSCALEB(GPR_T1), vaddr, 0);
}

#define ame_mfmacc_s_nvfp4(acc_reg, tr_a, tr_b) \
    AME_ISSUE(AME_MFMACC_S_NVFP4(acc_reg, tr_a, tr_b))

// Phase B: base = t1, stride = t2, follows the mlae8/mlbe8 pattern in ame.h.
// d_size passed as ESIZE_8 (canonical zero) — see comment on AME_MLAE4.
#define ame_mlae4(tr_reg, base_addr, stride) \
    AME_ISSUE_WITH_GPR(AME_MLAE4(ESIZE_8, tr_reg, GPR_T1, GPR_T2), base_addr, stride)
#define ame_mlbe4(tr_reg, base_addr, stride) \
    AME_ISSUE_WITH_GPR(AME_MLBE4(ESIZE_8, tr_reg, GPR_T1, GPR_T2), base_addr, stride)

#endif // AME_NVFP4_EXT_H
