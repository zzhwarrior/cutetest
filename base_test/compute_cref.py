#!/usr/bin/env python3
"""
Compute C_ref = A * B^T from a matmul_value header file.

Usage:
    python3 compute_cref.py matmul_value_mnk_128_128_128.h
    python3 compute_cref.py matmul_value_mnk_128_128_64.h

Output: matmul_cref_M_N_K.h with precomputed c_ref[M][N]
"""

import re
import sys

def parse_matrix(content, varname):
    """Parse a matrix from the header file content."""
    # Try int8_t first, then char
    for typename in ['int8_t', 'char']:
        pattern = rf'static {typename} {varname}\[(\d+)\]\[(\d+)\].*?=\s*\{{(.*?)\}};'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            rows_str = match.group(3)
            rows = re.findall(r'\{([^}}]+)\}', rows_str)
            matrix = []
            for row in rows:
                vals = [int(x.strip()) for x in row.split(',') if x.strip()]
                matrix.append(vals)
            return matrix
    return None

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <matmul_value_header.h>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r') as f:
        content = f.read()

    a = parse_matrix(content, 'a')
    b = parse_matrix(content, 'b')

    if a is None:
        print("Error: could not parse matrix 'a'")
        sys.exit(1)
    if b is None:
        print("Error: could not parse matrix 'b'")
        sys.exit(1)

    M = len(a)
    K = len(a[0])
    N = len(b)
    assert len(b[0]) == K, f"K mismatch: a has {K} cols, b has {len(b[0])} cols"

    print(f"Parsed: A[{M}][{K}], B[{N}][{K}]")
    print(f"Computing C_ref[{M}][{N}] = A * B^T ...")

    # Compute C_ref = A * B^T
    c_ref = [[0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            s = 0
            for k in range(K):
                s += a[i][k] * b[j][k]
            c_ref[i][j] = s

    # Write output
    output_file = f"matmul_cref_{M}_{N}_{K}.h"
    with open(output_file, 'w') as f:
        f.write(f'#ifndef MATMUL_CREF_{M}_{N}_{K}_H\n')
        f.write(f'#define MATMUL_CREF_{M}_{N}_{K}_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'// C_ref[{M}][{N}] = A[{M}][{K}] * B[{N}][{K}]^T, int32\n')
        f.write(f'static const int32_t c_ref[{M}][{N}] __attribute__((aligned(256))) = {{\n')
        for i in range(M):
            f.write('{')
            f.write(','.join(str(c_ref[i][j]) for j in range(N)))
            f.write('},\n')
        f.write('};\n\n')
        f.write('#endif\n')

    print(f"Generated: {output_file}")
    print(f"  c_ref[0][0] = {c_ref[0][0]}")
    print(f"  c_ref[0][1] = {c_ref[0][1]}")
    print(f"  c_ref[{M-1}][{N-1}] = {c_ref[M-1][N-1]}")

if __name__ == '__main__':
    main()
