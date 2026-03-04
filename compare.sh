#!/bin/bash

echo "=== File Comparison: .linear vs .snplist ==="
echo

# Get SNPs in linear file
echo "SNPs in .linear file: $(tail -n +2 result_sim.assoc.linear | cut -f2 | wc -l)"

# Get SNPs in snplist file
echo "SNPs in .snplist file: $(wc -l < sim/poly200_h05_phenp.snplist)"
echo

# Check overlap
echo "SNPs present in both files:"
tail -n +2 result_sim.assoc.linear | cut -f2 > /tmp/linear_snps.txt
cut -f1 sim/poly200_h05_phenp.snplist > /tmp/snplist_snps.txt
comm -12 <(sort /tmp/linear_snps.txt) <(sort /tmp/snplist_snps.txt) | wc -l
echo

# Show sample comparison (first 10 SNPs that match)
echo "Sample comparison (first 10 matching SNPs):"
echo "SNP_ID          ESTIMATED_BETA  EXPECTED_BETA"
join <(tail -n +2 result_sim.assoc.linear | awk '{print $2, $7}' | sort) <(sort sim/poly200_h05_phenp.snplist) | head -10 | awk '{printf "%-15s %14.6f %14.6f\n", $1, $2, $3}'
