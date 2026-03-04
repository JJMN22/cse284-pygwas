#!/usr/bin/env python3
"""Compare GWAS .linear results with expected effect sizes from .snplist"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the .linear file (GWAS results)
linear_df = pd.read_csv(
    'result100_pca30_covar.assoc.linear',
    sep=r'\s+',
    dtype={'CHR': int, 'SNP': str, 'BP': int, 'A1': str, 'TEST': str, 'NMISS': int, 'BETA': float, 'SE': float, 'T': float, 'P': float}
)

# Load the .snplist file (expected effect sizes)
snplist_df = pd.read_csv(
    'sim/poly100_h05.snplist',
    sep=r'\s+',
    header=None,
    names=['SNP', 'TRUE_BETA']
)
plt.hist(snplist_df['TRUE_BETA'], bins=20, alpha=0.5, label='TRUE_BETA')
plt.xlabel('Effect Size')
plt.ylabel('Frequency')
plt.title('Distribution of True Effect Sizes')
plt.legend()
plt.show()
print(f"GWAS Results: {len(linear_df)} SNPs")
print(f"Expected (snplist): {len(snplist_df)} SNPs")
print()

# Merge on SNP ID
merged = pd.merge(linear_df, snplist_df, on='SNP', how='inner')
print(f"SNPs in both files: {len(merged)}")
print()



# Calculate correlation between estimated and true betas
if len(merged) > 0:
    correlation = merged['BETA'].corr(merged['TRUE_BETA'])
    print(f"Correlation (BETA vs TRUE_BETA): {correlation:.6f}")
    print()
    plt.scatter(merged['TRUE_BETA'], merged['BETA'], alpha=0.5)
    plt.xlabel('TRUE_BETA')
    plt.ylabel('BETA')
    plt.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')  # Line y=x for reference
    plt.title('Comparison of Estimated and True Effect Sizes')
    plt.show()
    # Calculate MSE and MAE
    mse = ((merged['BETA'] - merged['TRUE_BETA']) ** 2).mean()
    mae = (abs(merged['BETA'] - merged['TRUE_BETA'])).mean()
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print()
    
    # Show top 10 matches and mismatches
    merged['diff'] = abs(merged['BETA'] - merged['TRUE_BETA'])
    print(merged)
    print("Top 10 closest estimates:")
    print(merged[['SNP', 'BETA', 'TRUE_BETA', 'P', 'diff']].nsmallest(10, 'diff').to_string(index=False))
    print()
    
    print("Top 10 largest discrepancies:")
    print(merged[['SNP', 'BETA', 'TRUE_BETA', 'P', 'diff']].nlargest(10, 'diff').to_string(index=False))
    print()
    
    # Summary statistics
    print("BETA Statistics:")
    print(f"  Range: [{merged['BETA'].min():.6f}, {merged['BETA'].max():.6f}]")
    print(f"  Mean: {merged['BETA'].mean():.6f}")
    print(f"  Std: {merged['BETA'].std():.6f}")
    print()
    
    print("TRUE_BETA Statistics:")
    print(f"  Range: [{merged['TRUE_BETA'].min():.6f}, {merged['TRUE_BETA'].max():.6f}]")
    print(f"  Mean: {merged['TRUE_BETA'].mean():.6f}")
    print(f"  Std: {merged['TRUE_BETA'].std():.6f}")
