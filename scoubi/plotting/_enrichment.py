import numpy as np
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def categorize_gene(row, lfc_threshold, pval_threshold):
    if abs(row['logfoldchanges']) > lfc_threshold and row['pvals_adj'] < pval_threshold:
        return 'Upregulated' if row['logfoldchanges'] > lfc_threshold else 'Downregulated'
    else:
        return 'Not Significant'

palette = {
    'Upregulated': 'crimson',
    'Downregulated': 'royalblue',
    'Not Significant': 'grey'
}

size_mapping = {
    'Upregulated': 60,
    'Downregulated': 60,
    'Not Significant': 15
}

def volcano(adata: ad.AnnData, mode = "a_vs_d", lfc_threshold = 3, pval_threshold = 1e-5):
    de_df = adata.uns[mode]
    de_df['significance'] = de_df.apply(categorize_gene, axis=1, args=(lfc_threshold, pval_threshold))
    de_df['point_size'] = de_df['significance'].map(size_mapping)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=de_df,
        x='logfoldchanges',
        y='-log10(FDR)',
        hue='significance',
        size='point_size',
        palette=palette,
        alpha=0.75,
        edgecolor="black", ax = ax, legend = False
    )

    plt.axvline(x=lfc_threshold, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-lfc_threshold, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', linewidth=1)
    plt.xlabel('Log2 Fold Change', fontsize=16)
    plt.ylabel('-Log10 FDR', fontsize=16)
    plt.grid(False)
    max_abs_lfc = de_df['logfoldchanges'].abs().max()
    plot_limit = np.ceil(max_abs_lfc * 1.1) 
    sns.despine()
    plot_limit = np.ceil(max_abs_lfc * 1.1)
    plt.xlim(-plot_limit, plot_limit)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    plt.show()