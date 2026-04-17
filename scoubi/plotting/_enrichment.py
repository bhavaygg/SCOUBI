import numpy as np
import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from adjustText import adjust_text

def categorize_gene(row, lfc_threshold, pval_threshold):
    if abs(row['logfoldchanges']) > lfc_threshold and row['pvals_adj'] < pval_threshold:
        return 'Upregulated' if row['logfoldchanges'] > lfc_threshold else 'Downregulated'
    else:
        return 'Not Significant'

size_mapping = {
    'Upregulated': 60,
    'Downregulated': 60,
    'Not Significant': 15
}

_DEFAULT_PALETTE = {
    'axon':  '#ff1a1a',
    'dendrite': '#1aa3ff',
    'ns':    'grey',
}

def enrichment_scatter(
    adata: ad.AnnData,
    top_k: int = 25,
    z_thr: float = 5.0,
    lfc_clip: float = 3.0,
    remove_markers: bool = True,
    palette: dict = None,
    figsize: tuple = (8, 6),
    label_fontsize: int = 6,
    show: bool = False,
    ax=None,
):
    """
    Volcano-like scatter of axon-vs-dendrite enrichment (log2fc vs z-score).

    Data source: ``adata.uns['proportion']['a_vs_d']`` — must already contain
    ``log2fc`` and ``zscore`` columns (call ``scoubi.tl.axon_dendrite_enrichment``
    first).

    Three-layer rendering:

    1. **Not Significant** – small grey points.
    2. **Significant** (|z| > z_thr) but not in the top/bottom highlight set.
    3. **Highlighted** top/bottom ``top_k`` genes by z-score — larger, fully
       opaque, with ``adjustText`` gene labels.

    Parameters
    ----------
    adata : ad.AnnData
        Must have ``adata.uns['proportion']['a_vs_d']`` with columns
        ``gene``, ``zscore``, ``log2fc``.
    top_k : int
        Number of top (axon-enriched) and bottom (dendrite-enriched) genes
        to label.
    z_thr : float
        Z-score threshold for significance classification.
    lfc_clip : float
        log2fc values are clipped to ``[-lfc_clip, lfc_clip]`` before plotting.
    remove_markers : bool
        If True, genes listed in ``adata.uns['axon_markers']`` and
        ``adata.uns['dendrite_markers']`` are excluded before plotting.
    palette : dict, optional
        Colour map with keys ``'axon'``, ``'dendrite'``, ``'ns'``.
        Defaults to ``{'axon': '#ff1a1a', 'dendrite': '#1aa3ff', 'ns': 'grey'}``.
    figsize : tuple
        Figure size passed to ``plt.subplots`` when ``ax`` is None.
    label_fontsize : int
        Font size for gene labels.
    show : bool
        If True, call ``plt.show()`` before returning.  Set to False
        (default) in Jupyter to avoid the figure rendering twice.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if palette is None:
        palette = _DEFAULT_PALETTE
    col_axon = palette['axon']
    col_dend = palette['dendrite']
    col_ns   = palette['ns']

    df_z = adata.uns['ptest_a_vs_d'].copy()

    # Compute labels on the full dataset and write back to adata
    df_z['label'] = 'ns'
    df_z.loc[df_z['zscore'] >  z_thr, 'label'] = 'axon'
    df_z.loc[df_z['zscore'] < -z_thr, 'label'] = 'dendrite'
    adata.uns['ptest_a_vs_d'] = df_z

    if remove_markers:
        marker_genes = (
            set(adata.uns.get('axon_markers', []))
            | set(adata.uns.get('dendrite_markers', []))
        )
        df = df_z[~df_z['gene'].isin(marker_genes)].copy()
    else:
        df = df_z.copy()

    df['log2fc_plot'] = df['log2fc'].clip(-lfc_clip, lfc_clip)

    highlight_idx = pd.concat([
        df.nlargest(top_k,  'zscore'),
        df.nsmallest(top_k, 'zscore'),
    ]).index.unique()

    df_ns       = df[df['label'] == 'ns']
    df_sig      = df[df['label'] != 'ns']
    df_sig_nohl = df_sig[~df_sig.index.isin(highlight_idx)]
    df_hl       = df.loc[df.index.isin(highlight_idx)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Layer 1 – Not Significant
    ax.scatter(
        df_ns['log2fc_plot'], df_ns['zscore'],
        c=col_ns, s=12, alpha=0.45, linewidths=0, rasterized=True,
    )

    # Layer 2 – Significant but not highlighted
    for label, color in [('axon', col_axon), ('dendrite', col_dend)]:
        sub = df_sig_nohl[df_sig_nohl['label'] == label]
        ax.scatter(
            sub['log2fc_plot'], sub['zscore'],
            c=color, s=28, alpha=0.7, linewidths=0.3, edgecolors='k',
        )

    # Layer 3 – Highlighted top/bottom K
    for label, color in [('axon', col_axon), ('dendrite', col_dend)]:
        sub = df_hl[df_hl['label'] == label]
        ax.scatter(
            sub['log2fc_plot'], sub['zscore'],
            c=color, s=60, alpha=1.0, linewidths=0.5, edgecolors='k', zorder=5,
        )

    # Reference lines
    ax.axhline( z_thr, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(-z_thr, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0,      color='black', linestyle='-',  linewidth=0.8, alpha=0.35)

    # Gene labels for highlighted genes
    texts = []
    for _, row in df_hl.iterrows():
        texts.append(
            ax.text(row['log2fc_plot'], row['zscore'], row['gene'],
                    fontsize=label_fontsize, ha='center')
        )
    if texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='grey', lw=0.5),
        )

    # Styling
    ax.set_xlabel('log$_2$ FC  (axon / dendrite)', fontsize=13)
    ax.set_ylabel('z-score', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=5)
    ax.tick_params(axis='both', which='minor', width=0.8,  length=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.grid(which='minor', linestyle=':',  linewidth=0.3, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    sns.despine(ax=ax)

    if show:
        plt.show()
        return None

    return fig


_DEFAULT_REGION_COLORS = {
    "CTX":          "#118165",
    "Hippocampus":  "#82C748",
    "Thalamus":     "#fe7f84",
    "Hypothalamus": "#f2473b",
    "CNU":          "#98d6f8",
    "CNU-HYa":      "#9AA7D8",
    "White Matter": "#CDCDCC",
}


def regional_enrichment(
    adata: ad.AnnData,
    best_rank_key: str = 'ptest_best_rank_neuronal',
    region_color_map: dict = None,
    ylim: int = 200,
    z_interface_thr: float = 3.0,
    z_esb_thr: float = 3.0,
    n_sig_size: dict = None,
    default_sig_size: int = 30,
    nonsig_size: int = 20,
    nonsig_color: str = '#d4c5b0',
    label_ratio_thr: float = 2.0,
    label_min_cell_rank: int = 100,
    label_fontsize: int = 7,
    show_legend: bool = True,
    figsize: tuple = (10, 10),
    show: bool = False,
    ax=None,
):
    """
    Scatter plot of per-gene best interface rank vs cell rank in that region.

    Each point is a gene.  Points are split into two layers by ESB z-score:

    * **Low ESB** (``z_esb < z_esb_thr``) – foreground, alpha 0.9.
    * **High ESB** (``z_esb >= z_esb_thr``) – background, alpha 0.5.

    Within the low-ESB layer, genes that are also interface-enriched
    (``z_interface >= z_interface_thr``) are coloured by their best
    anatomical region and sized by ``n_sig_regions``; all others are
    shown in ``nonsig_color``.

    Labels are placed on genes that are interface-specific, rank within
    ``ylim``, have ``rank_cell / rank_interface > label_ratio_thr``, and
    ``rank_cell > label_min_cell_rank``.

    Data source: ``adata.uns[best_rank_key]`` — populate with
    ``scoubi.tl.p_test`` first.  Expected columns: ``best_region``,
    ``rank_interface``, ``rank_cell``, ``z_interface``, ``z_esb``,
    ``n_sig_regions``.

    Parameters
    ----------
    adata : ad.AnnData
        Must have ``adata.uns[best_rank_key]``.
    best_rank_key : str
        Key in ``adata.uns`` for the best-rank DataFrame.
        Default: ``'ptest_best_rank_neuronal'``.
    region_color_map : dict, optional
        Mapping of region name → hex colour.  Defaults to the 7-region
        mouse brain palette.
    ylim : int
        Upper limit of the y-axis (interface rank).  Default: 200.
    z_interface_thr : float
        Minimum interface z-score to call a gene significant.  Default: 3.0.
    z_esb_thr : float
        Maximum ESB z-score for interface-specific classification.
        Default: 3.0.
    n_sig_size : dict, optional
        Mapping of ``n_sig_regions`` → marker size for significant genes.
        Default: ``{1: 300, 2: 300, 3: 300, 4: 300}``.
    default_sig_size : int
        Fallback marker size for significant genes with
        ``n_sig_regions`` not in ``n_sig_size``.  Default: 30.
    nonsig_size : int
        Marker size for non-significant points.  Default: 20.
    nonsig_color : str
        Fill colour for non-significant points.  Default: ``'#d4c5b0'``.
    label_ratio_thr : float
        Minimum ``rank_cell / rank_interface`` ratio required to label a
        gene.  Default: 2.0.
    label_min_cell_rank : int
        Minimum ``rank_cell`` required to label a gene.  Default: 100.
    label_fontsize : int
        Font size for gene labels.  Default: 7.
    show_legend : bool
        Whether to draw the region + size legend.  Default: True.
    figsize : tuple
        Figure size when ``ax`` is None.  Default: ``(10, 10)``.
    show : bool
        If True, call ``plt.show()`` before returning.  Default: False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when None.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns None when ``show=True``.
    """
    from matplotlib.lines import Line2D

    if region_color_map is None:
        region_color_map = _DEFAULT_REGION_COLORS
    if n_sig_size is None:
        n_sig_size = {1: 300, 2: 300, 3: 300, 4: 300}

    br = adata.uns[best_rank_key].copy().reset_index(drop=True)

    # A gene is "interface-specific": low ESB AND high interface z-score
    iface_specific = (br['z_esb'] < z_esb_thr) & (br['z_interface'] >= z_interface_thr)
    low_esb        = br['z_esb'] < z_esb_thr

    colors = np.where(
        iface_specific,
        br['best_region'].map(lambda r: region_color_map.get(r, '#aaaaaa')),
        nonsig_color,
    )
    edge_colors = np.where(iface_specific, '#555555', 'none')
    edge_widths  = np.where(iface_specific, 0.6, 0.0)
    sizes = np.where(
        iface_specific,
        br['n_sig_regions'].map(lambda n: n_sig_size.get(n, default_sig_size)),
        float(nonsig_size),
    ).astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Layer by ESB: high-ESB (background) then low-ESB (foreground)
    for is_low, alpha, zorder in [(False, 0.5, 2), (True, 0.9, 3)]:
        m = low_esb if is_low else ~low_esb
        ax.scatter(
            br.loc[m, 'rank_cell'],
            br.loc[m, 'rank_interface'],
            s=sizes[m],
            c=colors[m],
            alpha=alpha,
            linewidths=edge_widths[m],
            edgecolors=edge_colors[m],
            zorder=zorder,
        )

    # Diagonal reference line
    max_val = br['rank_cell'].max()
    ax.plot([0, max_val], [0, max_val], color='#aaaaaa', lw=0.7, ls='--', zorder=1)

    # Labels: interface-specific genes within ylim, with large cell/interface rank ratio
    label_mask = (
        iface_specific &
        (br['rank_interface'] <= ylim) &
        (br['rank_cell'] / br['rank_interface'] > label_ratio_thr) &
        (br['rank_cell'] > label_min_cell_rank)
    )
    subset = br[label_mask]
    texts = []
    for _, row in subset.iterrows():
        texts.append(ax.text(
            row['rank_cell'], row['rank_interface'],
            row['gene'], fontsize=label_fontsize,
            style='italic', color='#1a1a1a', zorder=4,
        ))
    if texts:
        adjust_text(
            texts,
            x=subset['rank_cell'].values,
            y=subset['rank_interface'].values,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5),
            expand=(1.5, 1.8), lim=400,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(0, ylim)
    ax.set_xlabel('Rank in that region (soma)', fontsize=9)
    ax.set_ylabel('Best rank across regions (interface)', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#aaaaaa')
    ax.tick_params(colors='#444444', labelsize=8)

    if show_legend:
        region_handles = [
            ax.scatter([], [], s=100, color=color, alpha=0.8,
                       linewidths=0, label=region)
            for region, color in region_color_map.items()
        ]
        region_handles.append(
            ax.scatter([], [], s=100, color=nonsig_color, alpha=0.4,
                       linewidths=0, label='High ESB (z ≥ 3)')
        )
        size_handles = [
            ax.scatter([], [], s=n_sig_size.get(n, default_sig_size),
                       color='#888888', alpha=0.8,
                       linewidths=0.6, edgecolors='#555555', label=label)
            for n, label in [(1, '1 region'), (2, '2 regions'),
                             (3, '3 regions'), (4, '4+ regions')]
        ]
        separator = Line2D([0], [0], color='none', label='')
        ax.legend(
            handles=region_handles + [separator] + size_handles,
            frameon=False,
            fontsize=7,
            loc='upper right',
            labelspacing=1.8,
            handletextpad=0.8,
            bbox_to_anchor=(1.25, 1),
        )

    if show:
        plt.show()
        return None

    return fig
