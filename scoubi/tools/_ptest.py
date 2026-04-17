import numpy as np
import anndata as ad
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm


def _prop_ztest_vec(
    k1: np.ndarray,
    k2: np.ndarray,
    n1: int,
    n2: int,
) -> tuple:
    """Two-sided proportions z-test, vectorised over an array of counts.

    Parameters
    ----------
    k1, k2 : (G,) int arrays
        Success counts for each group.
    n1, n2 : int
        Total observations in each group.

    Returns
    -------
    z, p : (G,) float arrays
        z-statistics and two-sided p-values.  Returns NaN arrays when either
        group size is zero.
    """
    nan_arr = np.full(len(k1), np.nan)
    if n1 == 0 or n2 == 0:
        return nan_arr, nan_arr
    p1 = k1.astype(float) / n1
    p2 = k2.astype(float) / n2
    p_pool = (k1 + k2).astype(float) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, (p1 - p2) / se, np.nan)
    pv = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return z, pv


def _safe_prop(k: np.ndarray, n: int, G: int) -> np.ndarray:
    """Return k/n if n > 0, otherwise a NaN array of length G."""
    return k.astype(float) / n if n > 0 else np.full(G, np.nan)


def p_test(
    adata: ad.AnnData,
    cell_type_key: str = 'cell_type',
    region_key: str = 'region',
    interface_key: str = 'interface_map',
    neuron_patterns: tuple = ('Glut', 'GABA'),
    min_interface_prop: float = 0.01,
    exclude_regions: list = None,
    z_sig_thr: float = 3.0,
) -> ad.AnnData:
    """
    Region-wise proportions z-test comparing gene expression across three
    compartments: cells, interface bins, and extrasomatic background (ESB)
    bins.

    For every (region, gene) pair the function tests whether the proportion
    of expressing units (cells, interface bins, or ESB bins) assigned to a
    given region differs from those in all other regions combined, using a
    two-sided z-test for proportions.

    Spatial bins are assigned to regions via nearest-neighbour (Voronoi)
    mapping to the cells in ``adata.obsm['bin']``.  Cell types are
    coarse-grained into **Neuronal** (matching any ``neuron_patterns``) and
    **Non-neuronal** (everything else).

    ESB (extrasomatic background) bins are defined as ECM bins that are
    *not* interface bins and have non-zero total expression; all such bins
    are used directly as the population (no resampling).

    Results stored in ``adata.uns``:

    * ``regional_ptest`` – per-(region, gene) DataFrame with columns:

      - ``z_interface``, ``z_cell``, ``z_cell_neuronal``, ``z_cell_non_neuronal``, ``z_esb``
      - ``interface_prop_region``, ``interface_prop_other``
      - ``cell_prop_region`` / ``_neuronal`` / ``_non_neuronal``
      - ``cell_prop_other`` / ``_neuronal`` / ``_non_neuronal``
      - ``esb_prop_region``, ``esb_prop_other``
      - ``p_interface``, ``p_cell``, ``p_cell_neuronal``, ``p_cell_non_neuronal``, ``p_esb``

    * ``ptest_best_rank``              – per-gene best-ranking region using
      all-cell z-score vs interface z-score.
    * ``ptest_best_rank_neuronal``     – same, using neuronal cell z-score.
    * ``ptest_best_rank_non_neuronal`` – same, using non-neuronal cell z-score.

    Each best-rank table has columns:

      - ``best_region``     – region with the best interface rank for that gene.
      - ``rank_interface``      – interface rank within ``best_region``.
      - ``rank_cell``       – cell rank within ``best_region`` (same region).
      - ``z_interface``         – interface z-score in ``best_region``.
      - ``z_cell``          – cell z-score in ``best_region``.
      - ``z_esb``           – ESB z-score in ``best_region``.
      - ``n_sig_regions``   – number of regions where z_interface > ``z_sig_thr``.

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``adata.obs[cell_type_key]``, ``adata.obs[region_key]``,
        ``adata.obsm['bin']``, and ``adata.uns`` keys: ``binned_data``,
        ``binned_data_shape``, ``mask_ecm``, ``genes``, and ``interface_key``.
    cell_type_key : str
        Column in ``adata.obs`` with fine-grained cell-type labels.
        Default: ``'cell_type'``.
    region_key : str
        Column in ``adata.obs`` with anatomical region labels.
        Default: ``'region'``.
    interface_key : str
        Key in ``adata.uns`` for the binary interface map (H × W).
        Default: ``'interface_map'``.
    neuron_patterns : tuple of str
        Substrings (case-insensitive) used to identify neurons within
        ``cell_type_key``.  Cells matching any pattern are labelled
        **Neuronal**; all others are labelled **Non-neuronal**.
        Default: ``('Glut', 'GABA')``.
    min_interface_prop : float
        Minimum ``interface_prop_region`` a gene must have within a region
        to be included in the best-rank calculation.  Default: ``0.01``.
    exclude_regions : list of str, optional
        Regions to exclude from the best-rank tables only.  Z-tests are
        still computed for all regions (so excluded regions contribute
        to the "other" pool), but they will not appear as a gene's
        ``best_region``.  Useful for non-neuronal compartments such as
        ``['White Matter']``.  Default: ``None`` (no exclusions).
    z_sig_thr : float
        z-score threshold used to count ``n_sig_regions`` in the best-rank
        tables.  Default: ``3.0``.

    Returns
    -------
    ad.AnnData
        Input ``adata`` with ``adata.uns['regional_ptest']``,
        ``adata.uns['ptest_best_rank']``, ``adata.uns['ptest_best_rank_neuronal']``,
        and ``adata.uns['ptest_best_rank_non_neuronal']`` populated.
    """
    exclude_regions = set(exclude_regions or [])

    # ------------------------------------------------------------------
    # 1. Validate required obs columns
    # ------------------------------------------------------------------
    for key in [cell_type_key, region_key]:
        if key not in adata.obs.columns:
            raise ValueError(
                f"'{key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # ------------------------------------------------------------------
    # 2. Coarse-grain cell types: Neuronal vs Non-neuronal
    # ------------------------------------------------------------------
    ct = adata.obs[cell_type_key].astype(str)
    is_neuron = np.zeros(len(ct), dtype=bool)
    for pattern in neuron_patterns:
        is_neuron |= ct.str.contains(pattern, case=False, na=False).values
    coarse = np.where(is_neuron, 'Neuronal', 'Non-neuronal')

    # ------------------------------------------------------------------
    # 3. Build spatial expression arrays
    # ------------------------------------------------------------------
    array_usr = (
        adata.uns['binned_data']
        .toarray()
        .reshape(adata.uns['binned_data_shape'])
        * adata.uns['mask_ecm'][:, :, None]
    ).copy()

    H, W, G = array_usr.shape
    flat_expr = array_usr.reshape(-1, G)

    flat_syn     = adata.uns[interface_key].reshape(-1).astype(bool)
    flat_ecm     = adata.uns['mask_ecm'].reshape(-1).astype(bool)
    flat_nonzero = flat_expr.sum(axis=1) > 0

    # Binary presence per bin per gene
    bin_binary = flat_expr > 0   # (N_bins, G) bool

    # ESB pool: ECM, not interface, non-zero total expression
    esb_pool = flat_ecm & ~flat_syn & flat_nonzero

    # ------------------------------------------------------------------
    # 4. Spatial Voronoi: assign every pixel to the nearest cell's region
    # ------------------------------------------------------------------
    tree = cKDTree(adata.obsm['bin'])
    all_coords = np.argwhere(np.ones((H, W), dtype=bool))   # all HxW positions
    _, cell_idxs = tree.query(all_coords, k=1)              # shape (H*W,)

    regions_obs    = adata.obs[region_key].values
    unique_regions = np.unique(regions_obs)

    # build region -> list-of-coords mapping
    inverted_dict = {r: [] for r in unique_regions}
    for pixel_i, (row, col) in enumerate(all_coords):
        label = regions_obs[cell_idxs[pixel_i]]
        inverted_dict[label].append((row, col))

    # ------------------------------------------------------------------
    # 5. Cell-level binary expression (above 75th percentile)
    # ------------------------------------------------------------------
    cellular_exp  = adata.to_df()
    gene_thresh   = cellular_exp.quantile(0.75, axis=0)
    cell_bin_mat  = cellular_exp.gt(gene_thresh)[adata.uns['genes']].values  # (n_cells, G)

    genes_arr = np.asarray(adata.uns['genes'])

    # ------------------------------------------------------------------
    # 6. Main loop: per-region z-tests, vectorised over genes
    # ------------------------------------------------------------------
    region_dfs = []

    for region in unique_regions:
        r_mask  = regions_obs == region
        o_mask  = ~r_mask
        rn_mask = r_mask & (coarse == 'Neuronal')
        on_mask = o_mask & (coarse == 'Neuronal')
        rg_mask = r_mask & (coarse == 'Non-neuronal')
        og_mask = o_mask & (coarse == 'Non-neuronal')

        n_r  = int(r_mask.sum());  n_o  = int(o_mask.sum())
        n_rn = int(rn_mask.sum()); n_on = int(on_mask.sum())
        n_rg = int(rg_mask.sum()); n_og = int(og_mask.sum())

        if n_r == 0 or n_o == 0:
            continue

        coords = inverted_dict.get(region, [])
        if len(coords) == 0:
            continue

        coords    = np.asarray(coords)
        region_2d = np.zeros((H, W), dtype=bool)
        region_2d[coords[:, 0], coords[:, 1]] = True
        flat_reg  = region_2d.reshape(-1)

        # -- interface sub-masks --
        iA = flat_reg  & flat_syn
        iB = ~flat_reg & flat_syn
        nA_i = int(iA.sum()); nB_i = int(iB.sum())

        # -- ESB sub-masks --
        eA = flat_reg  & esb_pool
        eB = ~flat_reg & esb_pool
        nA_e = int(eA.sum()); nB_e = int(eB.sum())

        # ---- cell-level counts per gene ----
        k_r  = cell_bin_mat[r_mask].sum(axis=0)
        k_o  = cell_bin_mat[o_mask].sum(axis=0)
        k_rn = cell_bin_mat[rn_mask].sum(axis=0)
        k_on = cell_bin_mat[on_mask].sum(axis=0)
        k_rg = cell_bin_mat[rg_mask].sum(axis=0)
        k_og = cell_bin_mat[og_mask].sum(axis=0)

        z_cell, p_cell = _prop_ztest_vec(k_r,  k_o,  n_r,  n_o)
        z_cn,   p_cn   = _prop_ztest_vec(k_rn, k_on, n_rn, n_on)
        z_cg,   p_cg   = _prop_ztest_vec(k_rg, k_og, n_rg, n_og)

        # ---- interface bin counts per gene ----
        if nA_i > 0 and nB_i > 0:
            ki_A = bin_binary[iA].sum(axis=0)
            ki_B = bin_binary[iB].sum(axis=0)
            z_interface, p_interface = _prop_ztest_vec(ki_A, ki_B, nA_i, nB_i)
        else:
            ki_A = ki_B = np.zeros(G, dtype=int)
            z_interface = p_interface = np.full(G, np.nan)

        interface_prop_r = _safe_prop(ki_A, nA_i, G)
        interface_prop_o = _safe_prop(ki_B, nB_i, G)

        # ---- ESB bin counts per gene ----
        if nA_e > 0 and nB_e > 0:
            ke_A = bin_binary[eA].sum(axis=0)
            ke_B = bin_binary[eB].sum(axis=0)
            z_esb, p_esb = _prop_ztest_vec(ke_A, ke_B, nA_e, nB_e)
        else:
            ke_A = ke_B = np.zeros(G, dtype=int)
            z_esb = p_esb = np.full(G, np.nan)

        esb_prop_r = _safe_prop(ke_A, nA_e, G)
        esb_prop_o = _safe_prop(ke_B, nB_e, G)

        region_dfs.append(pd.DataFrame({
            'region':                        region,
            'gene':                          genes_arr,
            # z-scores: interface → cell → esb
            'z_interface':                   z_interface,
            'z_cell':                        z_cell,
            'z_cell_neuronal':               z_cn,
            'z_cell_non_neuronal':           z_cg,
            'z_esb':                         z_esb,
            # proportions: interface → cell → esb
            'interface_prop_region':         interface_prop_r,
            'interface_prop_other':          interface_prop_o,
            'cell_prop_region':              _safe_prop(k_r,  n_r,  G),
            'cell_prop_region_neuronal':     _safe_prop(k_rn, n_rn, G),
            'cell_prop_region_non_neuronal': _safe_prop(k_rg, n_rg, G),
            'cell_prop_other':               _safe_prop(k_o,  n_o,  G),
            'cell_prop_other_neuronal':      _safe_prop(k_on, n_on, G),
            'cell_prop_other_non_neuronal':  _safe_prop(k_og, n_og, G),
            'esb_prop_region':               esb_prop_r,
            'esb_prop_other':                esb_prop_o,
            # p-values: interface → cell → esb
            'p_interface':                   p_interface,
            'p_cell':                        p_cell,
            'p_cell_neuronal':               p_cn,
            'p_cell_non_neuronal':           p_cg,
            'p_esb':                         p_esb,
        }))

    df_cell_region = pd.concat(region_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 7. Compute best_rank: best-performing region per gene
    #
    #    For each gene, selects the region where it ranks best by
    #    interface z-score, then reports cell rank in that same region.
    #    Extra columns: z-scores at best region, n_sig_regions.
    # ------------------------------------------------------------------
    def _best_rank(df: pd.DataFrame, z_cell_col: str) -> pd.DataFrame:
        sub = df[
            (df['interface_prop_region'] >= min_interface_prop) &
            (~df['region'].isin(exclude_regions))
        ].copy()
        if sub.empty:
            return pd.DataFrame()

        # Rank within each region
        sub['_rank_cell']  = sub.groupby('region')[z_cell_col].rank(
            method='first', ascending=False)
        sub['_rank_interface'] = sub.groupby('region')['z_interface'].rank(
            method='first', ascending=False)
        sub = sub.dropna(subset=['_rank_interface'])

        # Best interface region per gene (lowest rank_interface, ties broken by region name)
        best = (
            sub.sort_values(['_rank_interface', 'region'])
            .groupby('gene', sort=False)
            .first()
            .reset_index()
        )

        # Number of regions where interface z exceeds threshold
        n_sig = (
            sub[sub['z_interface'] > z_sig_thr]
            .groupby('gene')['region']
            .nunique()
            .rename('n_sig_regions')
            .reset_index()
        )

        result = (
            best[['gene', 'region', '_rank_interface', '_rank_cell',
                  'z_interface', z_cell_col, 'z_esb']]
            .rename(columns={
                'region':          'best_region',
                '_rank_interface': 'rank_interface',
                '_rank_cell':      'rank_cell',
                z_cell_col:        'z_cell',
            })
            .merge(n_sig, on='gene', how='left')
        )
        result['n_sig_regions'] = (
            result['n_sig_regions'].fillna(0).astype(int).clip(lower=1)
        )
        return result.sort_values('rank_interface').reset_index(drop=True)

    best_rank              = _best_rank(df_cell_region, 'z_cell')
    best_rank_neuronal     = _best_rank(df_cell_region, 'z_cell_neuronal')
    best_rank_non_neuronal = _best_rank(df_cell_region, 'z_cell_non_neuronal')

    # ------------------------------------------------------------------
    # 8. Store results
    # ------------------------------------------------------------------
    adata.uns['regional_ptest']               = df_cell_region
    adata.uns['ptest_best_rank']              = best_rank
    adata.uns['ptest_best_rank_neuronal']     = best_rank_neuronal
    adata.uns['ptest_best_rank_non_neuronal'] = best_rank_non_neuronal
    return adata
