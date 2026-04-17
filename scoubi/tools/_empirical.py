import numpy as np
import anndata as ad
import pandas as pd


def empirical_background(
    adata: ad.AnnData,
    n_boot: int = 1000,
    n_strata: int = 10,
    interface_key: str = 'interface_map',
    random_seed: int = 0,
) -> ad.AnnData:
    """
    Build an empirical null distribution from library-size-matched background
    (extrasomatic) bins and compare interface bin expression against it.

    Background bins are defined as bins that fall within the ECM mask
    (``mask_ecm``), are **not** part of the interface (``interface_key``),
    and have non-zero total expression.  For each bootstrap iteration the
    background pool is stratified by library size to match the library-size
    distribution of the interface bins, then sampled without replacement.

    Results are stored in ``adata.uns['empirical']``:

    ``df_empirical`` – DataFrame with per-gene columns:

    * ``gene``                      – gene name
    * ``Interface Mean Expression`` – mean expression in interface bins
    * ``ESR Mean Expression``       – mean bootstrapped background expression
    * ``empirical_pvalue``          – empirical one-sided p-value
      (fraction of bootstrap samples with mean >= interface mean)
    * ``zscore``                    – (interface mean - background mean) /
      background std across bootstrap iterations
    * ``is_marker``                 – True if the gene is in
      ``axon_markers`` or ``dendrite_markers``
    * ``is_LR``                     – True if the gene appears in any
      significant ligand-receptor pair (``cellwhisper_lr``)

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.  Must contain keys populated by
        ``scoubi.tl.overview``: ``binned_data``, ``binned_data_shape``,
        ``mask_ecm``, and whatever map is referenced by ``interface_key``.
    n_boot : int
        Number of bootstrap iterations.  Default: 1000.
    n_strata : int
        Number of library-size quantile strata used for stratified
        sampling.  Default: 10.
    interface_key : str
        Key in ``adata.uns`` that holds the binary interface map (H x W).
        Default: ``'interface_map'``.
    random_seed : int
        Numpy random seed for reproducibility.  Default: 0.

    Returns
    -------
    ad.AnnData
        The input ``adata`` with ``adata.uns['empirical']`` set to the
        resulting ``df_empirical`` DataFrame.
    """
    rng = np.random.default_rng(random_seed)

    array_usr = (
        adata.uns['binned_data']
        .toarray()
        .reshape(adata.uns['binned_data_shape'])
        * adata.uns['mask_ecm'][:, :, None]
    ).copy()

    H, W, G = array_usr.shape
    flat_expr = array_usr.reshape(-1, G)

    flat_syn = adata.uns[interface_key].reshape(-1).astype(bool)
    flat_ecm = adata.uns['mask_ecm'].reshape(-1).astype(bool)
    flat_nonzero = flat_expr.sum(axis=1) > 0

    # Interface (A) group
    A_mask = flat_syn
    A_mean = flat_expr[A_mask].mean(axis=0)

    # Library sizes
    libsize = flat_expr.sum(axis=1)
    libsize_A = libsize[A_mask]

    # Background pool: ECM bins, not interface, with non-zero expression
    C_pool = np.flatnonzero(flat_ecm & ~flat_syn & flat_nonzero)
    libsize_C = libsize[C_pool]

    # Stratify by library size
    bins = np.quantile(libsize_A, np.linspace(0, 1, n_strata + 1))
    bins[0] -= 1e-6

    A_strata = np.digitize(libsize_A, bins) - 1
    C_strata = np.digitize(libsize_C, bins) - 1

    C_boot = np.zeros((n_boot, G))
    for b in range(n_boot):
        sampled_idx = []
        for s in range(n_strata):
            n_s = int(np.sum(A_strata == s))
            if n_s == 0:
                continue
            pool_s = C_pool[C_strata == s]
            if len(pool_s) < n_s:
                continue
            sampled_idx.append(rng.choice(pool_s, size=n_s, replace=False))
        if len(sampled_idx) == 0:
            continue
        sampled_idx = np.concatenate(sampled_idx)
        C_boot[b] = flat_expr[sampled_idx].mean(axis=0)

    C_mean = C_boot.mean(axis=0)
    sd_C = C_boot.std(axis=0, ddof=1)

    emp_pval = (
        (C_boot >= A_mean[None, :]).sum(axis=0) + 1
    ) / (n_boot + 1)

    zscore = (A_mean - C_mean) / np.where(sd_C > 0, sd_C, np.nan)

    df_empirical = pd.DataFrame({
        'gene': adata.uns['genes'],
        'Interface Mean Expression': A_mean,
        'ESR Mean Expression': C_mean,
        'empirical_pvalue': emp_pval,
        'zscore': zscore,
    })

    marker_genes = (
        set(adata.uns.get('axon_markers', []))
        | set(adata.uns.get('dendrite_markers', []))
    )
    lr_genes = set(
        gene for pair in adata.uns.get('cellwhisper_lr', []) for gene in pair
    )
    df_empirical['is_marker'] = df_empirical['gene'].isin(marker_genes)
    df_empirical['is_LR']     = df_empirical['gene'].isin(lr_genes)

    adata.uns['empirical'] = df_empirical
    return adata
