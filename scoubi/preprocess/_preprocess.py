import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp
import sparse


def bin_data(adata: ad.AnnData, binsize=5, debug=False):
    """
    Bin transcript-level data into a spatial grid and build a 3-D gene-expression tensor.

    Reads the raw transcript table via ``adata.get_transcripts()``, assigns each
    transcript to a 2-D spatial bin, and constructs a sparse (X_BINS × Y_BINS × n_genes)
    count array.  Binary ECM and cell masks are derived from unassigned vs. assigned
    transcripts.  The updated transcript table (with bin columns added) is written back
    to the sidecar Parquet.

    Parameters
    ----------
    adata : ad.AnnData
        Must have ``adata.uns['transcripts_path']`` (set by :func:`scoubi.io.load_data`)
        and a bound ``get_transcripts()`` method.  The transcript table must contain
        columns ``x_location``, ``y_location``, ``cell_id``, and ``gene``.
    binsize : int or float, optional
        Physical size of each square bin in the same units as ``x_location`` /
        ``y_location``.  Smaller values give finer resolution at higher memory cost.
        Default: 5.
    debug : bool, optional
        Reserved for future use.  Default: False.

    Returns
    -------
    ad.AnnData
        The input ``adata`` updated in-place with:

        * ``uns['binned_data']``       – CSR sparse matrix (X_BINS*Y_BINS × n_genes)
        * ``uns['binned_data_shape']`` – ``[X_BINS, Y_BINS, n_genes]``
        * ``uns['genes']``             – list of gene names (column order of the tensor)
        * ``uns['mask_ecm']``          – binary (X_BINS × Y_BINS) ECM mask
        * ``uns['mask_cell']``         – binary (X_BINS × Y_BINS) cell mask
        * ``obsm['bin']``              – (n_cells × 2) mean binned coordinates per cell
    """
    df = adata.get_transcripts().load()

    # -----------------------
    # 1) Compute bin edges
    # -----------------------
    x_min, x_max = df.x_location.min(), df.x_location.max()
    y_min, y_max = df.y_location.min(), df.y_location.max()

    X_BINS = int((x_max - x_min) / binsize)
    Y_BINS = int((y_max - y_min) / binsize)

    x_bins = np.linspace(x_min, x_max, X_BINS + 1)
    y_bins = np.linspace(y_min, y_max, Y_BINS + 1)

    # Assign bins
    df["x_binned"] = np.digitize(df.x_location, x_bins) - 1
    df["y_binned"] = np.digitize(df.y_location, y_bins) - 1

    df["x_binned"] = df["x_binned"].clip(0, X_BINS - 1)
    df["y_binned"] = df["y_binned"].clip(0, Y_BINS - 1)

    # -----------------------
    # 2) Gene mapping
    # -----------------------
    genes = np.unique(df["gene"])
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    df["gene_idx"] = df["gene"].map(gene_to_idx)
    n_genes = len(genes)

    adata.uns["genes"] = list(genes)

    # -----------------------
    # 3) Build sparse 3-D tensor
    # -----------------------
    xs = df["x_binned"].to_numpy()
    ys = df["y_binned"].to_numpy()
    gs = df["gene_idx"].to_numpy()
    vals = np.ones(len(df), dtype=np.int16)

    array_3d_sparse = sparse.COO(
        coords=np.vstack([xs, ys, gs]),
        data=vals,
        shape=(X_BINS, Y_BINS, n_genes),
    )

    # -----------------------
    # 4) Save in AnnData safely
    # -----------------------
    # Flatten to 2-D for AnnData (SciPy cannot reshape sparse 3-D)
    binned_2d = array_3d_sparse.reshape((X_BINS * Y_BINS, n_genes)).todense()
    binned_2d = sp.csr_matrix(binned_2d)
    #Test THis
    # binned_2d = array_3d_sparse.reshape((X_BINS * Y_BINS, n_genes))
    # binned_2d = binned_2d.tocsr()

    adata.uns["binned_data"] = binned_2d
    adata.uns["binned_data_shape"] = [X_BINS, Y_BINS, n_genes]

    # -----------------------
    # 5) ECM and cell masks
    # -----------------------
    binary_ecm = np.zeros((X_BINS, Y_BINS), dtype=np.int8)
    binary_cell = np.zeros((X_BINS, Y_BINS), dtype=np.int8)

    usr_label = adata.uns.get('usr_label', 'UNASSIGNED')
    ecm = df[df.cell_id == usr_label]
    cell = df[df.cell_id != usr_label]

    binary_ecm[ecm.x_binned, ecm.y_binned] = 1
    binary_cell[cell.x_binned, cell.y_binned] = 1

    # remove ECM if cell count >= ECM count
    binary_ecm[binary_cell >= binary_ecm] = 0

    adata.uns["mask_ecm"] = binary_ecm
    adata.uns["mask_cell"] = binary_cell

    # -----------------------
    # 6) Cell-level AnnData object
    # -----------------------
    df_st = df[df.cell_id != "UNASSIGNED"]

    adata_st = pd.crosstab(df_st.cell_id, df_st.gene)
    adata_st = ad.AnnData(adata_st)

    cell_coords = df_st.groupby(["cell_id"])[["x_binned", "y_binned"]].mean()
    aligned = cell_coords.reindex(adata_st.obs.index)
    adata.obsm["bin"] = aligned.to_numpy()
    adata.get_transcripts().save()
    return adata
