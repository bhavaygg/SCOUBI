import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sparse


def get_binned_data_3d(adata: ad.AnnData) -> sparse.COO:
    """Reconstruct the 3-D sparse binned tensor stored in ``adata.uns``.

    The tensor is stored in AnnData as COO components rather than as a live
    ``sparse.COO`` object so it can survive H5AD save/load round trips.

    Parameters
    ----------
    adata : AnnData
        Object containing ``uns['binned_data_3d']`` with ``coords``, ``data``,
        and ``shape`` arrays.

    Returns
    -------
    sparse.COO
        Sparse tensor with shape ``(x_bins, y_bins, n_genes)``.
    """
    try:
        payload = adata.uns["binned_data_3d"]
    except KeyError as exc:
        raise KeyError("No 'binned_data_3d' found in adata.uns") from exc

    required = {"coords", "data", "shape"}
    missing = required.difference(payload)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise KeyError(f"adata.uns['binned_data_3d'] is missing: {missing_keys}")

    return sparse.COO(
        coords=np.asarray(payload["coords"]),
        data=np.asarray(payload["data"]),
        shape=tuple(np.asarray(payload["shape"], dtype=np.int64).tolist()),
    )


def get_binned_data_2d(adata: ad.AnnData) -> sp.csr_matrix:
    """Return the flattened bin-by-gene matrix reconstructed without densifying.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix with shape ``(x_bins * y_bins, n_genes)``.
    """
    array_3d = get_binned_data_3d(adata)
    x_bins, y_bins, n_genes = array_3d.shape
    array_2d = array_3d.reshape((x_bins * y_bins, n_genes))
    return array_2d.to_scipy_sparse().tocsr()


def _store_binned_data_3d(adata: ad.AnnData, array_3d: sparse.COO) -> None:
    """Store COO internals using AnnData/H5AD-compatible objects."""
    adata.uns["binned_data_3d"] = {
        "coords": np.asarray(array_3d.coords, dtype=np.int32),
        "data": np.asarray(array_3d.data),
        "shape": np.asarray(array_3d.shape, dtype=np.int64),
        "format": "sparse.COO",
    }


def _store_binned_data_3d_from_flat_coo(
    adata: ad.AnnData,
    array_2d: sp.coo_matrix,
    y_bins: int,
    shape_3d: tuple[int, int, int],
) -> None:
    """Store 3-D COO components from a flattened bin-by-gene COO matrix."""
    adata.uns["binned_data_3d"] = {
        "coords": np.vstack([
            array_2d.row // y_bins,
            array_2d.row % y_bins,
            array_2d.col,
        ]).astype(np.int32, copy=False),
        "data": np.asarray(array_2d.data),
        "shape": np.asarray(shape_3d, dtype=np.int64),
        "format": "sparse.COO",
    }


def bin_data(adata: ad.AnnData, binsize=5, debug=False):
    """
    Bin transcript-level data into a sparse spatial gene-expression grid.

    Reads the raw transcript table via ``adata.get_transcripts()``, assigns each
    transcript to a 2-D spatial bin, and constructs sparse binned counts without
    materializing the full ``(x_bins, y_bins, n_genes)`` dense array. The
    flattened bin-by-gene matrix is stored in ``uns['binned_data']`` as CSR. The
    equivalent 3-D tensor is stored in AnnData-safe COO components under
    ``uns['binned_data_3d']``.

    Parameters
    ----------
    adata : AnnData
        Must have a bound ``get_transcripts()`` method returning a transcript
        table with ``x_location``, ``y_location``, ``cell_id``, and ``gene``.
        ``uns['usr_label']`` is used as the unassigned/extrasomatic transcript
        label, defaulting to ``'UNASSIGNED'`` when absent.
    binsize : int or float, optional
        Spatial bin size in the same units as ``x_location`` and ``y_location``.
        Default: 5.
    debug : bool, optional
        Reserved for compatibility. Currently unused.

    Returns
    -------
    AnnData
        Input object updated in-place with:

        * ``uns['binned_data']`` – CSR matrix, ``x_bins * y_bins`` by genes
        * ``uns['binned_data_3d']`` – COO ``coords``, ``data``, ``shape``
        * ``uns['binned_data_shape']`` – ``[x_bins, y_bins, n_genes]``
        * ``uns['genes']`` – gene order used by both binned representations
        * ``uns['mask_ecm']`` and ``uns['mask_cell']`` – binary spatial masks
        * ``obsm['bin']`` – mean binned coordinate per cell, aligned to obs
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

    # Assign bins with the same semantics as np.digitize on increasing bins.
    xs = np.searchsorted(x_bins, df.x_location.to_numpy(), side="right") - 1
    ys = np.searchsorted(y_bins, df.y_location.to_numpy(), side="right") - 1
    xs = np.clip(xs, 0, X_BINS - 1).astype(np.int32, copy=False)
    ys = np.clip(ys, 0, Y_BINS - 1).astype(np.int32, copy=False)

    df["x_binned"] = xs
    df["y_binned"] = ys

    # -----------------------
    # 2) Gene mapping
    # -----------------------
    gs, genes = pd.factorize(df["gene"], sort=True)
    gs = gs.astype(np.int32, copy=False)
    genes = np.asarray(genes)
    n_genes = len(genes)

    adata.uns["genes"] = list(genes)

    # -----------------------
    # 3) Build and store sparse 3-D tensor
    # -----------------------
    vals = np.ones(len(df), dtype=np.int32)
    flat_bins = xs.astype(np.int64, copy=False) * Y_BINS + ys
    binned_2d = sp.coo_matrix(
        (vals, (flat_bins, gs)),
        shape=(X_BINS * Y_BINS, n_genes),
    ).tocsr()
    binned_2d.sum_duplicates()

    adata.uns["binned_data"] = binned_2d
    adata.uns["binned_data_shape"] = [X_BINS, Y_BINS, n_genes]
    _store_binned_data_3d_from_flat_coo(
        adata,
        binned_2d.tocoo(copy=False),
        Y_BINS,
        (X_BINS, Y_BINS, n_genes),
    )

    # -----------------------
    # 4) ECM and cell masks
    # -----------------------
    binary_ecm = np.zeros((X_BINS, Y_BINS), dtype=np.int8)
    binary_cell = np.zeros((X_BINS, Y_BINS), dtype=np.int8)

    usr_label = adata.uns.get("usr_label", "UNASSIGNED")
    is_ecm = df.cell_id.eq(usr_label).to_numpy()

    binary_ecm[xs[is_ecm], ys[is_ecm]] = 1
    binary_cell[xs[~is_ecm], ys[~is_ecm]] = 1

    # remove ECM if cell count >= ECM count
    binary_ecm[binary_cell >= binary_ecm] = 0

    adata.uns["mask_ecm"] = binary_ecm
    adata.uns["mask_cell"] = binary_cell

    # -----------------------
    # 5) Cell-level binned coordinates
    # -----------------------
    df_st = df.loc[~is_ecm, ["cell_id", "x_binned", "y_binned"]]
    cell_coords = df_st.groupby(["cell_id"])[["x_binned", "y_binned"]].mean()
    aligned = cell_coords.reindex(adata.obs.index)
    adata.obsm["bin"] = aligned.to_numpy()
    return adata
