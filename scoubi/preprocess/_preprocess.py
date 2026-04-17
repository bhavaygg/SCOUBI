# import pandas as pd
# import anndata as ad
# import numpy as np
# import sparse


# def bin_data(adata: ad.AnnData, binsize=5):
#     df = adata.get_transcripts().load()
#     x_min, x_max = df.x_location.min(), df.x_location.max()
#     y_min, y_max = df.y_location.min(), df.y_location.max()
#     X_BINS = int((x_max - x_min) / binsize)
#     Y_BINS = int((y_max - y_min) / binsize)

#     x_bins = np.linspace(x_min, x_max, X_BINS + 1)
#     y_bins = np.linspace(y_min, y_max, Y_BINS + 1)

#     genes = np.unique(df['gene'])
#     adata.uns['genes'] = list(genes)
#     n_genes = len(genes)

#     # Compute the dense 3D array per gene
#     array_3d = np.zeros((X_BINS, Y_BINS, n_genes), dtype=np.int16)
#     for idx, (gene, group) in enumerate(df.groupby('gene')):
#         H, _, _ = np.histogram2d(group['x_location'], group['y_location'], bins=[x_bins, y_bins])
#         array_3d[:, :, idx] = H

#     # Compute ECM (UNASSIGNED) and cell (assigned) binary matrices
#     # ECM
#     df_ecm = df[df.cell_id == "UNASSIGNED"].copy()
#     df_ecm['x_binned'] = pd.cut(df_ecm['x_location'], bins=x_bins, include_lowest=True)
#     df_ecm['y_binned'] = pd.cut(df_ecm['y_location'], bins=y_bins, include_lowest=True)
#     pivot_ecm = pd.pivot_table(
#         df_ecm, index='y_binned', columns='x_binned', aggfunc='size', fill_value=0, observed=False
#     )

#     # Cell
#     df_cell = df[df.cell_id != "UNASSIGNED"].copy()
#     df_cell['x_binned'] = pd.cut(df_cell['x_location'], bins=x_bins, include_lowest=True)
#     df_cell['y_binned'] = pd.cut(df_cell['y_location'], bins=y_bins, include_lowest=True)
#     pivot_cell = pd.pivot_table(
#         df_cell, index='y_binned', columns='x_binned', aggfunc='size', fill_value=0, observed=False
#     )

#     df['x_binned'] = pd.cut(df['x_location'], bins=x_bins, include_lowest=True, labels = False)
#     df['y_binned'] = pd.cut(df['y_location'], bins=y_bins, include_lowest=True, labels = False)
#     df_st = df[df.cell_id != "UNASSIGNED"]
#     adata_st = pd.crosstab(df_st.cell_id, df_st.gene)
#     adata_st = ad.AnnData(adata_st)
#     cell_coords = df_st.groupby('cell_id')[['x_binned', 'y_binned']].mean()
#     aligned_coords = cell_coords.reindex(adata_st.obs.index)
#     adata.obsm['bin'] = aligned_coords.to_numpy()
#     df.to_parquet(adata.uns['transcripts_path'], index=False)
    
#     # Remove ECM bins where cell count >= ECM count
#     pivot_ecm[pivot_cell >= pivot_ecm] = 0

#     # Convert to binary matrices (transpose to match X/Y orientation)
#     binary_matrix_ecm = (pivot_ecm > 0).astype(int).T.values
#     binary_matrix_cell = (pivot_cell > 0).astype(int).T.values
#     adata.uns['binned_data'] = array_3d
#     adata.uns['mask_ecm'] = binary_matrix_ecm
#     adata.uns['mask_cell'] = binary_matrix_cell
#     return adata

import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp
import sparse


def bin_data(adata: ad.AnnData, binsize=5, debug=False):
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

    ecm = df[df.cell_id == "UNASSIGNED"]
    cell = df[df.cell_id != "UNASSIGNED"]

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
