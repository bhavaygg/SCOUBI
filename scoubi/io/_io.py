import os
import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import coo_matrix
import types

class LazyTranscripts:
    """Lazy loader for transcripts parquet (cached after first load)."""
    def __init__(self, path):
        self.path = path
        self._df = None
    def load(self):
        if self._df is None:
            self._df = pd.read_parquet(self.path)
        return self._df
    def save(self):
        if self._df is not None:
            self._df.to_parquet(self.path)
        else:
            raise ValueError("DataFrame not loaded.")
    def __call__(self):
        return self.load()
    def __getattr__(self, name):
        raise AttributeError("Access only via .load()")
    def __getitem__(self, key):
        return self.load()[key]

def _attach_lazy_loader(adata):
    """Attach get_transcripts() method to adata in memory (not stored in .uns)."""
    def _get_transcripts(self):
        path = self.uns.get('transcripts_path', None)
        if path is None:
            raise KeyError("No 'transcripts_path' in adata.uns")
        if not hasattr(self, '_transcripts') or self._transcripts.path != path:
            self._transcripts = LazyTranscripts(path)
        return self._transcripts
    adata.get_transcripts = types.MethodType(_get_transcripts, adata)

def _summarize(self):
    """Structured summary of AnnData with grouped .uns fields.

    Groups are: data, annotations, maps, enrichment, comparison, testing,
    interface, communication, misc.
    Any .uns keys not in a known group are printed under 'other'.
    """
    print(f"AnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}")
    print("obs:", ", ".join(self.obs.columns.tolist()))
    print("obsm:", ", ".join(self.obsm.keys()))
    print("uns:")

    # --- Predefined grouping map ---
    group_map = {
        'data': [
            'transcripts_path', 'usr_label', 'data_shape', 'binned_data_shape',
            'mask_cell', 'mask_ecm', 'binned_data',
            'axon_markers', 'dendrite_markers', 'genes', 'lr_pairs'
        ],
        'annotations': ['bin_probabilities', 'bin_scores'],
        'maps': [
            'axon_map', 'dendrite_map', 'interface_map'
        ],
        'enrichment': ['axon_enrichment_df', 'dendrite_enrichment_df'],
        'comparison': ['ptest_a_vs_d'],
        'testing': [
            'empirical', 'regional_ptest',
            'ptest_best_rank', 'ptest_best_rank_neuronal', 'ptest_best_rank_non_neuronal'
        ],
        'interface': ['interface_cell_type_profile', 'interface_region_profile'],
        'communication': [
            'cellwhisper', 'cellwhisper_unfiltered', 'cellwhisper_lr', 'cellwhisper_regionwise',
            'communication_cell_type_profile', 'communication_region_profile', 'lr_edges'
        ],
        'misc': ['interface_knn_idx', 'interface_knn_dists', 'model_weights'],
    }

    # flatten group map for lookup
    assigned = set(k for keys in group_map.values() for k in keys)
    uns_keys = set(self.uns.keys())
    ungrouped = sorted(uns_keys - assigned)

    # print present keys per group
    for group, keys in group_map.items():
        present = [k for k in keys if k in self.uns]
        if present:
            print(f"  {group:<22}: {', '.join(present)}")

    if ungrouped:
        print(f"  {'other':<22}: {', '.join(ungrouped)}")


def _attach_summarize(adata):
    """Attach summarize() method to a single AnnData object in memory."""
    adata.summarize = types.MethodType(_summarize, adata)
    return adata

def load_data(
    filename: str,
    cell_type=None,
    region=None,
    qv_threshold=None,
    filter_genes=('BLANK', 'NegControl', 'DeprecatedCodeword', 'UnassignedCodeword', 'Blank'),
    usr_label="UNASSIGNED",
    overwrite=False,
) -> ad.AnnData:
    """
    Load spatial transcriptomics data and return an annotated AnnData object.

    Reads transcript-level data from a CSV, Parquet, or pre-built H5AD file,
    constructs a sparse cell × gene count matrix, stores spatial coordinates,
    writes a sidecar Parquet for lazy transcript access, and attaches two
    convenience methods to the returned object:

    * ``adata.summarize()``        – prints a structured overview (see :func:`_summarize`)
    * ``adata.get_transcripts()``  – lazily loads the transcript table (see :func:`_attach_lazy_loader`)

    Parameters
    ----------
    filename : str
        Path to the input file.  Accepted formats:

        * ``.parquet`` / ``.csv``  – raw transcript table with columns
          ``cell_id``, ``gene`` (or ``feature_name``), and optionally
          ``x_location``, ``y_location``, ``z_location``, ``qv``.
        * ``.h5ad`` – previously saved AnnData; loaded directly and
          returned after re-attaching the lazy-loader and summarize methods.
    cell_type : str, optional
        Path to a CSV file (index = cell IDs) containing a single column
        of cell-type labels.  Stored in ``adata.obs['cell_type']``.
    region : str, optional
        Path to a CSV file (index = cell IDs) containing a single column
        of region labels.  Stored in ``adata.obs['region']``.
    qv_threshold : float, optional
        Minimum quality-value score.  Transcripts with ``qv < qv_threshold``
        are dropped before matrix construction.  No filtering if ``None``.
    filter_genes : tuple[str, ...], optional
        Gene-name prefixes to exclude (e.g. blanks, negative controls).
        Default: ``('BLANK', 'NegControl', 'DeprecatedCodeword',
        'UnassignedCodeword', 'Blank')``.
    usr_label : str, optional
        Cell-ID value used to mark unassigned transcripts.  Rows with this
        label are excluded from the count matrix but retained in the saved
        transcript Parquet.  Default: ``'UNASSIGNED'``.
    overwrite : bool, optional
        If ``True``, the sidecar Parquet is written to `filename` directly
        instead of ``<stem>_scoubi.parquet``.  Default: ``False``.

    Returns
    -------
    ad.AnnData
        AnnData object with:

        * ``X``                  – sparse (CSR) cell × gene count matrix
        * ``obsm['spatial']``    – mean (x, y[, z]) coordinates per cell
        * ``uns['transcripts_path']`` – path to the sidecar Parquet
        * ``uns['usr_label']``        – unassigned transcript label (forwarded to :func:`bin_data`)
        * ``adata.summarize()``  – bound summary method
        * ``adata.get_transcripts()`` – bound lazy transcript accessor

    Raises
    ------
    ValueError
        If ``filename`` does not end with ``.csv``, ``.parquet``, or ``.h5ad``.

    See Also
    --------
    _summarize : Prints a grouped overview of all AnnData fields.
    _attach_lazy_loader : Attaches the ``get_transcripts()`` method.
    """
    # --- read input ---
    if filename.endswith('.parquet'):
        df = pd.read_parquet(filename)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.h5ad'):
        adata = ad.read_h5ad(filename)
        if 'transcripts_path' in adata.uns:
            _attach_lazy_loader(adata)
        _attach_summarize(adata)
        return adata
    else:
        raise ValueError("Unsupported file format. Use .csv, .parquet, or .h5ad.")

    # --- filter ---
    if 'qv' in df.columns and qv_threshold is not None:
        df = df[df.qv >= qv_threshold]
    if 'feature_name' in df.columns:
        df = df.rename(columns={'feature_name':'gene'})
    df = df[~df['gene'].str.startswith(tuple(filter_genes), na=False)]
    df.sort_values(by=['cell_id'], inplace=True, ascending=True)
    df_cell = df[df.cell_id != usr_label].copy()

    # --- build sparse counts from df_cell ---
    cell_codes, cell_uniques = pd.factorize(df_cell['cell_id'].astype(str))
    gene_codes, gene_uniques = pd.factorize(df_cell['gene'].astype(str))
    data = np.ones(len(df_cell), dtype=np.int32)
    X = coo_matrix((data, (cell_codes, gene_codes)),
                   shape=(len(cell_uniques), len(gene_uniques))).tocsr()

    obs = pd.DataFrame(index=cell_uniques.astype(str))
    var = pd.DataFrame(index=gene_uniques.astype(str))
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # --- spatial coordinates ---
    loc_cols = ['x_location','y_location']
    if 'z_location' in df_cell.columns:
        loc_cols.append('z_location')
    if loc_cols:
        spatial = df_cell.groupby(df_cell['cell_id'].astype(str))[loc_cols].mean()
        spatial = spatial.reindex(obs.index)
        adata.obsm['spatial'] = spatial.values

    # --- save transcripts parquet (full df) ---
    if not overwrite:
        parquet_path = os.path.abspath(os.path.splitext(filename)[0] + "_scoubi.parquet")
    else:
        parquet_path = os.path.abspath(filename)
    df.to_parquet(parquet_path, index=False)
    adata.uns['transcripts_path'] = parquet_path
    adata.uns['usr_label'] = usr_label
    _attach_lazy_loader(adata)
    _attach_summarize(adata)
    ad.AnnData.summarize = _summarize
    adata._transcripts = LazyTranscripts(parquet_path)  # optional pre-cache

    # --- cell type annotations ---
    if cell_type is not None:
        df_ct = pd.read_csv(cell_type, index_col=0)
        df_ct.index = df_ct.index.astype(str)
        adata.obs['cell_type'] = df_ct.reindex(obs.index)
    if region is not None:
        df_rg = pd.read_csv(region, index_col=0)
        df_rg.index = df_rg.index.astype(str)
        adata.obs['region'] = df_rg.reindex(obs.index)

    del df, X, data
    return adata
