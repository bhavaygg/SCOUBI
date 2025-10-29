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
    def __call__(self):
        return self.load()
    def __getattr__(self, name):
        return getattr(self.load(), name)
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
    """Structured summary of AnnData with grouped .uns fields."""
    print(f"AnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}")
    print("obs:", ", ".join(self.obs_keys()))
    print("obsm:", ", ".join(self.obsm_keys()))
    print("uns:")

    # --- Predefined grouping map ---
    group_map = {
        'data': [
            'transcripts_path', 'data_shape', 'mask_cell', 'mask_ecm', 'binned_data', 'axon_markers', 'dendrite_markers', 'genes', 'lr_pairs'
        ],
        'annotations': ['bin_probabilities'],
        'maps': [
            'axon_map', 'dendrite_map', 'presynapse_map', 'postsynapse_map'
        ],
        'enrichment': ['axon_enrichment', 'dendrite_enrichment',
                       'presynapse_enrichment', 'postsynapse_enrichment'],
        'comparison': ['a_vs_d', 'n_vs_s', 'pre_vs_post'],
        'synapse': ['synapse_map', 'synapse_cell_type_profile', 'synapse_cell_type_region_profile', 'synapse_region_profile'],
        'axonic-profiles': [
            'axon_cell_type_profile', 'axon_cell_type_region_profile', 'presynapse_cell_type_profile', 'presynapse_cell_type_region_profile',
            'axon_region_profile', 'presynapse_region_profile'
        ],
        'dendritic-profiles': [
            'dendrite_cell_type_profile', 'dendrite_cell_type_region_profile',
            'postsynapse_cell_type_profile', 'postsynapse_cell_type_region_profile',
            'dendrite_region_profile', 'postsynapse_region_profile'
        ],
        'communication': ['cellwhisper', 'cellwhisper_lr', 'communication_cell_type_profile', 'communication_cell_type_region_profile',
                          'communication_region_profile', 'lr_edges'],
        'misc': ['knn_idx', 'knn_dists', 'presynapse_knn_idx', 'presynapse_knn_dists',
                 'postsynapse_knn_idx', 'postsynapse_knn_dists', 'model_weights'],
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

def load_data(filename: str, cell_type=None, region = None, qv_thresold=None,
              filter_genes=('BLANK','NegControl','DeprecatedCodeword','UnassignedCodeword','Blank'),
              usr_label="UNASSIGNED", overwrite = False) -> ad.AnnData:
    """
    Load data from CSV/Parquet/H5AD, build sparse cell x gene matrix,
    and store transcripts lazily.
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
    if 'qv' in df.columns and qv_thresold is not None:
        df = df[df.qv >= qv_thresold]
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
        spatial = df_cell.groupby(df_cell['cell_id'].astype(str))[loc_cols].first()
        spatial = spatial.reindex(obs.index)
        adata.obsm['spatial'] = spatial

    # --- save transcripts parquet (full df) ---
    if not overwrite:
        parquet_path = os.path.splitext(filename)[0] + "_scoubi.parquet"
    else:
        parquet_path = filename
    df.to_parquet(parquet_path, index=False)
    adata.uns['transcripts_path'] = parquet_path
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

    return adata
