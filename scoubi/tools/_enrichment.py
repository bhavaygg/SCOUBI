import numpy as np
import anndata as ad
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.cm as cm
from scipy import stats
from ..model import get_zscore, do_conv, _prep_dict, kernel
import pickle
from ..plotting import size_mapping, categorize_gene
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

datasets = {"BP": "GO:0008150", "CC": "GO:0005575", "MF": "GO:0003674"}

def get_panther(geneset, background, organism = 10090):
    '''
        Perform GO enrichment analysis using PANTHERDB.
        Arguments
        ----------
        geneset : str
            Geneset to perform the analysis.
        background : str
            Background genes.
        dataset : str
            Dataset to perform the analysis.
        organism : str
            Organism to perform the analysis.
        setting : str
            Setting to perform the analysis. Either 'module' or 'cell'.
    '''
    df = pd.DataFrame()
    r_session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    r_session.mount('http://', HTTPAdapter(max_retries=retries))
    for dataset in datasets:
        params = {
            'geneInputList': geneset,
            'organism': organism,
            'refInputList': background,
            'refOrganism': organism,
            'annotDataSet': datasets[dataset],
            'enrichmentTestType': 'FISHER',
            'correction': 'FDR'
        }
        x = r_session.post(f"https://pantherdb.org/services/oai/pantherdb/enrich/overrep", data=params)
        rows = []
        for i in x.json()['results']['result']:
            if i['number_in_list'] > 0:
                try:
                    rows.append([i['number_in_list'], i['fold_enrichment'], i['fdr'], i['expected'],
                            i['number_in_reference'], i['pValue'], i['term']['label'], i['term']['id']])
                except:
                    rows.append([i['number_in_list'], i['fold_enrichment'], i['fdr'], i['expected'],
                            i['number_in_reference'], i['pValue'], i['term']['label'], np.nan])
        df_ds = pd.DataFrame(rows, columns = ['number_in_list', 'fold_enrichment', 'fdr', 'expected', 'number_in_reference', 'pValue', 'term', 'id'])
        df_significant = df_ds[(df_ds.pValue < 0.05) & (df_ds.fold_enrichment > 1)].sort_values(by=['pValue'])
        df_significant['dataset'] = dataset
        df = pd.concat([df, df_significant])
    df.sort_values(by=['pValue'], inplace=True)
    df = df[df.number_in_reference > 10].copy()
    fdr_new = stats.false_discovery_control(df.pValue.values)
    df['fdr'] = fdr_new
    df['id'] = df['id'].astype('str')
    return df

def axon_dendrite_enrichment(
    adata: ad.AnnData,
    z_thr: float = 5.0,
) -> ad.AnnData:
    """
    Compute axon-vs-dendrite proportions z-test, compartment specificity
    DataFrames, and derived enrichment statistics.

    The function does the following in one pass:

    1. **Proportions z-test** (vectorised over all genes): for each gene,
       tests whether its expression rate in axonic bins differs from its rate
       in dendritic bins.  All bins of the same compartment are pooled; the
       test statistic is the standard two-proportions z-score.

    2. **Compartment specificity**: ``df_axon`` / ``df_dendrite`` give each
       gene's fraction of total spatial expression found in axon / dendrite
       bins, respectively.

    3. **Derived columns** added to the z-test result:
       ``p1``, ``p2``, ``log2fc``, ``mean_prop``, ``label``, ``is_marker``.

    Results stored in ``adata.uns``:

    * ``proportion['a_vs_d']``   – z-test DataFrame (gene, c1, c2, n1, n2,
      zscore, pvalue, fdr, p1, p2, log2fc, mean_prop, label, is_marker)
    * ``axon_enrichment_df``     – compartment specificity for axonic bins
    * ``dendrite_enrichment_df`` – compartment specificity for dendritic bins

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``adata.uns`` keys: ``binned_data``, ``binned_data_shape``,
        ``mask_ecm``, ``axon_map``, ``dendrite_map``, ``genes``.
    z_thr : float
        Z-score threshold for the ``label`` column.  Genes with
        ``zscore > z_thr`` are labelled ``'axon'``, genes with
        ``zscore < -z_thr`` are labelled ``'dendrite'``, and the rest ``'ns'``.

    Returns
    -------
    ad.AnnData
        Input ``adata`` with the new keys populated.
    """
    from scipy.stats import norm as _norm

    array_usr = (
        adata.uns['binned_data']
        .toarray()
        .reshape(adata.uns['binned_data_shape'])
        * adata.uns['mask_ecm'][:, :, None]
    ).copy()

    axon_map     = adata.uns['axon_map']
    dendrite_map = adata.uns['dendrite_map']
    genes        = list(adata.uns['genes'])

    # --- expression sums per compartment ------------------------------------
    total_expr = array_usr.sum(axis=(0, 1))  # (G,)
    binary_expr = (array_usr > 0).astype(float)
    axon_expr = binary_expr[axon_map == 1].sum(axis=0)      # (G,)
    dendrite_expr = binary_expr[dendrite_map == 1].sum(axis=0)  # (G,)

    n1 = int((axon_map == 1).sum())      # total axon bins
    n2 = int((dendrite_map == 1).sum())  # total dendrite bins

    # --- vectorised proportions z-test --------------------------------------
    c1 = axon_expr.astype(float)
    c2 = dendrite_expr.astype(float)
    p1 = c1 / n1
    p2 = c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z = np.zeros_like(se, dtype=float)
    np.divide(p1 - p2, se, out=z, where=se > 0)
    pv = 2.0 * (1.0 - _norm.cdf(np.abs(z)))

    _, fdr, _, _ = multipletests(np.nan_to_num(pv, nan=1.0), method='bonferroni')

    # --- proportions z-test DataFrame ---------------------------------------
    df_z = pd.DataFrame({
        'gene':      genes,
        'c1':        c1,
        'c2':        c2,
        'n1':        n1,
        'n2':        n2,
        'zscore':    z,
        'pvalue':    pv,
        'fdr':       fdr,
        'p1':        p1,
        'p2':        p2,
        'log2fc':    np.log2((p1 + 1e-12) / (p2 + 1e-12)),
        'mean_prop': (p1 + p2) / 2,
    }).sort_values('zscore', ascending=False, ignore_index=True)

    df_z['label'] = 'ns'
    df_z.loc[df_z['zscore'] >  z_thr, 'label'] = 'axon'
    df_z.loc[df_z['zscore'] < -z_thr, 'label'] = 'dendrite'

    marker_genes = (
        set(adata.uns.get('axon_markers', []))
        | set(adata.uns.get('dendrite_markers', []))
    )
    df_z['is_marker'] = df_z['gene'].isin(marker_genes)

    # --- compartment specificity DataFrames ---------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        axon_frac = np.where(total_expr > 0, axon_expr / total_expr, 0.0)
        dendrite_frac = np.where(total_expr > 0, dendrite_expr / total_expr, 0.0)

    df_axon     = pd.DataFrame({'count': axon_frac},     index=genes)
    df_dendrite = pd.DataFrame({'count': dendrite_frac}, index=genes)

    # --- store results ------------------------------------------------------
    # if 'proportion' not in adata.uns:
    #     adata.uns['proportion'] = {}
    adata.uns['ptest_a_vs_d']   = df_z
    adata.uns['axon_enrichment_df']     = df_axon
    adata.uns['dendrite_enrichment_df'] = df_dendrite
    return adata
