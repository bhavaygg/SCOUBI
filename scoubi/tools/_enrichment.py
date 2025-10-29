import numpy as np
import anndata as ad
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.cm as cm
from scipy import stats
import scanpy as sc
from ..model import get_zscore, do_conv, _prep_dict, kernel
import pickle
from ..plotting import size_mapping, categorize_gene

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

def compare(adata: ad.AnnData, mode = "n_vs_s", threshold = None, lfc_cap = 50, device = 'cpu'):
    array_usr = (adata.uns['binned_data'] * adata.uns["mask_ecm"][:, :, None]).copy()
    genes = list(adata.uns['genes'])
    # pairs = adata.uns['pairs']
    genes = adata.uns['genes'].tolist()
    adata.uns['axon_markers'] = axon_markers
    adata.uns['dendrite_markers'] = dendrite_markers
    axon_markers = [gene for gene in axon_markers if gene in genes]
    dendrite_markers = [gene for gene in dendrite_markers if gene in genes]
    threshold = threshold if threshold is not None else 0.5
    a_map = adata.uns['axon_map']
    d_map = adata.uns['dendrite_map']
    mask = (a_map + d_map) == 1
    adata_ad = ad.AnnData(array_usr[mask])
    adata_ad.var_names = genes
    if mode == "a_vs_d":
        axonic_flags = a_map[mask]
        labels = np.where(axonic_flags == 1, 'axonic', 'dendritic')
        adata_ad.obs['a/d'] = pd.Categorical(labels)
        sc.pp.normalize_total(adata_ad, target_sum=1e4)
        sc.tl.rank_genes_groups(adata_ad, groupby='a/d', method='wilcoxon')
        de_df = sc.get.rank_genes_groups_df(adata_ad, group = "axonic")
        de_df = de_df[(~de_df.names.isin(axon_markers)) & (~de_df.names.isin(dendrite_markers))]
        de_df['-log10(FDR)'] = -np.log10(de_df['pvals_adj'].fillna(1.0)+ 1e-300)
        de_df['logfoldchanges'] = de_df['logfoldchanges'].clip(-lfc_cap, lfc_cap)
        adata.uns["a_vs_d"] = de_df
    elif mode == "n_vs_s":
        exp_cell = np.asarray(adata.X.sum(axis=0)).ravel()
        exp_neurites = np.asarray(adata_ad.X.sum(axis=0)).ravel()
        df_exp = pd.DataFrame({'Soma': exp_cell, "Neurites": exp_neurites})
        df_exp['ratio'] = (df_exp['Neurites'] + 1) / (df_exp['Soma'] + 1)   
        df_exp.sort_values(by="ratio", ascending=False, inplace=True)
        adata.uns["n_vs_s"] = df_exp
    elif mode == "pre_vs_post":
        presynapse_map = adata.uns['presynapse_map']
        postsynapse_map = adata.uns['postsynapse_map']
        mask = (presynapse_map + postsynapse_map) == 1
        adata_pp = ad.AnnData(array_usr[mask])
        adata_pp.var_names = genes
        presyn_flags = presynapse_map[mask]
        labels = np.where(presyn_flags == 1, 'presynaptic', 'postsynaptic')
        adata_pp.obs['pre/post'] = pd.Categorical(labels)
        sc.pp.normalize_total(adata_pp, target_sum=1e4)
        sc.tl.rank_genes_groups(adata_pp, groupby='pre/post', method='wilcoxon')
        de_df = sc.get.rank_genes_groups_df(adata_pp, group = "presynaptic")
        de_df = de_df[(~de_df.names.isin(axon_markers)) & (~de_df.names.isin(dendrite_markers))]
        de_df['-log10(FDR)'] = -np.log10(de_df['pvals_adj'].fillna(1.0)+ 1e-300)
        de_df['logfoldchanges'] = de_df['logfoldchanges'].clip(-lfc_cap, lfc_cap)
        adata.uns["pre_vs_post"] = de_df
    else:
        raise ValueError("Mode should be either 'a_vs_d' or 'n_vs_s'")
    return adata

def enrichment(adata: ad.AnnData, mode = "a_vs_d", lfc_threshold = 3, pval_threshold = 1e-5, threshold = None, organism = 10090, device = 'cpu'):
    de_df = adata.uns[mode]
    genes = list(adata.uns['genes'])
    de_df['significance'] = de_df.apply(categorize_gene, axis=1, args=(lfc_threshold, pval_threshold))
    de_df['point_size'] = de_df['significance'].map(size_mapping)
    df_up = get_panther(','.join(de_df[de_df.significance == "Upregulated"].names.values), ','.join(genes)).sort_values(by="fdr")
    df_down = get_panther(','.join(de_df[de_df.significance == "Downregulated"].names.values), ','.join(genes)).sort_values(by="fdr")
    if mode == "a_vs_d":
        adata.uns['axon_enrichment'] = df_up
        adata.uns['dendrite_enrichment'] = df_down
    elif mode == "pre_vs_post":
        adata.uns['presynapse_enrichment'] = df_up
        adata.uns['postsynapse_enrichment'] = df_down
    else:
        raise ValueError("Mode should be either 'a_vs_d' or 'pre_vs_post'")
    return adata