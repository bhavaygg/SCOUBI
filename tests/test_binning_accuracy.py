import tempfile
import types
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scoubi


class _TranscriptLoader:
    def __init__(self, df):
        self.df = df

    def load(self):
        return self.df


def _adata_with_transcripts(df, obs_names, usr_label="UNASSIGNED"):
    adata = ad.AnnData(X=sp.csr_matrix((len(obs_names), 0)), obs=pd.DataFrame(index=obs_names))
    adata.uns["usr_label"] = usr_label
    loader = _TranscriptLoader(df.copy())

    def _get_transcripts(self):
        return loader

    adata.get_transcripts = types.MethodType(_get_transcripts, adata)
    return adata


class BinningAccuracyTests(unittest.TestCase):
    def test_bin_data_exact_counts_masks_and_coordinate_alignment(self):
        df = pd.DataFrame(
            {
                "x_location": [0.0, 5.0, 10.0, 0.0, 5.0],
                "y_location": [0.0, 0.0, 10.0, 5.0, 5.0],
                "cell_id": ["cell_a", "cell_a", "cell_b", "UNASSIGNED", "cell_b"],
                "gene": ["gene_b", "gene_a", "gene_a", "gene_b", "gene_b"],
            }
        )
        adata = _adata_with_transcripts(df, ["cell_a", "cell_b"])

        scoubi.pp.bin_data(adata, binsize=5)

        self.assertEqual(adata.uns["binned_data_shape"], [2, 2, 2])
        self.assertEqual(adata.uns["genes"], ["gene_a", "gene_b"])

        expected = np.zeros((2, 2, 2), dtype=np.int16)
        expected[0, 0, 1] = 1
        expected[0, 1, 1] = 1
        expected[1, 0, 0] = 1
        expected[1, 1, 0] = 1
        expected[1, 1, 1] = 1

        actual_flat = adata.uns["binned_data"].toarray()
        np.testing.assert_array_equal(actual_flat, expected.reshape(4, 2))

        actual_3d = scoubi.pp.get_binned_data_3d(adata).todense()
        np.testing.assert_array_equal(actual_3d, expected)

        flattened_from_3d = scoubi.pp.get_binned_data_2d(adata).toarray()
        np.testing.assert_array_equal(flattened_from_3d, actual_flat)

        expected_ecm = np.array([[0, 1], [0, 0]], dtype=np.int8)
        expected_cell = np.array([[1, 0], [1, 1]], dtype=np.int8)
        np.testing.assert_array_equal(adata.uns["mask_ecm"], expected_ecm)
        np.testing.assert_array_equal(adata.uns["mask_cell"], expected_cell)

        expected_bins = np.array([[0.5, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(adata.obsm["bin"], expected_bins)

    def test_binned_data_3d_survives_h5ad_round_trip(self):
        df = pd.DataFrame(
            {
                "x_location": [0.0, 5.0, 10.0, 0.0, 5.0],
                "y_location": [0.0, 0.0, 10.0, 5.0, 5.0],
                "cell_id": ["cell_a", "cell_a", "cell_b", "UNASSIGNED", "cell_b"],
                "gene": ["gene_b", "gene_a", "gene_a", "gene_b", "gene_b"],
            }
        )
        adata = _adata_with_transcripts(df, ["cell_a", "cell_b"])
        scoubi.pp.bin_data(adata, binsize=5)

        expected_3d = scoubi.pp.get_binned_data_3d(adata)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "binned.h5ad"
            adata.write_h5ad(path)
            loaded = ad.read_h5ad(path)

        actual_3d = scoubi.pp.get_binned_data_3d(loaded)
        self.assertEqual(actual_3d.shape, expected_3d.shape)
        np.testing.assert_array_equal(actual_3d.coords, expected_3d.coords)
        np.testing.assert_array_equal(actual_3d.data, expected_3d.data)
        np.testing.assert_array_equal(
            loaded.uns["binned_data"].toarray(),
            adata.uns["binned_data"].toarray(),
        )

    def test_bin_counts_do_not_overflow_int16(self):
        n_transcripts = 40000
        df = pd.DataFrame(
            {
                "x_location": np.concatenate([
                    np.full(n_transcripts, 0.0),
                    np.array([10.0]),
                ]),
                "y_location": np.concatenate([
                    np.full(n_transcripts, 0.0),
                    np.array([10.0]),
                ]),
                "cell_id": ["cell_a"] * (n_transcripts + 1),
                "gene": ["gene_a"] * (n_transcripts + 1),
            }
        )
        adata = _adata_with_transcripts(df, ["cell_a"])

        scoubi.pp.bin_data(adata, binsize=5)

        self.assertEqual(adata.uns["binned_data"][0, 0], n_transcripts)
        self.assertEqual(scoubi.pp.get_binned_data_3d(adata)[0, 0, 0], n_transcripts)


if __name__ == "__main__":
    unittest.main()
