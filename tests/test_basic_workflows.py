import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import scoubi


FIXTURE = Path(__file__).parent / "data" / "transcripts_small.parquet"


def _finite(values) -> bool:
    return np.isfinite(np.asarray(values, dtype=float)).all()


def _first_expressed_gene(array, coords):
    for x, y in coords:
        expressed = np.flatnonzero(array[x, y, :] > 0)
        if expressed.size:
            return expressed[0]
    return None


def _interface_fixture_inputs(adata):
    x_bins, y_bins, _ = adata.uns["binned_data_shape"]
    array = (
        adata.uns["binned_data"].toarray().reshape(adata.uns["binned_data_shape"])
        * adata.uns["mask_ecm"][:, :, None]
    )
    genes = list(adata.uns["genes"])

    for x in range(x_bins - 1):
        for y in range(y_bins - 1):
            axon_coords = [(x, y), (x, y + 1)]
            dendrite_coords = [(x + 1, y), (x + 1, y + 1)]
            ligand_idx = _first_expressed_gene(array, axon_coords)
            receptor_idx = _first_expressed_gene(array, dendrite_coords)
            if ligand_idx is None or receptor_idx is None:
                continue

            probabilities = np.zeros((x_bins * y_bins, 2), dtype=float)
            for row, col in axon_coords:
                probabilities[row * y_bins + col, 0] = 1.0
            for row, col in dendrite_coords:
                probabilities[row * y_bins + col, 1] = 1.0

            return probabilities, genes[ligand_idx], genes[receptor_idx]

    raise AssertionError("Fixture does not contain an expressed adjacent 2x2 block")


class BasicWorkflowTests(unittest.TestCase):
    def _binned_adata(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        transcript_path = Path(tmp.name) / FIXTURE.name
        shutil.copyfile(FIXTURE, transcript_path)

        adata = scoubi.io.load_data(str(transcript_path))
        adata.obs["cell_type"] = [f"cell-type-{i % 2}" for i in range(adata.n_obs)]
        adata.obs["region"] = [f"region-{i % 2}" for i in range(adata.n_obs)]
        return scoubi.pp.bin_data(adata, binsize=5)

    def test_public_import_aliases(self):
        self.assertIs(scoubi.pp, scoubi.preprocess)
        self.assertIs(scoubi.md, scoubi.model)
        self.assertIs(scoubi.tl, scoubi.tools)
        self.assertIs(scoubi.pl, scoubi.plotting)

    def test_load_and_bin_data_have_expected_finite_outputs(self):
        adata = self._binned_adata()

        self.assertGreater(adata.n_obs, 0)
        self.assertIn("spatial", adata.obsm)
        self.assertIn("binned_data", adata.uns)
        self.assertIn("mask_ecm", adata.uns)
        self.assertIn("mask_cell", adata.uns)
        self.assertIn("genes", adata.uns)

        self.assertTrue(_finite(adata.X.toarray()))
        self.assertTrue(_finite(adata.obsm["spatial"]))
        self.assertTrue(_finite(adata.uns["binned_data"].toarray()))
        self.assertTrue(_finite(adata.uns["mask_ecm"]))
        self.assertTrue(_finite(adata.uns["mask_cell"]))
        self.assertGreater(adata.uns["mask_ecm"].sum(), 0)
        self.assertGreater(adata.uns["mask_cell"].sum(), 0)

    def test_overview_enrichment_and_profiles_are_finite(self):
        adata = self._binned_adata()
        probabilities, ligand, receptor = _interface_fixture_inputs(adata)

        adata.uns["bin_probabilities"] = probabilities
        adata.uns["lr_pairs"] = [(ligand, receptor)]
        adata.uns["axon_markers"] = [ligand]
        adata.uns["dendrite_markers"] = [receptor]

        with contextlib.redirect_stdout(io.StringIO()):
            adata = scoubi.tl.overview(
                adata,
                threshold=0.5,
                zscore_threshold=-1e9,
                cw_edges_threshold=0,
                kernel_size=2,
                filter=(1, 1),
            )
        adata = scoubi.tl.axon_dendrite_enrichment(adata)
        adata = scoubi.tl.distance(adata)
        adata = scoubi.tl.expression_profile(adata, key="cell_type")
        adata = scoubi.tl.communication_profile(adata, key="cell_type")

        for key in ["axon_map", "dendrite_map", "interface_map"]:
            self.assertIn(key, adata.uns)
            self.assertTrue(_finite(adata.uns[key]))

        self.assertGreater(adata.uns["interface_map"].sum(), 0)
        self.assertTrue(_finite(adata.uns["cellwhisper_unfiltered"].select_dtypes("number")))
        self.assertTrue(_finite(adata.uns["ptest_a_vs_d"][["c1", "c2", "n1", "n2", "pvalue", "fdr"]]))
        self.assertTrue(_finite(adata.uns["interface_knn_dists"]))
        self.assertTrue(_finite(adata.uns["interface_cell_type_profile"]))
        self.assertTrue(_finite(adata.uns["communication_cell_type_profile"]))


if __name__ == "__main__":
    unittest.main()
