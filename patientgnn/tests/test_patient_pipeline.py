import unittest

import numpy as np

from patientgnn.graphs import physical_contact_graph
from patientgnn.metrics import bootstrap_auc_difference_ci
from patientgnn.normalization import MarkerNormalizer, stratified_cell_indices
from patientgnn.preprocessing import group_rois_by_patient, validate_roi_cache
from patientgnn.schema import CACHE_VERSION


def patient(patient_id, values):
    return {
        "patient_id": patient_id,
        "label": int(patient_id == "positive"),
        "rois": [{"expression": np.asarray(values, dtype=np.float32)}],
    }


class MarkerNormalizerTest(unittest.TestCase):
    def test_fit_is_patient_balanced_and_test_independent(self):
        train = [
            patient("negative", np.zeros((1000, 2))),
            patient("positive", np.full((10, 2), 10.0)),
        ]
        scaler = MarkerNormalizer(
            lower_quantile=0,
            upper_quantile=1,
            max_cells_per_patient=10,
            seed=3,
        ).fit(train)
        np.testing.assert_allclose(scaler.median_, [5, 5])
        state_before = scaler.state_dict()
        transformed = scaler.transform(np.full((2, 2), 1000.0))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(state_before, scaler.state_dict())

    def test_missing_values_are_imputed_and_optionally_flagged(self):
        train = [patient("negative", [[0.0, 1.0], [1.0, 2.0]])]
        scaler = MarkerNormalizer(include_missing_indicators=True).fit(train)
        transformed = scaler.transform(np.asarray([[np.nan, 1.5]], dtype=np.float32))
        self.assertEqual(transformed.shape, (1, 4))
        self.assertEqual(transformed[0, 2], 1.0)
        self.assertEqual(transformed[0, 3], 0.0)


class SamplingTest(unittest.TestCase):
    def test_stratified_cap_retains_rare_types(self):
        cell_types = np.asarray([0] * 100 + [1] * 3 + [2])
        selected = stratified_cell_indices(cell_types, max_cells=20, seed=9, min_per_type=2)
        self.assertEqual(len(selected), 20)
        self.assertEqual(set(cell_types[selected]), {0, 1, 2})
        np.testing.assert_array_equal(
            selected,
            stratified_cell_indices(cell_types, max_cells=20, seed=9, min_per_type=2),
        )


class GraphTest(unittest.TestCase):
    def test_contact_graph_is_symmetric_without_self_edges(self):
        positions = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        sizes = np.ones(4, dtype=np.float32)
        edge_index, edge_attr, radius = physical_contact_graph(
            positions, sizes, contact_scale=1.1, max_neighbors=4, num_rbf=4
        )
        edges = {tuple(edge) for edge in edge_index.T.tolist()}
        self.assertGreater(radius, 1.0)
        self.assertTrue(all(left != right for left, right in edges))
        self.assertTrue(all((right, left) in edges for left, right in edges))
        self.assertEqual(edge_attr.shape, (edge_index.shape[1], 7))

    def test_isolated_cell_gets_flagged_fallback(self):
        positions = np.asarray([[0, 0], [1, 0], [100, 100]], dtype=np.float32)
        sizes = np.ones(3, dtype=np.float32)
        edge_index, edge_attr, _ = physical_contact_graph(
            positions, sizes, contact_scale=1.1, num_rbf=3
        )
        far_edges = np.flatnonzero((edge_index == 2).any(axis=0))
        self.assertGreater(len(far_edges), 0)
        self.assertTrue(np.all(edge_attr[far_edges, -1] == 1.0))

    def test_partial_missing_cell_sizes_do_not_create_nan_edges(self):
        positions = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        sizes = np.asarray([1.0, np.nan, 1.0], dtype=np.float32)
        _, edge_attr, _ = physical_contact_graph(
            positions, sizes, contact_scale=2.1, max_neighbors=2, num_rbf=3
        )
        self.assertTrue(np.isfinite(edge_attr).all())

    def test_degree_cap_applies_after_symmetrization(self):
        positions = np.asarray(
            [[row, column] for row in range(4) for column in range(4)],
            dtype=np.float32,
        )
        sizes = np.full(len(positions), 10.0, dtype=np.float32)
        edge_index, _, _ = physical_contact_graph(
            positions, sizes, contact_scale=2.0, max_neighbors=3, num_rbf=3
        )
        degree = np.bincount(edge_index[0], minlength=len(positions))
        self.assertLessEqual(int(degree.max()), 3)


class CacheValidationTest(unittest.TestCase):
    def test_patient_grouping_and_shapes(self):
        roi = {
            "roi_id": "Leap001_1",
            "patient_id": "1",
            "label": 0,
            "stain_batch": "1",
            "expression": np.ones((3, 2), dtype=np.float32),
            "position": np.ones((3, 2), dtype=np.float32),
            "morphology": np.ones((3, 6), dtype=np.float32),
            "cell_type": np.asarray([0, 1, 1], dtype=np.int16),
        }
        cache = {
            "version": CACHE_VERSION,
            "markers": ["a", "b"],
            "cell_types": ["x", "y"],
            "rois": [roi],
            "manifest": {},
        }
        validate_roi_cache(cache)
        grouped = group_rois_by_patient(cache)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["patient_id"], "1")


class MetricsTest(unittest.TestCase):
    def test_paired_auc_bootstrap_uses_matched_patients(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = bootstrap_auc_difference_ci(
            labels, probability, probability, seed=7, replicates=100
        )
        self.assertEqual(result["auc_difference"], 0.0)
        np.testing.assert_allclose(result["ci"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
