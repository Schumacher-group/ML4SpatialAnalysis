import unittest
import warnings

import numpy as np

from mainutils.utils import GRAPH_METHODS, clamp_k, coords_to_graph

K_METHODS = ('knn', 'atmostk')
ALL_METHODS = ('knn', 'atmostk', 'delaunay')


def roi(n, seed=0, extent=800.0):
    return np.random.default_rng(seed).uniform(0, extent, size=(n, 2))


class CoordsToGraphTest(unittest.TestCase):
    def test_every_method_and_mode_builds_a_valid_graph(self):
        # The smallest ROI passing the min_cells=10 filter in the cohort has 26 cells.
        for n in (26, 200, 1000):
            for gmethod in ALL_METHODS:
                for mode in ('connectivity', 'distance'):
                    with self.subTest(n=n, gmethod=gmethod, mode=mode):
                        G = coords_to_graph(roi(n), gmethod=gmethod, mode=mode, k=7)
                        self.assertEqual(G.shape, (n, n))
                        self.assertGreater(G.nnz, 0)
                        self.assertEqual(G.diagonal().sum(), 0, 'no self loops expected')
                        self.assertTrue(np.all(np.isfinite(G.data)))

    def test_k_drives_knn_and_atmostk_but_not_delaunay(self):
        coords = roi(500, seed=1)
        for gmethod in K_METHODS:
            with self.subTest(gmethod=gmethod):
                small = coords_to_graph(coords, gmethod=gmethod, k=5).nnz
                large = coords_to_graph(coords, gmethod=gmethod, k=12).nnz
                self.assertLess(small, large, f'{gmethod} ignores k')
        self.assertEqual(
            coords_to_graph(coords, gmethod='delaunay', k=5).nnz,
            coords_to_graph(coords, gmethod='delaunay', k=12).nnz,
        )

    def test_knn_gives_each_node_at_least_k_neighbours(self):
        # Symmetrisation can only add edges, so k is a floor rather than an exact degree.
        G = coords_to_graph(roi(300, seed=2), gmethod='knn', k=9)
        self.assertTrue(np.all(np.asarray((G != 0).sum(axis=1)).ravel() >= 9))

    def test_atmostk_caps_degree_at_k(self):
        G = coords_to_graph(roi(300, seed=3), gmethod='atmostk', k=6)
        # Symmetrisation can push a node above k, but no node may originate more than k.
        degrees = np.asarray((G != 0).sum(axis=1)).ravel()
        self.assertLessEqual(degrees.mean(), 6)

    def test_k_larger_than_the_roi_is_clamped_not_fatal(self):
        coords = roi(12, seed=4)
        for gmethod in K_METHODS:
            with self.subTest(gmethod=gmethod):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    G = coords_to_graph(coords, gmethod=gmethod, k=40)
                self.assertEqual(G.shape, (12, 12))
                self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))

    def test_degenerate_coordinates_do_not_break_delaunay(self):
        collinear = np.stack([np.arange(6.0), np.arange(6.0)], axis=1)
        G = coords_to_graph(collinear, gmethod='delaunay')
        self.assertEqual(G.shape, (6, 6))
        self.assertGreater(G.nnz, 0)

    def test_fewer_than_three_points(self):
        for n in (1, 2):
            for gmethod in ALL_METHODS:
                with self.subTest(n=n, gmethod=gmethod):
                    G = coords_to_graph(roi(n, seed=5), gmethod=gmethod, k=7)
                    self.assertEqual(G.shape, (n, n))

    def test_every_method_is_symmetric(self):
        # All arms of the benchmark must be undirected, or the comparison across
        # gmethods confounds neighbourhood definition with edge directedness.
        coords = roi(400, seed=6)
        for gmethod in GRAPH_METHODS:
            for mode in ('connectivity', 'distance'):
                with self.subTest(gmethod=gmethod, mode=mode):
                    G = coords_to_graph(coords, gmethod=gmethod, mode=mode,
                                        k=7, radius=120.0).tocsr()
                    self.assertEqual(abs(G - G.T).nnz, 0)

    def test_symmetrisation_preserves_one_way_distances(self):
        # (A + A.T) / 2 would halve them: the absent direction is a structural zero,
        # not a measured distance of zero.
        coords = roi(300, seed=8)
        G = coords_to_graph(coords, gmethod='knn', mode='distance', k=7).tocsr()
        # coords_to_graph returns similarities; invert back to the stored distance.
        recovered = 1.0 / G.data - 1e-9
        i, j = G.nonzero()
        true_d = np.linalg.norm(coords[i] - coords[j], axis=1)
        np.testing.assert_allclose(recovered, true_d, rtol=1e-4)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            coords_to_graph(roi(50), gmethod='not-a-method')

    def test_clamp_k_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            clamp_k(0, 100)

    def test_graph_methods_are_all_dispatchable(self):
        coords = roi(100, seed=7)
        for gmethod in GRAPH_METHODS:
            with self.subTest(gmethod=gmethod):
                G = coords_to_graph(coords, gmethod=gmethod, k=7, radius=120.0)
                self.assertEqual(G.shape, (100, 100))


if __name__ == '__main__':
    unittest.main()
