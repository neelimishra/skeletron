import unittest
from pathlib import Path
import numpy as np
from .skeleton_parser import *


class TestSkeletonParser(unittest.TestCase):
    """Tests for skeleton_parser file."""

    def setUp(self):
        """
        0-1-2-3
          |
          4
          |
          5
        """
        nodes = [
            (0, 0, 00, 0, 0, 0, -1),
            (1, 0, 10, 0, 0, 0, 0),
            (2, 0, 20, 0, 0, 0, 1),
            (3, 0, 30, 0, 0, 0, 2),
            (4, 0, 10, 10, 0, 0, 1),
            (5, 0, 10, 20, 0, 0, 4),
        ]
        self.G = get_graph(nodes)

    def test_get_root(self):
        calculate_root(self.G)
        self.assertEqual(self.G.graph["root_node"], 0)

    def test_dfs(self):
        dfs_iter = iter(dfs(self.G)[::-1])
        self.assertEqual(next(dfs_iter), 3)
        self.assertEqual(next(dfs_iter), 2)
        self.assertEqual(next(dfs_iter), 5)
        self.assertEqual(next(dfs_iter), 4)
        self.assertEqual(next(dfs_iter), 1)
        self.assertEqual(next(dfs_iter), 0)

    def test_get_strahler(self):
        calculate_strahlers(self.G)
        self.assertEqual(self.G.nodes[0]["strahler"], 2)
        self.assertEqual(self.G.nodes[1]["strahler"], 2)
        self.assertEqual(self.G.nodes[2]["strahler"], 1)
        self.assertEqual(self.G.nodes[4]["strahler"], 1)
        self.assertEqual(self.G.nodes[3]["strahler"], 1)
        self.assertEqual(self.G.nodes[5]["strahler"], 1)

    def test_new_root(self):
        N = new_root(self.G, 5)
        dfs_iter = iter(dfs(N))
        self.assertEqual(next(dfs_iter), 5)
        self.assertEqual(next(dfs_iter), 4)
        self.assertEqual(next(dfs_iter), 1)
        self.assertEqual(next(dfs_iter), 2)
        self.assertEqual(next(dfs_iter), 3)
        self.assertEqual(next(dfs_iter), 0)

    def test_split_node(self):
        S, R, a, b = split_tree(self.G, split_node=2)
        dfs_iter_s = iter(dfs(S))
        self.assertEqual(next(dfs_iter_s), 2)
        self.assertEqual(next(dfs_iter_s), 3)
        dfs_iter_r = iter(dfs(R))
        self.assertEqual(next(dfs_iter_r), 0)
        self.assertEqual(next(dfs_iter_r), 1)
        self.assertEqual(next(dfs_iter_r), 4)
        self.assertEqual(next(dfs_iter_r), 5)

    def test_merge_trees(self):
        b_nodes = [
            (0, 0, 0, 0, 10, 0, -1),
            (1, 0, 0, 0, 20, 0, 0),
            (2, 0, 0, 0, 30, 0, 1),
            (3, 0, 0, 0, 40, 0, 2),
        ]

        B = get_graph(b_nodes)
        C = merge_trees(self.G, B, (0, 0))
        print(
            [
                (node, C.nodes[node]["x"], C.nodes[node]["y"], C.nodes[node]["z"])
                for node, attrs in C.nodes.items()
            ]
        )
        print([edge for edge in C.edges])

    def test_swc_file(self):
        for f in Path("/home/pattonw/Downloads/allen cell types/CNG version").iterdir():
            raw_data = np.loadtxt(f)
            G = get_graph(raw_data)
            S, R, a, b = split_tree(G)
            break

    def test_re_index(self):
        nodes = [
            (0, 0, 00, 0, 0, 0, -1),
            (1, 0, 10, 0, 0, 0, 0),
            (2, 0, 20, 0, 0, 0, 1),
            (3, 0, 30, 0, 0, 0, 2),
            (5, 0, 10, 10, 0, 0, 1),
            (7, 0, 10, 20, 0, 0, 5),
        ]
        G = get_graph(nodes)
        F = re_index(G)
        nodes = set(F.nodes)
        self.assertEqual(nodes, set(range(6)))
