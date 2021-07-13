import unittest
from pathlib import Path
import numpy as np
from .skeleton_resample import resample
from .skeleton_parser import get_graph, calculate_root


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
            (0, 0, 0, 0, 0, 0, -1),
            (1, 0, 0, 0, 10, 0, 0),
            (2, 0, 0, 0, 20, 0, 1),
            (3, 0, 0, 0, 30, 0, 2),
            (4, 0, 10, 0, 10, 0, 1),
            (5, 0, 20, 0, 10, 0, 4),
        ]
        self.G = get_graph(nodes)

    def test_resample(self):
        calculate_root(self.G)
        N = resample(self.G, 15)
        # print([(node, value) for node, value in N.nodes.items()])
        # print(N.edges)
