# -*- coding: utf-8 -*-

from .context import sample

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        sample.hmm()

    def load_dataset(self):
        """Loads the input file and divides it into 'train' and 'test' sets."""
        


if __name__ == '__main__':
    unittest.main()