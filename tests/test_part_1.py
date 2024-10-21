import os
import unittest
import pickle as pkl
from gradescope_utils.autograder_utils.decorators import weight, number

from random_control import seed as student_seed
import numpy as np


class TestMLP(unittest.TestCase):
    @weight(0)
    @number("part_1c.01")
    def test_random_seed(self):
        self.assertEqual(student_seed, 0, "Random seed is wrong")


if __name__ == "__main__":
    os.environ["TEST_SEED"] = str(0)
    unittest.main()
