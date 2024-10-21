import os
import unittest
import pickle as pkl
from gradescope_utils.autograder_utils.decorators import weight, number

from mlp import Layer
import numpy as np


class TestMLP(unittest.TestCase):
    seed = int(os.environ["TEST_SEED"])
    input_size = 100

    @weight(0)
    @number("part_5a.1")
    def test_tanh_activation(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs = 2 * np.random.random(size=input_size) - 1
        outputs = dl.tanH(inputs)

        with open(f"test_files_seed={seed}/part_5/tanh_activation_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid activation function",
        )

    @weight(2)
    @number("part_5a.2")
    def test_tanh_derivative(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs_Z = 2 * np.random.random(size=input_size) - 1
        outputs = dl.tanH_derivative(inputs_Z)

        with open(f"test_files_seed={seed}/part_5/tanh_derivative_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid derivative function",
        )

    @weight(2)
    @number("part_5a.3")
    def test_relu_activation(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs = 2 * np.random.random(size=input_size) - 1
        outputs = dl.relu(inputs)

        with open(f"test_files_seed={seed}/part_5/relu_activation_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid activation function",
        )

    @weight(2)
    @number("part_5a.4")
    def test_relu_derivative(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs_Z = 2 * np.random.random(size=input_size) - 1
        outputs = dl.relu_derivative(inputs_Z)

        with open(f"test_files_seed={seed}/part_5/relu_derivative_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid derivative function",
        )


if __name__ == "__main__":
    os.environ["TEST_SEED"] = str(0)
    unittest.main()
