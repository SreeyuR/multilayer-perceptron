import os
import unittest
import pickle as pkl
from gradescope_utils.autograder_utils.decorators import weight, number
from mlp import Layer
import numpy as np


class TestMLP(unittest.TestCase):
    seed = int(os.environ["TEST_SEED"])
    input_size = 100

    @weight(5)
    @number("part_2i.01")
    def test_sigmoid_activation(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs = 2 * np.random.random(size=input_size) - 1
        outputs = dl.sigmoid(inputs)

        with open(
            f"test_files_seed={seed}/part_2/sigmoid_activation_out.pkl", "rb"
        ) as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid activation function",
        )

    @weight(5)
    @number("part_2i.02")
    def test_sigmoid_derivative(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        inputs_Z = 2 * np.random.random(size=input_size) - 1
        outputs = dl.sigmoid_derivative(inputs_Z)

        with open(
            f"test_files_seed={seed}/part_2/sigmoid_derivative_out.pkl", "rb"
        ) as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for sigmoid derivative function",
        )

    @weight(5)
    @number("part_2i.03")
    def test_softmax(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        dl = Layer(1)
        num_out = 2
        inputs = 2 * np.random.random(size=(input_size, num_out)) - 1
        outputs = dl.softmax(inputs)
        with open(f"test_files_seed={seed}/part_2/softmax_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for softmax function",
        )

    @weight(5)
    @number("part_2i.04")
    def test_forward_sigmoid(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 6
        inputs = 2 * np.random.random((input_size, 3)) - 1
        weights = 2 * np.random.random((3, num_neurons)) - 1
        bias = 2 * np.random.random((1, num_neurons)) - 1
        dl = Layer(num_neurons)
        outputs = dl.forward(inputs, weights, bias, activation="sigmoid")

        with open(f"test_files_seed={seed}/part_2/forward_sigmoid_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for the forward function with sigmoid activation",
        )

    @weight(5)
    @number("part_2i.05")
    def test_forward_softmax(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 6
        inputs = 2 * np.random.random((input_size, 3)) - 1
        weights = 2 * np.random.random((3, num_neurons)) - 1
        bias = 2 * np.random.random((1, num_neurons)) - 1
        dl = Layer(num_neurons)
        outputs = dl.forward(inputs, weights, bias, activation="softmax")

        with open(f"test_files_seed={seed}/part_2/forward_softmax_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for the forward function with softmax activation",
        )

    @weight(0)
    @number("part_2i.06")
    def test_forward_relu(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 6
        inputs = 2 * np.random.random((input_size, 3)) - 1
        weights = 2 * np.random.random((3, num_neurons)) - 1
        bias = 2 * np.random.random((1, num_neurons)) - 1
        dl = Layer(num_neurons)
        outputs = dl.forward(inputs, weights, bias, activation="relu")

        with open(f"test_files_seed={seed}/part_2/forward_relu_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for the forward function with ReLU activation",
        )

    @weight(0)
    @number("part_2i.07")
    def test_forward_tanh(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 6
        inputs = 2 * np.random.random((input_size, 3)) - 1
        weights = 2 * np.random.random((3, num_neurons)) - 1
        bias = 2 * np.random.random((1, num_neurons)) - 1
        dl = Layer(num_neurons)
        outputs = dl.forward(inputs, weights, bias, activation="tanH")

        with open(f"test_files_seed={seed}/part_2/forward_tanh_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        self.assertTrue(
            np.allclose(outputs, corr_outputs, atol=1e-4),
            "Incorrect output for the forward function with tanH activation",
        )

    @weight(10)
    @number("part_2i.08")
    def test_backward_sigmoid(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 10
        num_out = 2
        dA_curr = 2 * np.random.random((input_size, num_out)) - 1
        W_curr = 2 * np.random.random((num_neurons, num_out)) - 1
        Z_curr = 2 * np.random.random((input_size, num_out)) - 1
        A_prev = 2 * np.random.random((input_size, num_neurons)) - 1
        dl = Layer(num_neurons)

        outputs = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation="sigmoid")

        with open(f"test_files_seed={seed}/part_2/backward_sigmoid_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        output_keys = ["dA", "dW", "db"]
        is_correct = {}
        for idx, key in enumerate(output_keys):
            is_correct[key] = np.allclose(outputs[idx], corr_outputs[idx])

        incorrect_outputs = [key for key, value in is_correct.items() if not value]

        self.assertTrue(
            all(is_correct.values()),
            "Incorrect output for the backward function with sigmoid activation. "
            "Incorrect outputs: " + ", ".join(incorrect_outputs),
        )

    @weight(10)
    @number("part_2i.09")
    def test_backward_softmax(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 10
        num_out = 2
        dA_curr = 2 * np.random.random((input_size, num_out)) - 1
        W_curr = 2 * np.random.random((num_neurons, num_out)) - 1
        Z_curr = 2 * np.random.random((input_size, num_out)) - 1
        A_prev = 2 * np.random.random((input_size, num_neurons)) - 1
        dl = Layer(num_neurons)

        outputs = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation="softmax")

        with open(f"test_files_seed={seed}/part_2/backward_softmax_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        output_keys = ["dA", "dW", "db"]
        is_correct = {}
        for idx, key in enumerate(output_keys):
            is_correct[key] = np.allclose(outputs[idx], corr_outputs[idx])

        incorrect_outputs = [key for key, value in is_correct.items() if not value]

        self.assertTrue(
            all(is_correct.values()),
            "Incorrect output for the backward function with softmax activation. "
            "Incorrect outputs: " + ", ".join(incorrect_outputs),
        )

    @weight(0)
    @number("part_2i.10")
    def test_backward_relu(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 10
        num_out = 2
        dA_curr = 2 * np.random.random((input_size, num_out)) - 1
        W_curr = 2 * np.random.random((num_neurons, num_out)) - 1
        Z_curr = 2 * np.random.random((input_size, num_out)) - 1
        A_prev = 2 * np.random.random((input_size, num_neurons)) - 1
        dl = Layer(num_neurons)

        outputs = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation="relu")

        with open(f"test_files_seed={seed}/part_2/backward_relu_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        output_keys = ["dA", "dW", "db"]
        is_correct = {}
        for idx, key in enumerate(output_keys):
            is_correct[key] = np.allclose(outputs[idx], corr_outputs[idx])

        incorrect_outputs = [key for key, value in is_correct.items() if not value]

        self.assertTrue(
            all(is_correct.values()),
            "Incorrect output for the backward function with ReLU activation. "
            "Incorrect outputs: " + ", ".join(incorrect_outputs),
        )

    @weight(0)
    @number("part_2i.11")
    def test_backward_tanh(self):
        seed = TestMLP.seed
        input_size = TestMLP.input_size
        np.random.seed(seed)

        num_neurons = 10
        num_out = 2
        dA_curr = 2 * np.random.random((input_size, num_out)) - 1
        W_curr = 2 * np.random.random((num_neurons, num_out)) - 1
        Z_curr = 2 * np.random.random((input_size, num_out)) - 1
        A_prev = 2 * np.random.random((input_size, num_neurons)) - 1
        dl = Layer(num_neurons)

        outputs = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation="tanH")

        with open(f"test_files_seed={seed}/part_2/backward_tanh_out.pkl", "rb") as f:
            corr_outputs = pkl.load(f)

        output_keys = ["dA", "dW", "db"]
        is_correct = {}
        for idx, key in enumerate(output_keys):
            is_correct[key] = np.allclose(outputs[idx], corr_outputs[idx])

        incorrect_outputs = [key for key, value in is_correct.items() if not value]

        self.assertTrue(
            all(is_correct.values()),
            "Incorrect output for the backward function with tanH activation. "
            "Incorrect outputs: " + ", ".join(incorrect_outputs),
        )


if __name__ == "__main__":
    os.environ["TEST_SEED"] = str(0)
    unittest.main()
