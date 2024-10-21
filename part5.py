import datasets
import part4_helper

N = 100
M = 100

dims = 3
A_dataset_points = datasets.generate_nd_dataset(
    N, M, datasets.kGaussian, dims
).get_dataset()
B_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kCircles).get_dataset()
C_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kIris).get_dataset()
A_train_test = datasets.train_test_split_(A_dataset_points)
B_train_test = datasets.train_test_split_(B_dataset_points)
C_train_test = datasets.train_test_split_(C_dataset_points)

activation_funcs = ["sigmoid", "relu", "tanH"]
# TODO: see how various activation functions perform
part4_helper.run_experiments(
    A_train_test,
    B_train_test,
    C_train_test,
    activation_funcs,
    "activation_func",
    plot_type="bar",
    save=True,
)
