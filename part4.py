import datasets
import mlp
import numpy as np
import plotting

import part4_helper


#### Generate datasets ####
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

# # TODO: Learning rate experiments
# learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
# part4_helper.run_experiments(
#     A_train_test,
#     B_train_test,
#     C_train_test,
#     learning_rates,
#     "learning_rate",
#     plot_type="line_xlog",
#     save=True,
# )
#
# # TODO: Epochs experiments
# epochs = [1e1, 1e2, 1e3, 1e4]
# part4_helper.run_experiments(
#     A_train_test,
#     B_train_test,
#     C_train_test,
#     epochs,
#     "epochs",
#     plot_type="line_xlog",
#     save=True,
# )
#
# # TODO: Imabalanced datasets, varying sample sizes
# num_sample = 20000  # this is just a big pool of data points from which you can draw samples of different sizes
# A_dataset_points_large = datasets.generate_nd_dataset(
#     num_sample, num_sample, datasets.kGaussian, dims
# ).get_dataset()
# B_dataset_points_large = datasets.generate_nd_dataset(
#     num_sample, num_sample, datasets.kCircles
# ).get_dataset()
# C_dataset_points_large = datasets.generate_nd_dataset(
#     num_sample, num_sample, datasets.kIris
# ).get_dataset()  # not used for this part
# A_train_test_large = datasets.train_test_split_(A_dataset_points_large)
# B_train_test_large = datasets.train_test_split_(B_dataset_points_large)
# C_train_test_large = datasets.train_test_split_(C_dataset_points_large)
# NMs = [(10, 10), (30, 30), (100, 100), (1000, 10)]
# part4_helper.run_experiments(
#     A_train_test_large,
#     B_train_test_large,
#     C_train_test_large,
#     NMs,
#     "num_sample",
#     plot_type="line",
#     save=True,
# )
#
#
# # TODO: Layer width experiments
# num_layers = [1, 3, 5, 7, 15]
# part4_helper.run_experiments(
#     A_train_test,
#     B_train_test,
#     C_train_test,
#     num_layers,
#     "layer_width",
#     plot_type="line",
#     save=True,
# )
