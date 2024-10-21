import datasets
import mlp
import plotting

# start with a totally balanced split in 3 dimensions
N = 100
M = 100
dims = 3
gaus_dataset_points = datasets.generate_nd_dataset(
    N, M, datasets.kGaussian, dims
).get_dataset()

X = gaus_dataset_points[:, :-1]
y = gaus_dataset_points[:, -1].astype(int)

model = mlp.MLP([3, 2])

model.train(X, y, epochs=10000)
title = f"losses_gaussian_{dims}d_mlp"
plotting.plot_arr_per_epoch(title, model.get_losses(), save=True, ylabel="loss")
title = f"accuracies_gaussian_{dims}d_mlp"
plotting.plot_arr_per_epoch(title, model.get_accuracy(), save=True, ylabel="accuracy")