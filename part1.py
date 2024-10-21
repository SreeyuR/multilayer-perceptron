import datasets

# start with a totally balanced split in 2 dimensions
N = 100
M = 100
dims = 2

# Part 1 A
# TODO: run the following lines.
#       Comment on the resulting plots
dataset = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims)
dataset.save_dataset_plot()

# Part 1 C
# TODO: go to random_control.py and set the random seed to 0
