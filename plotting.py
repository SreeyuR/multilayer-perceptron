# Plotting Functions

import numpy as np
import matplotlib.pyplot as plt
import os
import mlp


def plot_arr_per_epoch(title, arr, save=False, ylabel="loss"):
    plt.clf()
    plt.scatter(np.arange(len(arr)), arr)
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel(ylabel)
    if save:
        os.makedirs(f"figs/{ylabel}", exist_ok=True)
        plt.savefig(f"figs/{ylabel}/{title}.png")
    else:
        plt.show()


def plot_gradients(grads, ax):
    ax.scatter(np.arange(len(grads)), grads, alpha=0.01, s=3)
    ax.set_xlabel("epochs")
    ax.set_ylabel("gradient values")
    return ax


def generate_plot(x, y, std, error_type, plot_type, ax=None):
    #
    # Generate a plot with error bars
    #

    if isinstance(x[0], tuple):
        x = [str(i) for i in x]
    if isinstance(x[0], str):
        x_labels = x
        x = range(len(x))
    else:
        x_labels = None

    bar_width = 0.35  # Width of the bars

    # Adjust colors and labels for train and test
    if error_type == "train":
        color = "C0"
        adjusted_x = [xi - bar_width / 2 for xi in x]
    else:  # error_type == "test"
        color = "C1"
        adjusted_x = [xi + bar_width / 2 for xi in x]

    if plot_type in ["line", "line_xlog"]:
        ax.plot(x, y, label=f"{error_type} error")
        ax.fill_between(x, y - std, y + std, alpha=0.2, facecolor=color)
    elif plot_type == "bar":
        ax.bar(
            adjusted_x,
            y + 0.001,  # to avoid zero height bars
            width=bar_width,
            yerr=std,
            capsize=5,
            color=color,
            label=f"{error_type} error",
            error_kw={"ecolor": "lightgray", "elinewidth": 2},
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    if plot_type == "line_xlog":
        ax.set_xscale("log")
    if x_labels:
        ax.set_xticks(x)  # Set the positions of the x-ticks
        ax.set_xticklabels(x_labels)  # Set the text labels for the x-ticks


def plot_error(train_error, test_error, x, axis_label, plot_type="line", ax=None):
    #
    # Plot the test and train error over 3 runs of the same experiment for specified
    # of training samples
    # Input:
    #   train_error: a 3xX matrix of training errors per run
    #   test_error: a 3xX matrix of test errors per run
    #   x: a 1xX vector that depicts the category that is varying (ie LR, epochs,
    #      sample size)
    #   axis_label: the xaxis label
    #   plot_type: either "line", "line_xlog", "bar"
    #   ax: default to None, otherwise plot on the axis passed in
    #
    # Across the 3 runs of the same experiment, calculate the mean error.
    # The output should be a 1xX vector
    #
    train_mean_errors = np.mean(train_error, axis=0)
    test_mean_errors = np.mean(test_error, axis=0)

    # Across the 3 runs of the same experiment, calculate the standard deviation.
    # The output should be a 1x5 vector
    train_std_devs = np.std(train_error, axis=0)
    test_std_devs = np.std(test_error, axis=0)

    # plt.clf() # clear previous plot

    # Axes labels
    ax.set_xlabel(f"{axis_label}")
    ax.set_ylabel(f"Error")

    # Create a plot with shading to indicate error bars
    # Plot the training error
    generate_plot(x, train_mean_errors, train_std_devs, "train", plot_type, ax)

    # Then plot the testing error on the same figure
    generate_plot(x, test_mean_errors, test_std_devs, "test", plot_type, ax)

    ax.legend()
    return ax


def get_errors(model, X_train, y_train, X_test, y_test, lr, changing_var, change_type):
    num_runs = 3
    train_err = np.zeros((num_runs, len(changing_var)))
    test_err = np.zeros((num_runs, len(changing_var)))

    # defaults if unchanged
    epochs = 1000
    activation_func = "relu"
    loss_func = "negative_log_likelihood"
    batch_size = 16

    for i in range(num_runs):
        tr_errs = []
        test_errs = []
        for v in changing_var:
            if change_type == "epochs":
                model.train(
                    X_train, y_train, v, lr, batch_size, activation_func, loss_func
                )
            elif change_type == "learning_rate":
                model.train(
                    X_train, y_train, epochs, v, batch_size, activation_func, loss_func
                )
            elif change_type == "num_sample":
                N, M = v
                assert (y_train == 1).sum() >= N and (
                    y_train == 0
                ).sum() >= M, "Not enough samples in the dataset"
                X_train_new = np.concatenate(
                    [X_train[y_train == 1][:N], X_train[y_train == 0][:M]], axis=0
                )
                y_train_new = np.concatenate(
                    [y_train[y_train == 1][:N], y_train[y_train == 0][:M]], axis=0
                )
                # shuffle the data
                idx = np.random.permutation(len(X_train_new))
                model.train(
                    X_train_new[idx],
                    y_train_new[idx],
                    epochs,
                    lr,
                    batch_size,
                    activation_func,
                    loss_func,
                )
            elif change_type == "layer_width":
                model = mlp.MLP([X_train.shape[1], v, len(np.unique(y_test))])
                model.train(
                    X_train, y_train, epochs, lr, batch_size, activation_func, loss_func
                )
            elif change_type == "activation_func":
                model.train(X_train, y_train, epochs, lr, batch_size, v, loss_func)
            elif change_type == "loss_func":
                model.train(
                    X_train, y_train, epochs, lr, batch_size, activation_func, v
                )

            model.predict(X_train, y_train)
            tr_errs.append(model.prediction_error)
            model.predict(X_test, y_test)
            test_errs.append(model.prediction_error)
        train_err[i, :] = tr_errs
        test_err[i, :] = test_errs
    return train_err, test_err, model
