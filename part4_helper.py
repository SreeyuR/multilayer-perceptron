import plotting
import datasets
import mlp
import matplotlib.pyplot as plt
import numpy as np
import os


# This helper function will be useful for Parts 4
def run_experiments(
    A_train_test,
    B_train_test,
    C_train_test,
    changing_var,
    changing_type,
    plot_type="log",
    save=False,
):
    title = f"{changing_type} - Train vs. Test Error"
    if changing_type == "num_sample":
        f, all_axs = plt.subplots(1, 2, figsize=(15, 5))
    else:
        f, all_axs = plt.subplots(1, 3, figsize=(15, 5))
    if changing_type == "epochs":
        g, grad_axs = plt.subplots(1, 3, figsize=(15, 5))
    else:
        grad_axs = [None, None, None]

    A_mlp = mlp.MLP([6, 4, 2])
    B_mlp = mlp.MLP([6, 4, 2])
    C_mlp = mlp.MLP([6, 4, 3])

    all_mlps = [A_mlp, B_mlp, C_mlp]
    names = ["Gaussian", "Circles", "Iris"]
    all_datasets = [A_train_test, B_train_test, C_train_test]

    # TODO: YOUR CODE HERE #
    # Fill in optimal learning rates after part4
    learning_rates = [1e-4, 1e-4, 1e-4]
    # END OF YOUR CODE #

    for i, (model, ax, name, data, lr, g_ax) in enumerate(
        zip(all_mlps, all_axs, names, all_datasets, learning_rates, grad_axs)
    ):
        if changing_type == "num_sample":
            if i == 2:
                break
        X_train, X_test, y_train, y_test = data

        train_err, test_err, model = plotting.get_errors(
            model, X_train, y_train, X_test, y_test, lr, changing_var, changing_type
        )

        ax = plotting.plot_error(
            train_err, test_err, changing_var, changing_type, plot_type, ax
        )
        ax.set_title(name)

        if changing_type == "epochs":
            grads = [gr["dW"] for gr in model.all_gradients]
            grad_norms = [np.linalg.norm(gr) for gr in grads]
            g_ax = plotting.plot_gradients(grad_norms, g_ax)
            g_ax.set_title(name)
            g_ax.set_xlabel("steps")
            g_ax.set_yscale("log")

    f.suptitle(title)
    if save:
        os.makedirs("figs/experiments/", exist_ok=True)
        f.savefig(f"figs/experiments/{changing_type}")
    else:
        plt.show()

    if changing_type == "epochs":
        g.suptitle("gradients")
        if save:
            g.savefig(f"figs/experiments/{changing_type}_gradients")
