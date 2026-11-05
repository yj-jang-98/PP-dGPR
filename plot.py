"""Plot the hyperparameter traces and PoE agreement curves in Figs. 2 and 3."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_hyperparameter(sarcos_history, diab_history, config, use_latex=True):
    """Plot lengthscale and signal-variance traces; save ``img/hyp.pdf``."""
    if use_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,  # Use LaTeX to render text
                "font.family": "serif",  # Match LaTeX serif font (Computer Modern)
                "pgf.rcfonts": False,  # Don't override LaTeX font config
                "font.size": 24,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,  # Pure matplotlib mathtext, no LaTeX dependency
                "font.family": "DejaVu Sans",
                "font.size": 24,
            }
        )

    M = config["M"]
    T = config["Thyp"]

    colors_sarcos = plt.cm.get_cmap("tab10", M)
    colors_diab = plt.cm.get_cmap("tab10", M)

    sarcos_history = np.array(sarcos_history)
    diab_history = np.array(diab_history)

    iterations_sarcos = range(T)
    iterations_diab = range(T)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    # SARCOS: θ_l
    ax = axs[0]
    ax.set_title("SARCOS: $\\theta_{i,l}(t)$")
    for agent in range(M):
        ax.plot(
            iterations_sarcos,
            sarcos_history[:, agent, 0],
            color=colors_sarcos(agent),
            linewidth=2,
            alpha=0.8,
        )
    ax.set_xlabel(r"$t$")
    ax.grid(True)
    ax.set_xlim(0, 30)
    ax.set_ylim(5, 15)
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([5, 10, 15])

    # SARCOS: θ_s
    ax = axs[1]
    ax.set_title("SARCOS: $\\theta_{i,s}(t)$")
    for agent in range(M):
        ax.plot(
            iterations_sarcos,
            sarcos_history[:, agent, 1],
            color=colors_sarcos(agent),
            linewidth=2,
            alpha=0.8,
        )
    ax.set_xlabel(r"$t$")
    ax.set_xlim(0, 30)
    ax.set_ylim(5, 15)
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([5, 10, 15])
    ax.grid(True)

    # Diabetes: θ_l
    ax = axs[2]
    ax.set_title("Diabetes: $\\theta_{i,l}(t)$")
    for agent in range(M):
        ax.plot(
            iterations_diab,
            diab_history[:, agent, 0],
            color=colors_diab(agent),
            linewidth=2,
            alpha=0.8,
        )
    ax.set_xlabel(r"$t$")
    ax.set_xlim(0, 30)
    ax.set_ylim(5, 15)
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([5, 10, 15])
    ax.grid(True)

    # Diabetes: θ_s
    ax = axs[3]
    ax.set_title("Diabetes: $\\theta_{i,s}(t)$")
    for agent in range(M):
        ax.plot(
            iterations_diab,
            diab_history[:, agent, 1],
            color=colors_diab(agent),
            linewidth=2,
            alpha=0.8,
        )
    ax.set_xlabel(r"$t$")
    ax.set_xlim(0, 30)
    ax.set_ylim(5, 20)
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([5, 10, 15, 20])
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("img/hyp.pdf")
    plt.show()


def plot_accuracy(
    sarcos_rmse_means,
    sarcos_rmse_vars,
    diab_rmse_means,
    diab_rmse_vars,
    config,
    Lz_values,
    use_latex=True,
):
    """Plot mean and variance RMSE against PoE; save ``img/accuracy.pdf``."""
    num_Lz = len(Lz_values)

    if use_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,  # Use LaTeX to render text
                "font.family": "serif",  # Match LaTeX serif font (Computer Modern)
                "pgf.rcfonts": False,  # Don't override LaTeX font config
                "font.size": 24,
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,  # Pure matplotlib mathtext, no LaTeX dependency
                "font.family": "DejaVu Sans",
                "font.size": 24,
            }
        )

    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]
    zoom_range = range(config["T"] - 4, config["T"] + 1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    rmse_hat_label = (
        r"$\mathsf{RMSE}_{\smash{\hat{f}}}(T)$"
        if use_latex
        else r"$\mathsf{RMSE}_{\hat{f}}(T)$"
    )
    rmse_var_label = r"$\mathsf{RMSE}_{V}(T)$"

    # SARCOS: mean
    ax = axs[0]
    ax.set_title(rf"SARCOS: {rmse_hat_label}", pad=4)
    for lz_idx in range(num_Lz):
        label = rf"$\mathsf{{L}}_z = 1/10^{{{int(np.log10(Lz_values[lz_idx]))}}}$"
        ax.plot(
            range(config["T"] + 1),
            sarcos_rmse_means[lz_idx].detach().cpu().numpy(),
            marker=markers[lz_idx],
            markersize=8,
            label=label,
            linestyle=linestyles[lz_idx],
            alpha=1,
            markevery=5,
            linewidth=2.5,
            color=colors[lz_idx],
        )

    ax.set_xlim(0, config["T"])

    ax.set_xticks([0, config["T"] / 2, config["T"]])

    ax.set_xlabel(r"$T$")
    ax.grid(True)

    axins = inset_axes(
        ax,
        width="40%",
        height="40%",
        bbox_to_anchor=(-0.05, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper right",
        borderpad=0,
    )

    for lz_idx in [1, 2]:
        axins.plot(
            zoom_range,
            sarcos_rmse_means[lz_idx, -5:].detach().cpu().numpy(),
            linestyle=linestyles[lz_idx],
            linewidth=2.5,
            alpha=1,
            color=colors[lz_idx],
        )

    axins.set_xticks([config["T"] - 4, config["T"]])
    axins.grid(True)
    axins.tick_params(labelsize=16)
    axins.set_xlim(config["T"] - 4, config["T"])

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # SARCOS: variance
    ax = axs[1]
    ax.set_title(rf"SARCOS: {rmse_var_label}", pad=6)
    for lz_idx in range(num_Lz):
        ax.plot(
            range(config["T"] + 1),
            sarcos_rmse_vars[lz_idx].detach().cpu().numpy(),
            marker=markers[lz_idx],
            linestyle=linestyles[lz_idx],
            alpha=1,
            markevery=5,
            markersize=8,
            linewidth=2.5,
            color=colors[lz_idx],
        )
    ax.set_xlabel(r"$T$")
    ax.grid(True)
    ax.set_xlim(0, config["T"])

    ax.set_xticks([0, config["T"] / 2, config["T"]])

    axins = inset_axes(
        ax,
        width="40%",
        height="40%",
        bbox_to_anchor=(-0.05, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper right",
        borderpad=0,
    )

    for lz_idx in [1, 2]:
        axins.plot(
            zoom_range,
            sarcos_rmse_vars[lz_idx, -5:].detach().cpu().numpy(),
            linestyle=linestyles[lz_idx],
            linewidth=2.5,
            alpha=1,
            color=colors[lz_idx],
        )

    axins.set_xticks([config["T"] - 4, config["T"]])
    axins.grid(True)
    axins.tick_params(labelsize=16)
    axins.set_xlim(config["T"] - 4, config["T"])
    axins.ticklabel_format(style="plain", axis="y")  # No scientific notation
    axins.get_yaxis().get_offset_text().set_visible(False)  # Hide offset

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Diabetes: mean
    ax = axs[2]
    ax.set_title(rf"Diabetes: {rmse_hat_label}", pad=5)
    for lz_idx in range(num_Lz):
        ax.plot(
            range(config["T"] + 1),
            diab_rmse_means[lz_idx].detach().cpu().numpy(),
            marker=markers[lz_idx],
            linestyle=linestyles[lz_idx],
            alpha=1,
            markevery=5,
            markersize=8,
            linewidth=2,
            color=colors[lz_idx],
        )

    ax.set_xlim(0, config["T"])

    ax.set_xticks([0, config["T"] / 2, config["T"]])

    ax.set_xlabel(r"$T$")
    ax.grid(True)

    axins = inset_axes(
        ax,
        width="40%",
        height="40%",
        bbox_to_anchor=(-0.05, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper right",
        borderpad=0,
    )

    for lz_idx in [0, 1, 2]:
        axins.plot(
            zoom_range,
            diab_rmse_means[lz_idx, -5:].detach().cpu().numpy(),
            linestyle=linestyles[lz_idx],
            linewidth=2.5,
            alpha=1,
            color=colors[lz_idx],
        )

    axins.set_xticks([config["T"] - 4, config["T"]])
    axins.grid(True)
    axins.tick_params(labelsize=16)
    axins.set_xlim(config["T"] - 4, config["T"])

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # Diabetes: variance
    ax = axs[3]
    ax.set_title(rf"Diabetes: {rmse_var_label}", pad=5)
    for lz_idx in range(num_Lz):
        ax.plot(
            range(config["T"] + 1),
            diab_rmse_vars[lz_idx].detach().cpu().numpy(),
            marker=markers[lz_idx],
            linestyle=linestyles[lz_idx],
            alpha=1,
            markevery=5,
            markersize=8,
            linewidth=2,
            color=colors[lz_idx],
        )

    ax.set_xlim(0, config["T"])

    ax.set_xticks([0, config["T"] / 2, config["T"]])

    ax.set_xlabel(r"$T$")
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter3))

    ax.grid(True)

    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.11), frameon=False)
    plt.tight_layout()
    plt.savefig("img/accuracy.pdf", bbox_inches="tight")
    plt.show()


def custom_formatter3(x, pos):
    if x == 0:
        return "0"
    else:
        return f"{x:.3f}"
