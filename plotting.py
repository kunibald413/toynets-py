import re
import matplotlib.pyplot as plt

RUN_1 = """
sdg_vanilla:
loss: 2.3700 step: 0 lr: 0.0050
loss: 2.3335 step: 1 lr: 0.0050
loss: 2.3885 step: 2 lr: 0.0050
loss: 2.4033 step: 3 lr: 0.0050
loss: 2.3597 step: 4 lr: 0.0050
loss: 2.3553 step: 5 lr: 0.0050
loss: 2.3463 step: 6 lr: 0.0050
loss: 2.2710 step: 7 lr: 0.0050
loss: 2.3686 step: 8 lr: 0.0050
loss: 2.3561 step: 9 lr: 0.0050
loss: 2.3762 step: 10 lr: 0.0050
loss: 1.7938 step: 1000 lr: 0.0050
loss: 1.5748 step: 2000 lr: 0.0050
loss: 1.5569 step: 3000 lr: 0.0050
loss: 1.6167 step: 4000 lr: 0.0050
loss: 1.4477 step: 5000 lr: 0.0050
loss: 1.1484 step: 6000 lr: 0.0050
loss: 1.4083 step: 7000 lr: 0.0050
loss: 1.2032 step: 8000 lr: 0.0050
loss: 1.1976 step: 9000 lr: 0.0050
loss: 1.2455 step: 10000 lr: 0.0050
loss: 0.9983 step: 11000 lr: 0.0050
loss: 1.0367 step: 12000 lr: 0.0050
loss: 0.9487 step: 13000 lr: 0.0050
loss: 1.2556 step: 14000 lr: 0.0050
loss: 1.1349 step: 15000 lr: 0.0050
loss: 1.1032 step: 16000 lr: 0.0050
loss: 1.0523 step: 17000 lr: 0.0050
loss: 1.0641 step: 18000 lr: 0.0050
loss: 0.8839 step: 19000 lr: 0.0050
loss: 0.7396 step: 20000 lr: 0.0050
loss: 0.9484 step: 21000 lr: 0.0050
loss: 0.8361 step: 22000 lr: 0.0050
loss: 0.8192 step: 23000 lr: 0.0050
loss: 0.8660 step: 24000 lr: 0.0050
test accuracy: 50.60 perc
"""

RUN_2 = """
with momentum:
loss: 2.3700 step: 0 lr: 0.0050
loss: 2.3335 step: 1 lr: 0.0050
loss: 2.3863 step: 2 lr: 0.0050
loss: 2.3939 step: 3 lr: 0.0050
loss: 2.3474 step: 4 lr: 0.0050
loss: 2.3357 step: 5 lr: 0.0050
loss: 2.3077 step: 6 lr: 0.0050
loss: 2.2448 step: 7 lr: 0.0050
loss: 2.3205 step: 8 lr: 0.0050
loss: 2.2824 step: 9 lr: 0.0050
loss: 2.2831 step: 10 lr: 0.0050
loss: 1.5089 step: 1000 lr: 0.0050
loss: 1.3445 step: 2000 lr: 0.0050
loss: 1.1230 step: 3000 lr: 0.0050
loss: 1.4535 step: 4000 lr: 0.0050
loss: 0.9712 step: 5000 lr: 0.0050
loss: 0.6387 step: 6000 lr: 0.0050
loss: 1.0189 step: 7000 lr: 0.0050
loss: 0.7362 step: 8000 lr: 0.0050
loss: 0.7723 step: 9000 lr: 0.0050
loss: 0.7318 step: 10000 lr: 0.0050
loss: 0.5153 step: 11000 lr: 0.0050
loss: 0.5327 step: 12000 lr: 0.0050
loss: 0.5673 step: 13000 lr: 0.0050
loss: 0.7034 step: 14000 lr: 0.0050
loss: 0.6498 step: 15000 lr: 0.0050
loss: 0.6083 step: 16000 lr: 0.0050
loss: 0.6452 step: 17000 lr: 0.0050
loss: 0.7427 step: 18000 lr: 0.0050
loss: 0.3707 step: 19000 lr: 0.0050
loss: 0.2710 step: 20000 lr: 0.0050
loss: 0.3375 step: 21000 lr: 0.0050
loss: 0.3077 step: 22000 lr: 0.0050
loss: 0.3554 step: 23000 lr: 0.0050
loss: 0.3736 step: 24000 lr: 0.0050
test accuracy: 49.00 perc
"""

LOSS_RE = re.compile(r"loss:\s*([-\d.eE+]+)\s*step:\s*(\d+)\s*lr:\s*([-\d.eE+]+)")
ACC_RE = re.compile(r"test accuracy:\s*([-\d.eE+]+)\s*perc")


def parse_run(text):
    """Return (name, steps, losses, lrs, test_acc)."""
    name = None
    for line in text.strip().splitlines():
        line = line.strip()
        if line.endswith(":") and not line.startswith("loss"):
            name = line[:-1]
            break

    steps, losses, lrs = [], [], []
    for m in LOSS_RE.finditer(text):
        loss, step, lr = float(m.group(1)), int(m.group(2)), float(m.group(3))
        steps.append(step)
        losses.append(loss)
        lrs.append(lr)

    acc = None
    m = ACC_RE.search(text)
    if m:
        acc = float(m.group(1))

    return name, steps, losses, lrs, acc


def plot_runs(runs, title="Training loss", log_x=False):
    """runs: list of raw text blobs. Plots side by side."""
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), sharey=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, text in zip(axes, runs):
        name, steps, losses, lrs, acc = parse_run(text)
        ax.plot(steps, losses, marker="o", markersize=3, linewidth=1)
        label = name if name else "run"
        if acc is not None:
            label += f"  (test acc {acc:.2f}%)"
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale("log")

    axes[0].set_ylabel("loss")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    fig = plot_runs([RUN_1, RUN_2], title="SGD vanilla vs momentum")
    plt.show()


if __name__ == "__main__":
    main()