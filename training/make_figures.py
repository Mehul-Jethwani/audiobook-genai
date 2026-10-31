import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def loss_curve(state_path, out):
    state = json.load(open(state_path, encoding="utf-8"))
    train = [(h["step"], h["loss"]) for h in state["log_history"] if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in state["log_history"] if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(*zip(*train), label="training loss", linewidth=1.6)
    if evals:
        ax.plot(*zip(*evals), label="validation loss", marker="o", linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("Fine-tuning loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def system_comparison(results, out):
    systems = results["systems"]
    names = {"majority baseline": "Majority baseline", "base model, zero-shot": "Base model (zero-shot)",
             "fine-tuned (ours)": "Fine-tuned (ours)", "LLM teacher": "LLM teacher"}
    colors = {"majority baseline": "#9aa0a6", "base model, zero-shot": "#e8a33d",
              "fine-tuned (ours)": "#2a6fdb", "LLM teacher": "#3aa57a"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})

    fields = ["emotion", "intensity", "pace"]
    group = ["majority baseline", "base model, zero-shot", "fine-tuned (ours)"]
    width = 0.26
    for i, s in enumerate(group):
        vals = [systems[s][f]["macro_f1"] for f in fields]
        bars = left.bar([x + (i - 1) * width for x in range(len(fields))], vals, width, label=names[s], color=colors[s])
        for b, v in zip(bars, vals):
            left.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    left.set_xticks(range(len(fields)))
    left.set_xticklabels([f.capitalize() for f in fields])
    left.set_ylim(0, 1)
    left.set_ylabel("macro-F1")
    left.set_title("Emotion, intensity, pace (macro-F1)")
    left.legend(fontsize=8)

    fields2 = ["speaker", "gender"]
    group2 = ["majority baseline", "base model, zero-shot", "fine-tuned (ours)", "LLM teacher"]
    width = 0.2
    for i, s in enumerate(group2):
        vals = [systems[s][f]["acc"] for f in fields2]
        bars = right.bar([x + (i - 1.5) * width for x in range(len(fields2))], vals, width, label=names[s], color=colors[s])
        for b, v in zip(bars, vals):
            right.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    right.set_xticks(range(len(fields2)))
    right.set_xticklabels(["Speaker", "Gender"])
    right.set_ylim(0, 125)
    right.set_yticks(range(0, 101, 20))
    right.set_ylabel("accuracy % (dialogue units)")
    right.set_title("Speaker and gender (human-verified)")
    right.legend(fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def confusion_heatmap(results, out):
    conf = results["emotion_confusion"]
    rows, cols, matrix = conf["rows"], conf["columns"], conf["matrix"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    top = max(max(r) for r in matrix)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > top / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Emotion confusion matrix (fine-tuned model)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="eval_results.json")
    ap.add_argument("--state", help="trainer_state.json from the run1 checkpoint folder")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = json.load(open(args.eval, encoding="utf-8"))
    system_comparison(results, f"{args.out}/system_comparison.png")
    confusion_heatmap(results, f"{args.out}/emotion_confusion.png")
    made = ["system_comparison.png", "emotion_confusion.png"]
    if args.state:
        loss_curve(args.state, f"{args.out}/loss_curve.png")
        made.append("loss_curve.png")
    print("wrote", ", ".join(made), "to", args.out)


if __name__ == "__main__":
    main()
