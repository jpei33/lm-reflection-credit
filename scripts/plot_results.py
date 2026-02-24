import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# Data (from your logs)
# -------------------------
rows = [
    # dataset, method, strict_acc, tokens_per_ex, latency_per_ex
    ("GSM8K-200", "baseline",     0.085,  389.8,  7.232),
    ("GSM8K-200", "retry",        0.165,  726.8, 13.143),
    ("GSM8K-200", "reflect_full", 0.115, 1111.3, 16.722),
    ("GSM8K-200", "reflect_plan", 0.185,  893.2, 17.274),
    ("GSM8K-200", "reflect_tail", 0.175,  947.8, 17.295),

    ("MATH-200",  "baseline",     0.150,  632.3, 14.244),
    ("MATH-200",  "retry",        0.255, 1223.4, 27.508),
    ("MATH-200",  "reflect_full", 0.255, 1881.1, 28.021),
    ("MATH-200",  "reflect_tail", 0.265, 1509.3, 28.079),
    ("MATH-200",  "reflect_plan", 0.240, 1432.9, 27.948),
]

df = pd.DataFrame(rows, columns=["dataset", "method", "strict_acc", "tokens_per_ex", "latency_per_ex"])

# Keep a consistent method order for plotting
method_order = ["baseline", "retry", "reflect_plan", "reflect_tail", "reflect_full"]

# Pretty labels (optional)
pretty = {
    "baseline": "Baseline",
    "retry": "Retry",
    "reflect_plan": "Reflect-Plan",
    "reflect_tail": "Reflect-Tail",
    "reflect_full": "Reflect-Full",
}

def plot_grouped_bar(metric_col, ylabel, title, filename):
    plt.figure()
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].set_index("method").reindex(method_order)
        x = range(len(method_order))
        plt.bar([i + (0.0 if dataset == df["dataset"].unique()[0] else 0.4) for i in x],
                sub[metric_col].values,
                width=0.4,
                label=dataset)

    plt.xticks([i + 0.2 for i in range(len(method_order))], [pretty[m] for m in method_order], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=200)
    plt.close()

def plot_tradeoff_scatter(title, filename):
    plt.figure()
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        plt.scatter(sub["tokens_per_ex"], sub["strict_acc"], label=dataset)

        # label each point with method
        for _, r in sub.iterrows():
            plt.annotate(pretty[r["method"]],
                         (r["tokens_per_ex"], r["strict_acc"]),
                         textcoords="offset points",
                         xytext=(6, 4),
                         fontsize=8)

    plt.xlabel("Tokens / example")
    plt.ylabel("Strict accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename), dpi=200)
    plt.close()

# -------------------------
# Make plots
# -------------------------
plot_grouped_bar(
    metric_col="strict_acc",
    ylabel="Strict accuracy",
    title="Strict Accuracy by Method (GSM8K-200 vs MATH-200)",
    filename="acc_strict_by_method.png",
)

plot_grouped_bar(
    metric_col="tokens_per_ex",
    ylabel="Tokens / example",
    title="Token Cost by Method (GSM8K-200 vs MATH-200)",
    filename="tokens_by_method.png",
)

plot_grouped_bar(
    metric_col="latency_per_ex",
    ylabel="Latency / example (s)",
    title="Latency by Method (GSM8K-200 vs MATH-200)",
    filename="latency_by_method.png",
)

plot_tradeoff_scatter(
    title="Accuracy–Compute Tradeoff (Strict Acc vs Tokens)",
    filename="tradeoff_acc_vs_tokens.png",
)

print("Wrote figures to:", OUTDIR)
print(df)
