import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("entity_swap_full_sweep/outputs/swap_summary.csv")

relations = list(df["relation_name"].unique())
source_factors = sorted(df["source_factor"].unique())

n_rows = len(relations)
n_cols = len(source_factors)

fig_w = 5 * n_cols
fig_h = 4 * n_rows
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

for i, rel in enumerate(relations):
    for j, sf in enumerate(source_factors):
        ax = axes[i, j]
        sub = df[(df["relation_name"] == rel) & (df["source_factor"] == sf)].sort_values("donor_factor")

        x = np.arange(len(sub))
        width = 0.35

        bars1 = ax.bar(x - width/2, sub["top1_hit_rate"], width, label="Top-1")
        bars2 = ax.bar(x + width/2, sub["top5_hit_rate"], width, label="Top-5")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["donor_factor"].astype(str))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Donor factor")
        ax.set_ylabel("Hit rate")
        ax.set_title(f"{rel}\nsource_factor = {sf:g}")

        for bars in [bars1, bars2]:
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width()/2,
                    h + 0.02,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

# Put legend only once
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
plt.savefig("entity_swap_full_sweep/outputs/barchart.png")