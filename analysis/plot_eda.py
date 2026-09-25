"""Plot true-link similarity and deceptive exact-key collision distributions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
positive = json.loads((root / "deep_eda_results.json").read_text(encoding="utf-8"))
collision = json.loads((root / "collision_eda_results.json").read_text(encoding="utf-8"))
name_bins = ["<30", "30-49", "50-69", "70-89", "90+"]
address_bins = ["<30", "30-49", "50-69", "70-89", "90+", "missing"]
grid = np.array([[positive["positive_pairs"]["similarity_grid"].get(f"name:{n}/address:{a}", 0)
                  for a in address_bins] for n in name_bins])
grid_pct = grid / grid.sum() * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
ax = axes[0]
im = ax.imshow(grid_pct, cmap="Blues", aspect="auto", vmin=0, vmax=max(20, grid_pct.max()))
for i in range(len(name_bins)):
    for j in range(len(address_bins)):
        value = grid_pct[i, j]
        if value >= 0.5:
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if value > 10 else "#152b44")
ax.set_xticks(range(len(address_bins)), address_bins, rotation=45, ha="right")
ax.set_yticks(range(len(name_bins)), name_bins)
ax.set_xlabel("Address token-sort similarity")
ax.set_ylabel("Name token-sort similarity")
ax.set_title("True links: name vs address\n103,685 labeled pairs")

score_bins = ["<30", "30-49", "50-69", "70-89", "90+", "missing"]
for ax, route, other in ((axes[1], "exact_name", "address"), (axes[2], "exact_address", "name")):
    data = collision[route]["other_field_similarity_bins"]
    bins = score_bins if route == "exact_name" else score_bins[:-1]
    x = np.arange(len(bins))
    true = np.array([data.get(f"true:{b}", 0) for b in bins], dtype=float)
    false = np.array([data.get(f"false:{b}", 0) for b in bins], dtype=float)
    ax.bar(x - 0.2, true / max(true.sum(), 1) * 100, 0.4, label="True links", color="#1c7c72")
    ax.bar(x + 0.2, false / max(false.sum(), 1) * 100, 0.4, label="Wrong candidates", color="#d17d30")
    ax.set_xticks(x, bins, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share within each class (%)")
    ax.set_xlabel(f"Other-field {other} similarity")
    ax.set_title(f"Candidates with exact {route.split('_')[1]}\n{collision[route]['counts']['all_pairs']:,} pairs from 5,000 S1")
    ax.legend(fontsize=8)

fig.suptitle("Amazon ML Challenge 2026: why both fields matter", fontsize=14, weight="bold")
out = root / "eda_patterns.png"
fig.savefig(out, dpi=160, facecolor="white")
print(out)
