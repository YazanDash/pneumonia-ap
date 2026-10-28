from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# --- Configuration Paths ---
final_root = Path("../../../chest_xray_final")
classes = ["normal", "pneumonia"]
splits = ["train", "val", "test"]

split_counts = {cls: [] for cls in classes}

print("==================================================")
print("     EDA: Label Distribution & Majority Baseline     ")
print("==================================================")

for split_name in splits:
  split_path = final_root / split_name
  print(f"\n[{split_name.upper()} SET]")

  if not split_path.exists():
    print(f"  Warning: Directory not found at {split_path.resolve()}")
    for cls in classes:
      split_counts[cls].append(0)
    continue

  counts = {}
  total = 0

  # Count files directly from disk using pathlib
  for cls in classes:
    cls_dir = split_path / cls
    if cls_dir.exists():
      count = len([f for f in cls_dir.iterdir() if f.is_file()])
    else:
      count = 0
    counts[cls] = count
    total += count

  for cls in classes:
    split_counts[cls].append(counts[cls])

  if total > 0:
    for cls, count in counts.items():
      pct = (count / total) * 100
      print(f"  - {cls}: {count} ({pct:.2f}%)")

    majority_class = max(counts, key=counts.get)
    baseline = (counts[majority_class] / total) * 100
    print(
        f"Majority Baseline: {baseline:.2f}% (guessing '{majority_class}')"
    )
  else:
    print("Warning: No images found in this split.")

print("==================================================")

# ==========================================
# MATPLOTLIB VISUALIZATION (Bar Chart)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(splits))
width = 0.35

rects1 = ax.bar(
    x - width / 2, split_counts["normal"], width, label="Normal", color="#4C72B0"
)
rects2 = ax.bar(
    x + width / 2,
    split_counts["pneumonia"],
    width,
    label="Pneumonia",
    color="#C44E52",
)

ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title(
    "Class Distribution Across Splits (Stratified Check)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels([s.upper() for s in splits], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.7)


# Add exact counts on top of bars for readability
def autolabel(rects):
  for rect in rects:
    height = rect.get_height()
    ax.annotate(
        f"{height}",
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
    )


autolabel(rects1)
autolabel(rects2)

print("==================================================")
print("  EDA: Mean Pixel Intensity Outliers (|z| > 3)    ")
print("==================================================")

file_paths = []
mean_intensities = []

# Scan all images across dataset splits
for img_path in final_root.glob("**/*.*"):
  if img_path.is_file() and img_path.suffix.lower() in [
      ".jpg",
      ".jpeg",
      ".png",
  ]:
    try:
      # Open image in grayscale (L mode) to compute pixel intensity cleanly
      with Image.open(img_path).convert("L") as img:
        arr = np.array(img, dtype=np.float32)
        mean_val = np.mean(arr)
        mean_intensities.append(mean_val)
        file_paths.append(img_path)
    except Exception:
      pass

mean_intensities = np.array(mean_intensities)

if len(mean_intensities) == 0:
  print("Error: No images found to analyze.")
  exit()

# --- Calculate Z-Scores for Pixel Intensity ---
mu = np.mean(mean_intensities)
sigma = np.std(mean_intensities)

if sigma == 0:
  sigma = 1e-5

z_scores = (mean_intensities - mu) / sigma

# Identify outliers where |z| > 3 (the extreme 0.3% tails)
outlier_mask = np.abs(z_scores) > 3
outlier_intensities = mean_intensities[outlier_mask]
normal_intensities = mean_intensities[~outlier_mask]

print(f"Total Images Analyzed: {len(mean_intensities)}")
print(f"Mean Intensity (μ): {mu:.2f} | Std Dev (σ): {sigma:.2f}")
print(
    f"Outliers Detected (|z| > 3): {len(outlier_intensities)} images found."
)

if len(outlier_intensities) > 0:
  print("Flagged Outlier Samples:")
  for path, z in zip(np.array(file_paths)[outlier_mask], z_scores[outlier_mask]):
    print(f"  - {path.relative_to(final_root)} (Z-score: {z:.2f})")

# ==========================================
# STYLIZED HISTOGRAM VISUALIZATION
# ==========================================
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
fig, ax = plt.subplots(figsize=(10, 6))


# Generate uniform bins covering the full range of ALL data
bins = np.linspace(mean_intensities.min(), mean_intensities.max(), 45)

# Plot normal range histogram
n, ax_bins, patches = ax.hist(
    normal_intensities,
    bins=bins,
    color="#4C72B0",
    edgecolor="white",
    alpha=0.85,
    label="Normal Range (|z| <= 3)",
)

# Highlight outliers in the extreme tails if any exist
if len(outlier_intensities) > 0:
  ax.hist(
      outlier_intensities,
      bins=bins,
      color="#C44E52",
      edgecolor="white",
      alpha=0.9,
      label="Outliers (|z| > 3 / Extreme 0.3%)",
  )

# Explicitly widen the X-axis so the tails have breathing room
ax.set_xlim(mean_intensities.min() - 5, mean_intensities.max() + 5)


# Vertical threshold lines for Mean and ±3 Standard Deviations
ax.axvline(
    mu, color="#2ca02c", linestyle="-", linewidth=2, label=f"Mean: {mu:.1f}"
)
ax.axvline(
    mu + 3 * sigma,
    color="#d62728",
    linestyle="--",
    linewidth=1.5,
    label="+3σ Threshold",
)
ax.axvline(
    mu - 3 * sigma,
    color="#d62728",
    linestyle="--",
    linewidth=1.5,
    label="-3σ Threshold",
)

# Styling & Formatting
ax.set_title(
    "Chest X-Ray Mean Pixel Intensity Distribution & Outlier Detection",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Mean Pixel Intensity (0–255 Grayscale)", fontsize=12, labelpad=10)
ax.set_ylabel("Frequency (Image Count)", fontsize=12, labelpad=10)
ax.legend(
    frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="upper right"
)

# Clean up modern minimalist spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()
plt.show()
