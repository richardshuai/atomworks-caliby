import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
# df = pd.read_csv("pipeline_test_af3_2024-09-08.timings.csv")
df = pd.read_csv("pipeline_test_rf2aa_2024-09-08.timings.csv")

# Convert example_id to string (if not already)
df["example_id"] = df["example_id"].astype(str)
n_examples = df["example_id"].nunique()

# Group by transform name and calculate median
grouped = df.groupby("name")["processing_time"].median().sort_values(ascending=True)

# Separate transforms into two groups based on median processing time
slow_transforms = grouped[grouped >= 0.5].index
fast_transforms = grouped[grouped < 0.5].index


# Function to create and save boxenplot
def create_boxenplot(data, transforms, title, filename, log_scale=False, plot_function=sns.boxenplot):
    ax = plot_function(
        y="name",
        x="processing_time",
        data=data[data["name"].isin(transforms)],
        order=grouped[grouped.index.isin(transforms)].index,
    )

    if log_scale:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)

    ax.set_title(title)
    ax.set_xlabel("Processing Time (s)")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# Create and save plots for slow and fast transforms
create_boxenplot(
    df,
    slow_transforms,
    f"Transforms with Median Processing Time >= 0.5s (n={n_examples})",
    "slow_transforms.png",
    plot_function=sns.violinplot,
)
create_boxenplot(
    df,
    fast_transforms,
    f"Transforms with Median Processing Time < 0.5s (n={n_examples}) (Log Scale)",
    "fast_transforms.png",
    log_scale=True,
    plot_function=sns.boxenplot,
)

# Calculate total processing time per sample
total_times = df.groupby("example_id")["processing_time"].sum()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[4, 1], sharex=True)

# Create distribution plot on the main axis
sns.histplot(total_times, kde=True, edgecolor="gray", ax=ax1, color="skyblue")
ax1.set_xlim((0, None))

# Calculate statistics
mean_time = total_times.mean()
median_time = total_times.median()
percentile_25 = total_times.quantile(0.25)
percentile_75 = total_times.quantile(0.75)

# Add vertical lines for statistics with labels on the main axis
ax1.axvline(mean_time, color="red", linestyle="--", label=f"Mean: {mean_time:.2f}s")
ax1.axvline(median_time, color="gray", linestyle="--", label=f"Median: {median_time:.2f}s")
ax1.axvline(percentile_25, color="gray", linestyle="--", label=f"25th Percentile: {percentile_25:.2f}s")
ax1.axvline(percentile_75, color="gray", linestyle="--", label=f"75th Percentile: {percentile_75:.2f}s")

ax1.set_title(f"Distribution of Total Processing Times per Sample (n={n_examples})")
ax1.set_ylabel("Count")
ax1.legend()

# Create boxenplot on the second axis
sns.violinplot(x=total_times, ax=ax2, color="skyblue")
ax2.set_xlabel("Total Processing Time (s)")


# Adjust layout and save
plt.tight_layout()
# Remove padding
plt.subplots_adjust(hspace=0)
plt.savefig("total_processing_times_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Print summary statistics
print(f"Average processing time: {mean_time:.2f}s")
print(f"Median processing time: {median_time:.2f}s")
print(f"25th percentile processing time: {percentile_25:.2f}s")
print(f"75th percentile processing time: {percentile_75:.2f}s")
