import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'font.weight': 700,
    'axes.labelsize': 16,       # X and Y axis labels
    'axes.labelweight': 'bold', # X and Y axis label weight
    'axes.titlesize': 18,       # Plot titles
    'axes.titleweight': 'bold',
    'xtick.labelsize': 13,      # X-axis tick labels (genre names)
    'ytick.labelsize': 13,      # Y-axis tick labels
    'legend.fontsize': 13,      # Legend
})

# Genres, in the same order as the classification reports
genres = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]

# Per-class F1 scores from the classification reports
# CV = out-of-fold cross-validation scores
# Test = scores on the official FMA_small test set
scores = {
    "HGB": {
        "CV": [
            0.605, 0.502, 0.627, 0.715,
            0.508, 0.614, 0.330, 0.625,
        ],
        "Test": [
            0.642, 0.337, 0.199, 0.753,
            0.441, 0.584, 0.390, 0.575,
        ],
    },
    "SVM": {
        "CV": [
            0.573, 0.512, 0.631, 0.680,
            0.519, 0.607, 0.298, 0.640,
        ],
        "Test": [
            0.632, 0.367, 0.209, 0.727,
            0.378, 0.591, 0.335, 0.573,
        ],
    },
    "MLP": {
        "CV": [
            0.563, 0.464, 0.623, 0.689,
            0.535, 0.608, 0.285, 0.625,
        ],
        "Test": [
            0.645, 0.374, 0.262, 0.723,
            0.390, 0.596, 0.347, 0.618,
        ],
    },
}

x = np.arange(len(genres))
bar_width = 0.38

for model_name, model_scores in scores.items():
    cv_f1 = np.array(model_scores["CV"])
    test_f1 = np.array(model_scores["Test"])

    # Positive = better test score; negative = worse test score
    differences = test_f1 - cv_f1

    # Create two vertically stacked plots
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # --------------------------------------------------
    # Top plot: compare CV and test F1 for each genre
    # --------------------------------------------------
    ax_top.bar(
        x - bar_width / 2,
        cv_f1,
        bar_width,
        label="CV (OOF)",
    )
    ax_top.bar(
        x + bar_width / 2,
        test_f1,
        bar_width,
        label="Test",
    )

    ax_top.set_ylabel("F1 score")
    ax_top.set_title(f"{model_name}: per-class F1 — CV vs. test")
    ax_top.set_ylim(0, 1)
    ax_top.grid(axis="y", linestyle="--", alpha=0.35)
    ax_top.legend()

    # --------------------------------------------------
    # Bottom plot: difference between test and CV F1
    # --------------------------------------------------
    ax_bottom.bar(x, differences, width=0.65)
    ax_bottom.axhline(0, linewidth=1)

    ax_bottom.set_ylabel("Test − CV")
    ax_bottom.set_xlabel("Genre")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(
        genres,
        rotation=30,
        ha="right",
    )
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.35)

    # Prevent labels from being cut off
    fig.tight_layout()

plt.show()
