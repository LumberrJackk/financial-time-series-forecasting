import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

layers = [
    ("Input",           "9 x 9 x 1 spectrogram",    11.2, "#B4B2A9"),
    ("Conv2D (32 filters, 3x3)", "ReLU · output: 9x9x32", 9.8, "#AFA9EC"),
    ("MaxPooling2D (2x2)",  "output: 5x5x32",        8.4, "#5DCAA5"),
    ("Conv2D (64 filters, 3x3)", "ReLU · output: 5x5x64", 7.0, "#AFA9EC"),
    ("MaxPooling2D (2x2)",  "output: 3x3x64",        5.6, "#5DCAA5"),
    ("Flatten",         "576 units",                 4.2, "#B4B2A9"),
    ("Dense (64 units)",    "ReLU + Dropout 30%",    2.8, "#F0997B"),
    ("Output",          "1 unit — predicted price",  1.4, "#EF9F27"),
]

for title, subtitle, y, color in layers:
    rect = mpatches.FancyBboxPatch((1.5, y-0.5), 7, 1.0,
        boxstyle="round,pad=0.1", linewidth=1,
        edgecolor="#888", facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.text(5, y+0.12, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#1a1a1a')
    ax.text(5, y-0.18, subtitle, ha='center', va='center', fontsize=9, color='#333333')
    if y > 1.4:
        ax.annotate("", xy=(5, y-0.5), xytext=(5, y-0.85),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

ax.text(5, 12.0, "CNN Architecture — Stock Price Prediction",
    ha='center', va='center', fontsize=13, fontweight='bold', color='#1a1a1a')
ax.text(5, 0.5, "Total parameters: 55,809  |  Optimizer: Adam  |  Loss: MSE",
    ha='center', va='center', fontsize=9, color='#555555')

plt.tight_layout()
plt.savefig('plot_cnn_architecture.png', dpi=150, bbox_inches='tight')
print("CNN architecture diagram saved!")
