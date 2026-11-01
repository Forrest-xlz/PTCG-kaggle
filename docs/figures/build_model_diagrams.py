"""Render the three diagrams used in the Kaggle solution write-up."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "figure.facecolor": "#f7f9fc",
        "savefig.facecolor": "#f7f9fc",
    }
)

INK = "#14213d"
MUTED = "#52627a"
LINE = "#71839e"
BLUE = ("#e8f2ff", "#4489d1")
GREEN = ("#e9f7ed", "#3ca367")
PURPLE = ("#f0eaff", "#8063d4")
ORANGE = ("#fff1df", "#d68b24")
RED = ("#ffeae8", "#d2695e")


def canvas(width=14, height=8):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    return fig, ax


def box(ax, x, y, w, h, title, body, palette=BLUE, size=11, title_size=12):
    face, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            linewidth=1.65, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x + 0.18, y + h - 0.25, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=INK)
    if body:
        ax.text(x + 0.18, y + h - 0.62, body, ha="left", va="top",
                fontsize=size, linespacing=1.35, color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=LINE):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=14,
                        lw=1.7, color=color, shrinkA=3, shrinkB=3),
    )


def title(ax, main, subtitle, top):
    ax.text(0.5, top, main, fontsize=21, fontweight="bold", va="top", color=INK)
    ax.text(0.5, top - 0.44, subtitle, fontsize=11.5, va="top", color=MUTED)


def save(fig, name):
    svg = OUT / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.22)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def state_tokens():
    fig, ax = canvas(14, 8.1)
    title(ax, "1  Build the state memory", "26 current-state tokens + one token for the previous three actions", 7.85)

    box(ax, 0.5, 5.65, 3.8, 1.13, "Bench · tokens 0–15",
        "8 own slots + 8 opponent slots", BLUE)
    box(ax, 4.55, 5.65, 3.8, 1.13, "Active · tokens 16–17",
        "Own and opponent Pokémon", BLUE)
    box(ax, 8.6, 5.65, 4.9, 1.13, "Player summaries · tokens 18–19",
        "Hand, deck, prizes, and other counts", BLUE)

    box(ax, 0.5, 4.2, 3.8, 1.13, "Discard · tokens 20–21",
        "Known cards in each discard pile", BLUE, size=10.0)
    box(ax, 4.55, 4.2, 3.8, 1.13, "Zones · tokens 22–24",
        "Own hand, deck, and Stadium", BLUE, size=10.0)
    box(ax, 8.6, 4.2, 4.9, 1.13, "Global · token 25",
        "Turn and current selection context", BLUE)

    box(ax, 2.35, 2.5, 9.3, 1.13, "State-token construction",
        "Card embedding bags + static features; summary MLPs; region MLPs → width 512", PURPLE)
    for x in (2.4, 6.45, 11.05):
        arrow(ax, x, 4.18, x, 3.65)

    box(ax, 0.5, 0.55, 8.0, 1.35, "History · token 26",
        "Previous 3 actions → sum each action's option vectors → action MLP\n"
        "Concatenate in order → sequence MLP → one 512-dimensional token", GREEN, size=10.6)
    box(ax, 9.1, 0.55, 4.4, 1.35, "Encoder input · [27, 512]",
        "26 state tokens + history token\nMask empty Bench and absent history", ORANGE, size=10.6)
    arrow(ax, 8.5, 1.22, 9.08, 1.22)
    arrow(ax, 11.65, 2.48, 11.65, 1.92)
    save(fig, "model_state_tokens")


def action_tokens():
    fig, ax = canvas(14, 7.5)
    title(ax, "2  Build candidate action queries", "Encode options once, then combine them into the currently legal actions", 7.22)

    box(ax, 0.5, 5.0, 3.75, 1.37, "Current selection",
        "Selection type and context\nLegal option list; min/max choices", BLUE, size=10.8)
    box(ax, 4.65, 5.0, 4.1, 1.37, "Each offered option",
        "Card, attack, target, area, relation\nPosition and current game features", GREEN, size=10.8)
    box(ax, 9.15, 5.0, 4.35, 1.37, "Option token · [512]",
        "11 categorical + 5 numeric channels\nStatic/dynamic projections → token MLP", PURPLE, size=10.2)
    arrow(ax, 4.27, 5.68, 4.62, 5.68)
    arrow(ax, 8.77, 5.68, 9.12, 5.68)

    box(ax, 0.5, 2.75, 4.0, 1.25, "Enumerate legal actions",
        "One or several options per action\nKeep at most 64 candidates", BLUE, size=10.8)
    box(ax, 5.0, 2.75, 4.05, 1.25, "Combine option tokens",
        "Sum selected options per candidate\nLearned vector for an empty choice", PURPLE, size=10.8)
    box(ax, 9.55, 2.75, 3.95, 1.25, "Action queries · [A, 512]",
        "LayerNorm + dropout\nOne query per legal action", ORANGE, size=10.8)
    arrow(ax, 11.32, 4.98, 7.05, 4.02)
    arrow(ax, 4.52, 3.37, 4.97, 3.37)
    arrow(ax, 9.07, 3.37, 9.52, 3.37)
    ax.text(0.6, 1.6, "A ≤ 64; each candidate is scored against the same state memory in the decoder.",
            fontsize=11.5, color=MUTED)
    save(fig, "model_action_tokens")


def policy_ensemble():
    fig, ax = canvas(15, 9.5)
    title(ax, "3  Score actions and combine two models",
          "Each legal-action query reads the encoded state; the two policies vote by probability", 9.2)

    box(ax, 0.5, 6.55, 2.6, 1.2, "State tokens",
        "[27, 512]", BLUE, size=11.5)
    box(ax, 3.55, 6.25, 4.0, 1.8, "State encoder · ×3 or ×4",
        "8-head self-attention + residual + PostNorm\n"
        "GEGLU FFN: 512 → 2048 → 512\nOutput: contextual state memory", PURPLE, size=9.25)
    arrow(ax, 3.13, 7.15, 3.52, 7.15)

    box(ax, 0.5, 3.9, 2.6, 1.2, "Action queries",
        "[A, 512], A ≤ 64", GREEN, size=11.2)
    box(ax, 8.1, 4.95, 4.15, 2.2, "Action decoder · ×3 or ×4",
        "Q = candidate action; K/V = state memory\n"
        "8-head cross-attention + PostNorm\n"
        "GEGLU FFN: 512 → 2048 → 512\n"
        "No attention between action queries", ORANGE, size=10.0)
    arrow(ax, 7.58, 7.0, 8.07, 6.45)
    arrow(ax, 3.13, 4.5, 8.07, 5.45)
    box(ax, 12.7, 5.35, 1.8, 1.3, "Policy head",
        "Linear 512 → 1\nA action logits", RED, size=9.6, title_size=11)
    arrow(ax, 12.28, 6.0, 12.67, 6.0)

    ax.text(0.55, 3.25, "Two independent passes with the same features; only Transformer depth differs", color=MUTED, fontsize=11)
    box(ax, 0.5, 1.65, 4.0, 1.12, "Model A · 3 + 3 layers",
        "Softmax(logits) → pA", BLUE, size=10.8)
    box(ax, 0.5, 0.25, 4.0, 1.12, "Model B · 4 + 4 layers",
        "Softmax(logits) → pB", BLUE, size=10.8)
    box(ax, 6.0, 0.95, 3.5, 1.12, "Equal ensemble",
        "p = (pA + pB) / 2", PURPLE, size=10.8)
    box(ax, 10.5, 0.95, 4.0, 1.12, "Final decision",
        "KO mask → highest-probability action", GREEN, size=10.4)
    arrow(ax, 4.52, 2.2, 5.97, 1.72)
    arrow(ax, 4.52, 0.8, 5.97, 1.3)
    arrow(ax, 9.52, 1.5, 10.47, 1.5)
    save(fig, "model_policy_ensemble")


if __name__ == "__main__":
    state_tokens()
    action_tokens()
    policy_ensemble()
