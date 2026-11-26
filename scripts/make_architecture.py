"""Generate architecture.png (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")


def box(x, y, w, h, text, color="#e8f0fe", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#344", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else None)


def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
    if label:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=7.5, color="#555")


box(4.2, 6.2, 2.6, 0.6, "Client / curl / scripts", "#fff3cd", True)
box(3.0, 4.6, 5.0, 1.2, "FastAPI  (src/main.py, api/routes.py)\nauth - validation - logging - error handlers", "#d1e7dd", True)
box(0.2, 2.6, 3.4, 1.4, "Ingestion\nPyMuPDF parse -> section detect\n-> sentence chunking -> embed")
box(4.0, 2.6, 3.4, 1.4, "RAG pipeline\n(rewrite) -> retrieve -> diversify\n-> prompt -> LLM -> citations")
box(7.8, 2.6, 3.0, 1.4, "LLM router (fallback chain)\nClaude -> Gemini -> (Ollama)")
box(0.2, 0.5, 2.4, 1.0, "sentence-transformers\n(local embeddings)")
box(3.0, 0.5, 2.4, 1.0, "Qdrant\nvectors + page/section payload", "#f8d7da")
box(5.8, 0.5, 2.6, 1.0, "PostgreSQL / SQLite\npapers, query history, stats", "#f8d7da")
box(8.8, 0.5, 2.0, 1.0, "PDF store\n(uploads/)", "#f8d7da")
arrow(5.5, 6.2, 5.5, 5.8); arrow(4.2, 4.6, 2.2, 4.0, "upload"); arrow(5.7, 4.6, 5.7, 4.0, "query")
arrow(7.4, 3.3, 7.8, 3.3); arrow(1.4, 2.6, 1.4, 1.5); arrow(2.8, 2.6, 4.0, 1.5); arrow(5.2, 2.6, 4.4, 1.5)
arrow(6.4, 2.6, 6.9, 1.5, "log"); arrow(3.0, 2.8, 5.8, 1.4)
ax.text(5.5, 6.95, "Research Paper Assistant - architecture", ha="center", fontsize=12, fontweight="bold")
plt.savefig("architecture.png", dpi=150, bbox_inches="tight")
