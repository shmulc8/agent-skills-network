"""Fetch, parse, embed agent skills from VoltAgent/awesome-agent-skills.

The source README contains ~911 one-liners in the form:
    - **[org/skill-name](url)** - short description

We parse those lines, embed `name: description` with MiniLM, reduce to 3D
with UMAP, cluster with KMeans, and label each cluster with TF-IDF terms.

Team = GitHub org prefix of the skill name (e.g. "anthropics/pdf" -> "anthropics").

Run with:
    uv run --python 3.12 \
      --with requests --with sentence-transformers \
      --with umap-learn --with scikit-learn --with numpy \
      pipeline/fetch_and_embed.py
"""
import colorsys
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"

REPO_ROOT = Path(__file__).resolve().parent.parent
README_CACHE = REPO_ROOT / "pipeline" / "voltagent-readme.md"
OUT = REPO_ROOT / "docs" / "data.json"

README_URL = (
    "https://raw.githubusercontent.com/"
    "VoltAgent/awesome-agent-skills/main/README.md"
)

LINE_RE = re.compile(r"^- \*\*\[([^\]]+)\]\(([^)]+)\)\*\* - (.+)$")


def fetch_readme() -> str:
    if README_CACHE.exists():
        return README_CACHE.read_text(encoding="utf-8")
    print(f"Fetching {README_URL} ...")
    req = urllib.request.Request(README_URL, headers={"User-Agent": "semantic-skills/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")
    README_CACHE.write_text(data, encoding="utf-8")
    return data


def parse_skills(readme: str) -> list[dict]:
    skills = []
    seen_names = set()
    for raw in readme.splitlines():
        m = LINE_RE.match(raw.rstrip())
        if not m:
            continue
        name, url, description = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if name in seen_names:
            continue
        seen_names.add(name)
        team = name.split("/", 1)[0] if "/" in name else name
        skills.append(
            {
                "name": name,
                "team": team,
                "url": url,
                "description": description,
            }
        )
    return skills


def gemma_label_cluster(bullets: str, n_members: int) -> str:
    """Ask Gemma for a 3-6 word human label for a cluster of agent skills."""
    system = (
        "You label clusters of AI agent skills. You will be shown a sample of "
        "skills from one cluster. Return ONLY a 3-to-6 word label capturing what "
        "the cluster is about — concrete domain or capability, not fluff. "
        "No quotes, no trailing period."
    )
    user = (
        f"Cluster of {n_members} agent skills. Sample:\n\n{bullets}\n\n"
        "Label this cluster in 3-6 words:"
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "think": False,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 32},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("message") or {}).get("content", "").strip()
    # Strip quotes, trailing punctuation, collapse whitespace
    text = text.strip().strip('"\'').strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".,;:")
    # Keep first line only (in case the model dumps multiple)
    text = text.splitlines()[0] if text else f"cluster"
    return text[:80] if text else "cluster"


def team_color(team: str) -> str:
    """Deterministic HSL color for a team name."""
    h = int(hashlib.md5(team.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hls_to_rgb(h, 0.68, 0.62)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def main():
    readme = fetch_readme()
    skills = parse_skills(readme)
    print(f"Parsed {len(skills)} skills")

    team_counts = Counter(s["team"] for s in skills)
    print(f"Teams: {len(team_counts)} (top 10: {team_counts.most_common(10)})")

    texts = [f"{s['name']}: {s['description']}" for s in skills]

    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding...")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)
    print(f"Embeddings shape: {embs.shape}")

    print("UMAP 3D...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.2,
        n_components=3,
        metric="cosine",
        random_state=42,
    )
    xyz = reducer.fit_transform(embs)
    xyz = xyz - xyz.mean(axis=0)
    radius = np.linalg.norm(xyz, axis=1).max()
    xyz = xyz / (radius + 1e-9)

    print("KMeans clustering...")
    n_clusters = 10
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(embs)

    # LLM cluster labels — Gemma 4 E2B via Ollama, /api/chat with think:false
    print(f"Labeling clusters with {OLLAMA_MODEL} via Ollama...")
    cluster_labels = []
    for i in range(n_clusters):
        members = [s for c, s in zip(clusters, skills) if c == i]
        # Pick a diverse-ish sample: skills closest to the centroid
        idxs = np.where(clusters == i)[0]
        centroid = embs[idxs].mean(axis=0)
        dists = np.linalg.norm(embs[idxs] - centroid, axis=1)
        order = idxs[np.argsort(dists)]
        sample_idxs = order[: min(30, len(order))]
        bullets = "\n".join(
            f"- {skills[j]['name']}: {skills[j]['description']}" for j in sample_idxs
        )
        label = gemma_label_cluster(bullets, len(members))
        cluster_labels.append(label)
        print(f"  Cluster {i} ({len(members)} skills): {label}")

    # Build team metadata — sorted by count desc, stable colors
    teams_sorted = sorted(team_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    team_list = [
        {"id": name, "count": count, "color": team_color(name)}
        for name, count in teams_sorted
    ]

    points = []
    for i, s in enumerate(skills):
        points.append(
            {
                "id": i,
                "x": float(xyz[i, 0]),
                "y": float(xyz[i, 1]),
                "z": float(xyz[i, 2]),
                "cluster": int(clusters[i]),
                "team": s["team"],
                "name": s["name"],
                "title": s["name"],
                "url": s["url"],
                "description": s["description"],
                "body": s["description"],
                "preview": s["description"][:280],
                "date": "",
            }
        )

    data = {
        "points": points,
        "clusters": [
            {"id": i, "label": cluster_labels[i], "count": int((clusters == i).sum())}
            for i in range(n_clusters)
        ],
        "teams": team_list,
    }

    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(points)} points, {len(team_list)} teams)")


if __name__ == "__main__":
    main()
