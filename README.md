# Agent Skills Latent Space

Interactive 3D map of 900+ agent skills from [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills), projected into a latent space and colored by topic cluster or authoring team.

## How it works

1. Fetch the `awesome-agent-skills` README — a curated list of ~900 skills in the form `- **[org/skill-name](url)** - short description`.
2. Parse each line into `(name, team, url, description)` with one regex. Team = GitHub-org prefix of the skill name.
3. Embed `name: description` with `sentence-transformers/all-MiniLM-L6-v2` (384-d).
4. Reduce to 3D with UMAP (cosine metric).
5. Cluster in the 384-d space with KMeans (k=10).
6. Label each cluster by asking **Gemma 4 E2B** (via Ollama `/api/chat` with `think:false`) for a 3–6 word title over the 30 cluster members closest to the centroid.
7. Render with Three.js: glowing points, persistent nearest-neighbor web, hover tooltip, search, click-to-read info panel, and a "Color by topic / team" toggle.

## Layout

```
docs/         static site (GitHub Pages serves from here)
pipeline/     fetch_and_embed.py — fetches, embeds, projects, writes docs/data.json
```

## Rebuild the data

Requires [Ollama](https://ollama.com) running locally with `gemma4:e2b` pulled.

```bash
ollama pull gemma4:e2b
ollama serve  # usually already running

uv run --python 3.12 \
  --with requests --with sentence-transformers \
  --with umap-learn --with scikit-learn --with numpy \
  pipeline/fetch_and_embed.py
```

The script caches the fetched README at `pipeline/voltagent-readme.md`. Delete it to force a re-fetch.

## Serve locally

```bash
python3 -m http.server 8767 --directory docs
# open http://localhost:8767
```
