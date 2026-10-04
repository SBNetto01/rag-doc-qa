# Deploying rag-doc-qa

The service is deliberately **key-free**, so deploying is mostly: run the image,
point `RAG_INDEX_DIR` at persistent storage, ingest documents.

## Option 1 — Docker (any host)

```bash
docker compose up --build        # API on :8000
docker compose --profile llm up --build   # + local Ollama for generation
```

Make the index volume persistent (done by default: `rag-index` volume).

## Option 2 — Fly.io

```bash
fly launch --name rag-doc-qa --no-deploy
fly volumes create rag_index --size 1      # persistent index
fly deploy
```

Common pitfalls:

- The working directory must allow writing `data/index` (set
  `RAG_INDEX_DIR` to a mounted volume).
- Ingestion is a separate step from serving: build your index locally
  (`ragdoc ingest ./docs`) and upload it, or POST documents to `/ingest`
  after deployment.

## Option 3 — Render (web service)

1. New Web Service → repo → Docker runtime (uses the `Dockerfile`).
2. Add a disk mount at `/data`; set `RAG_INDEX_DIR=/data/index`.
3. Optional env for local LLM generation: `RAG_OLLAMA_BASE_URL` pointing at
   an Ollama service (e.g. another Render service, or run Ollama where you host it).

## Option 4 — Railway

Use the Dockerfile via the "Deploy from Dockerfile" flow. Add a volume at
`/data` and set `RAG_INDEX_DIR=/data/index`.

## After deployment

```bash
curl -X POST https://<your-host>/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"filename":"policy.md","text":"..."}]}'

curl -X POST https://<your-host>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the document about?","k":3}'
```

## Sizing

- `offline` embedder: no model memory, tiny footprint; a tiniest instance is
  enough for demo loads.
- `transformers` embedder: needs ~2–4 GB RAM (model + torch); use at least a
  small instance with 4 GB.
- Optional Ollama generation: separate service, aim for 8 GB+ if models > 3B.