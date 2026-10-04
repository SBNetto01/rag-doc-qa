"""Command-line interface for rag-doc-qa.

Examples::

    ragdoc ingest ./docs
    ragdoc ask "How is payroll calculated?"
    ragdoc ask "How is payroll calculated?" --use-llm
    ragdoc serve
"""

from __future__ import annotations

from pathlib import Path

import typer

from app.embedders import build_embedder
from app.pipeline import RagPipeline

app = typer.Typer(help="Key-free RAG question answering over your own documents.")

INDEX_DIR = "data/index"


def _pipeline(embedder: str, index_dir: str) -> RagPipeline:
    return RagPipeline(build_embedder(embedder), index_dir=index_dir)


@app.command()
def ingest(
    paths: list[Path],
    embedder: str = typer.Option("offline", help="offline | transformers"),
    index_dir: str = typer.Option(INDEX_DIR, help="Directory where the FAISS index is stored"),
    chunk_size: int = typer.Option(800, help="Chunk size in characters"),
) -> None:
    """Index text/markdown/PDF files into a local vector index."""
    expanded: list[Path] = []
    for raw in paths:
        if raw.is_dir():
            expanded.extend(raw.rglob("*"))
        else:
            expanded.append(raw)
    files = [p for p in expanded if p.is_file()]
    supported = [p for p in files if p.suffix.lower() in {".txt", ".md", ".markdown", ".csv", ".pdf"}]
    if not supported:
        typer.echo("No supported files found (txt/md/markdown/csv/pdf).")
        raise typer.Exit(2)
    pipeline = RagPipeline(build_embedder(embedder), index_dir=index_dir, chunk_size=chunk_size)
    summary = pipeline.ingest_files(supported)
    typer.echo(f"Ingested {summary['documents']} document(s) -> {summary['chunks']} chunks "
               f"into {index_dir} ({embedder} embeddings).")


@app.command()
def ask(
    question: str,
    embedder: str = typer.Option("offline", help="offline | transformers"),
    index_dir: str = typer.Option(INDEX_DIR, help="Directory where the FAISS index is stored"),
    k: int = typer.Option(4, help="Number of chunks to retrieve"),
    use_llm: bool = typer.Option(False, help="Generate the answer with a local Ollama server"),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON"),
) -> None:
    """Ask a question against the local index."""
    pipeline = _pipeline(embedder, index_dir)
    if not pipeline.index_exists():
        typer.echo("No index found — run `ragdoc ingest ./docs` first.", err=True)
        raise typer.Exit(2)
    result = pipeline.ask(question, k=k, use_llm=use_llm)
    if as_json:
        import json

        typer.echo(json.dumps({
            "answer": result.answer,
            "sources": [{"source": s.source, "score": s.score} for s in result.sources],
            "generation": result.generation,
            "generated": result.generated,
        }, ensure_ascii=False, indent=2))
        return
    typer.echo(result.answer)
    typer.echo("")
    typer.echo("Sources (score = cosine similarity, higher is closer):")
    for hit in result.sources:
        typer.echo(f"  - {hit.source}  (score {hit.score})")
    if result.generation:
        typer.echo("")
        typer.echo(f"[LLM generation] {result.generation}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Start the FastAPI service."""
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()