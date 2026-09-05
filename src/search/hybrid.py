import sys
from pathlib import Path

import pandas as pd
from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from settings import settings

MODEL = settings.embedding_model

INDEX_PATH = Path(settings.index_path)


def load_docs() -> list[Document]:
    df = pd.read_parquet(settings.szkolenia_parquet)
    return [
        Document(
            content=f"{row.nazwa}\n{row.opis}",
            meta={
                "nazwa": row.nazwa,
                "kategoria": row.kategoria,
                "dni": int(row.dni),
                "pdf_url": row.pdf_url,
            },
        )
        for row in df.itertuples()
    ]


def build_index(path: Path = INDEX_PATH) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
    doc_embedder.warm_up()
    store.write_documents(doc_embedder.run(load_docs())["documents"], policy=DuplicatePolicy.OVERWRITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_to_disk(str(path))
    return store


def load_index(path: Path = INDEX_PATH) -> InMemoryDocumentStore:
    if path.exists():
        return InMemoryDocumentStore.load_from_disk(str(path))
    print(f"Brak indeksu {path} – buduję go (embeddingi {MODEL})...", file=sys.stderr)
    return build_index(path)


store = load_index()


def create_pipeline() -> Pipeline:
    # Pipeline zapytania: dwa retrievery + złączenie wyników (Reciprocal Rank Fusion)
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=MODEL))
    pipeline.add_component("bm25", InMemoryBM25Retriever(store, top_k=10))
    pipeline.add_component("embedding", InMemoryEmbeddingRetriever(store, top_k=10))
    pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=5))
    pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
    pipeline.connect("bm25", "joiner")
    pipeline.connect("embedding", "joiner")
    return pipeline


def search(query: str) -> list[Document]:
    pipeline = create_pipeline()
    result = pipeline.run({"text_embedder": {"text": query}, "bm25": {"query": query}})
    return result["joiner"]["documents"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--rebuild" in args:
        # Przebudowa indeksu po zmianie danych: python src/search/hybrid.py --rebuild [zapytanie]
        args.remove("--rebuild")
        store = build_index()
    query = " ".join(args) or "szkolenie z Docker"
    print(f"Zapytanie: {query}\n")
    for doc in search(query):
        print(f"{doc.score:.4f}  {doc.meta['nazwa']}  ({doc.meta['kategoria']}, {doc.meta['dni']} dni)")
        print(f"        {doc.meta['pdf_url']}")
