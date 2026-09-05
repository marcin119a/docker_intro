import sys

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

df = pd.read_parquet(settings.szkolenia_parquet)
docs = [
    Document(
        content=f"{row.nazwa}\n{row.opis}",
        meta={
            "nazwa": row.nazwa,
            "kategoria": row.kategoria,
            "dni": row.dni,
            "pdf_url": row.pdf_url,
        },
    )
    for row in df.itertuples()
]

store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
doc_embedder.warm_up()
store.write_documents(doc_embedder.run(docs)["documents"], policy=DuplicatePolicy.OVERWRITE)

def create_pipeline() -> Pipeline:
    # 3. Pipeline zapytania: dwa retrievery + złączenie wyników (Reciprocal Rank Fusion)
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
    query = " ".join(sys.argv[1:]) or "szkolenie z Docker"
    print(f"Zapytanie: {query}\n")
    for doc in search(query):
        print(f"{doc.score:.4f}  {doc.meta['nazwa']}  ({doc.meta['kategoria']}, {doc.meta['dni']} dni)")
        print(f"        {doc.meta['pdf_url']}")