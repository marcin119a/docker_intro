
import sys

from haystack import Document, Pipeline
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from settings import settings

# Te same modele co w hybrid_search_qdrant.py — zapytanie musi być zembedowane tak samo jak fragmenty (indexer.py).
MODEL = settings.embedding_model
SPARSE_MODEL = "Qdrant/bm25"
SPARSE_KWARGS = {"disable_stemmer": True}
COLLECTION = "szkolenia_chunki"

# 1. Embeddery zapytania: gęsty (sentence-transformers) i rzadki (BM25).
text_embedder = SentenceTransformersTextEmbedder(model=MODEL)
text_embedder.warm_up()
sparse_text_embedder = FastembedSparseTextEmbedder(model=SPARSE_MODEL, model_kwargs=SPARSE_KWARGS)
sparse_text_embedder.warm_up()

# 2. Magazyn: kolekcja z dwoma wektorami na fragment (gęsty + rzadki). Jeśli jej nie ma, powstaje pusta —
#    wypełnia ją indexer.py, a do tego czasu search() zwraca pustą listę.
store = QdrantDocumentStore(
    url=settings.qdrant_url,
    index=COLLECTION,
    embedding_dim=len(text_embedder.run("wymiar")["embedding"]),
    use_sparse_embeddings=True,
    sparse_idf=True,  # IDF dla BM25 liczy Qdrant po swojej stronie
)


def create_pipeline() -> Pipeline:
    # 3. Pipeline zapytania: dwa embeddery + jeden retriever; fuzję rankingów (RRF) robi Qdrant.
    #    RRF waży POZYCJE w obu rankingach, nie surowe score, więc różne skale BM25 i cosinusa nie psują wyniku.
    #    rrf_weights w kolejności [rzadki (BM25), gęsty (embeddingi)]: embeddingi mają większy wpływ.
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("sparse_embedder", sparse_text_embedder)
    pipeline.add_component(
        "retriever",
        QdrantHybridRetriever(document_store=store, top_k=10, rrf_weights=[0.4, 0.6]),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    return pipeline


# Pipeline budujemy raz: Haystack nie pozwala dodać tej samej instancji komponentu do drugiego pipeline'u.
pipeline = create_pipeline()


# 4. Funkcja wyszukiwania — ten sam interfejs co hybrid_search_qdrant.search (używa jej api.py: GET /search/chunks).
def search(query: str) -> list[Document]:
    result = pipeline.run({"text_embedder": {"text": query}, "sparse_embedder": {"text": query}})
    return result["retriever"]["documents"]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "wolumeny i sieci w docker compose"
    print(f"Zapytanie: {query}\n")
    for doc in search(query):
        # Pierwsza linia fragmentu to tytuł szkolenia (patrz indexer.split_program), dalej punkt programu.
        _, _, fragment = doc.content.partition("\n")
        print(f"{doc.score:.4f}  {doc.meta['nazwa']}  ({doc.meta['kategoria']}, {doc.meta['dni']} dni)")
        print("        " + fragment.replace("\n", "\n        "))
        print()
