
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder

from search.hybrid_search_chunks import COLLECTION, MODEL, SPARSE_KWARGS, SPARSE_MODEL, store
from settings import settings


def load_programs() -> list[Document]:
    """Jeden dokument na plik Markdown; metadane szkolenia (kategoria, dni, PDF) z parqueta po nazwie pliku."""
    szkolenia = pd.read_parquet(settings.szkolenia_parquet).set_index("plik")
    docs = []
    for path in sorted(Path(settings.programy_dir).glob("*.md")):
        row = szkolenia.loc[unicodedata.normalize("NFC", path.name)]
        meta = {"nazwa": row.nazwa, "kategoria": row.kategoria, "dni": int(row.dni), "pdf_url": row.pdf_url, "plik": path.name}
        docs.append(Document(content=path.read_text(encoding="utf-8"), meta=meta))
    return docs


def main() -> None:
    docs = load_programs()
    if not docs:
        sys.exit(
            f"Brak plików .md w {settings.programy_dir} "
            "(rozpakuj data_rag/programy.zip)."
        )

    splitter = RecursiveDocumentSplitter(
        split_length=1000,       # maksymalny rozmiar chunka w znakach
        split_overlap=150,       # nakładanie się chunków
        split_unit="char",
        separators=[
            "\n## ",             # sekcje Markdown
            "\n### ",            # podsekcje
            "\n\n",              # akapity
            "\n",                # linie
            ". ",                # zdania
            " ",                 # słowa
            "",                  # ostatecznie pojedyncze znaki
        ],
    )
    splitter.warm_up()

    chunks = splitter.run(documents=docs)["documents"]

    for chunk in chunks:
        title = chunk.meta["nazwa"]
        if not chunk.content.lstrip("# ").startswith(title):
            chunk.content = f"{title}\n\n{chunk.content}"

    print(
        f"{len(docs)} plików -> {len(chunks)} fragmentów",
        flush=True,
    )

    if store.count_documents() == len({chunk.id for chunk in chunks}):
        print(f"Kolekcja {COLLECTION} jest już zaindeksowana.", flush=True)
        return

    doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
    doc_embedder.warm_up()

    sparse_doc_embedder = FastembedSparseDocumentEmbedder(
        model=SPARSE_MODEL,
        model_kwargs=SPARSE_KWARGS,
    )
    sparse_doc_embedder.warm_up()

    dense_documents = doc_embedder.run(chunks)["documents"]
    embedded = sparse_doc_embedder.run(dense_documents)["documents"]

    store.write_documents(
        embedded,
        policy=DuplicatePolicy.OVERWRITE,
    )

    print(
        f"Zapisano {store.count_documents()} fragmentów "
        f"do kolekcji {COLLECTION}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
