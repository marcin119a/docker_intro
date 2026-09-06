
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder

from search.hybrid_search_chunks import COLLECTION, MODEL, SPARSE_KWARGS, SPARSE_MODEL, store
from settings import settings


def load_programs() -> list[Document]:
    """Jeden dokument na plik Markdown; metadane szkolenia (kategoria, dni, PDF) z parqueta po nazwie pliku."""
    # macOS zapisuje polskie znaki w nazwach plików w postaci rozłożonej (NFD), parquet ma złożoną (NFC) — normalizujemy.
    szkolenia = pd.read_parquet(settings.szkolenia_parquet).set_index("plik")
    docs = []
    for path in sorted(Path(settings.programy_dir).glob("*.md")):
        row = szkolenia.loc[unicodedata.normalize("NFC", path.name)]
        meta = {"nazwa": row.nazwa, "kategoria": row.kategoria, "dni": int(row.dni), "pdf_url": row.pdf_url, "plik": path.name}
        docs.append(Document(content=path.read_text(encoding="utf-8"), meta=meta))
    return docs


def split_program(text: str) -> list[str]:
    """Dzieli program szkolenia na punkty ("1.  Wstęp…", "2.  …" wraz z podpunktami a., b., …) — jeden fragment to jeden punkt.

    Każdy fragment zaczyna się od tytułu szkolenia (pierwsza linia pliku), żeby zapytanie "wolumeny w dockerze"
    trafiało w punkt "Wolumeny" właśnie ze szkolenia o Dockerze — sam punkt często nie powtarza tematu szkolenia.
    """
    title, _, program = text.partition("\n")
    title = title.lstrip("# ")
    points = re.split(r"\n(?=\d+\.\s)", program.replace("## Program", "", 1))
    return [f"{title}\n{point.strip()}" for point in points if point.strip()]


def main() -> None:
    docs = load_programs()
    if not docs:
        sys.exit(f"Brak plików .md w {settings.programy_dir} (rozpakuj data_rag/programy.zip).")

    # 1. Chunki: DocumentSplitter kopiuje metadane szkolenia do każdego fragmentu i dodaje source_id (id pliku).
    splitter = DocumentSplitter(split_by="function", splitting_function=split_program)
    chunks = splitter.run(documents=docs)["documents"]
    print(f"{len(docs)} plików -> {len(chunks)} fragmentów", flush=True)

    # 2. Kolekcja kompletna (tyle samo fragmentów, co teraz) -> nic do roboty.
    if store.count_documents() == len({chunk.id for chunk in chunks}):
        print(f"Kolekcja {COLLECTION} jest już zaindeksowana.", flush=True)
        return

    # 3. Embeddingi: gęste (sentence-transformers) i rzadkie (BM25) — te same modele, którymi hybrid_search_chunks.py
    #    embeduje zapytania.
    doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
    doc_embedder.warm_up()
    sparse_doc_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_MODEL, model_kwargs=SPARSE_KWARGS)
    sparse_doc_embedder.warm_up()
    embedded = sparse_doc_embedder.run(doc_embedder.run(chunks)["documents"])["documents"]

    # 4. Zapis do Qdranta. OVERWRITE: fragment o tym samym id nadpisuje się, nie dubluje.
    store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
    print(f"Zapisano {store.count_documents()} fragmentów do kolekcji {COLLECTION}.", flush=True)


if __name__ == "__main__":
    main()
