from typing import Annotated

from haystack.tools import tool

from search import hybrid_openai
from search import hybrid_search_chunks


@tool
def search_courses(
    query: Annotated[str, "topic or skill the user is asking about"],
) -> str:
    """Wyszukuje szkolenia w katalogu (hybrydowo: BM25 + embeddingi).
    Zwraca pasujące szkolenia: nazwę z opisem, kategorię, liczbę dni i link do PDF."""
    docs = hybrid_openai.search(query)
    if not docs:
        return "Brak pasujących szkoleń."
    return "\n\n".join(
        f"{doc.content}\n"
        f"Kategoria: {doc.meta['kategoria']}, dni: {doc.meta['dni']}, PDF: {doc.meta['pdf_url']}"
        for doc in docs
    )


@tool
def search_in_markdown(
    query: Annotated[str, "topic or skill the user is asking about"],
) -> str:
    """Wyszukuje informacje w plikach Markdown w katalogu danych.
    Zwraca pasujące fragmenty z nazwą pliku i treścią."""
    
    docs = hybrid_search_chunks.search(query)
    if not docs:
        return "Brak pasujących informacji w Markdown."
    return "\n\n".join(
        f"Plik: {doc.meta['filename']}\n{doc.content}"
        for doc in docs
    )