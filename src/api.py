
from fastapi import FastAPI
from haystack.dataclasses import ChatMessage
from pydantic import BaseModel

from search.hybrid import search as hybrid_search
from search import hybrid_search_chunks
from agents.course.agent import agent

app = FastAPI(title="Doradca szkoleniowy", version="0.1.0")


class Question(BaseModel):
    pytanie: str


class Answer(BaseModel):
    odpowiedz: str


class Course(BaseModel):
    nazwa: str
    kategoria: str
    dni: int
    pdf_url: str
    score: float

class Fragment(BaseModel):
    nazwa: str
    kategoria: str
    dni: int
    pdf_url: str
    fragment: str 
    score: float

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(q: str) -> list[Course]:
    """Wyszukiwanie hybrydowe w katalogu szkoleń (bez LLM)."""
    return [
        Course(
            nazwa=doc.meta["nazwa"],
            kategoria=doc.meta["kategoria"],
            dni=int(doc.meta["dni"]),
            pdf_url=doc.meta["pdf_url"],
            score=doc.score,
        )
        for doc in hybrid_search.search(q)
    ]

@app.get("/search/chunks")
def search_chunks(q: str) -> list[Fragment]:
    """Wyszukiwanie hybrydowe po fragmentach programów szkoleń (kolekcja szkolenia_chunki — buduje ją usługa indexer)."""
    return [
        Fragment(
            nazwa=doc.meta["nazwa"],
            kategoria=doc.meta["kategoria"],
            dni=int(doc.meta["dni"]),
            pdf_url=doc.meta["pdf_url"],
            fragment=doc.content.removeprefix(f"{doc.meta['nazwa']}\n\n"),  # bez tytułu szkolenia, który indexer dokleja na początku
            score=doc.score,
        )
        for doc in hybrid_search_chunks.search(q)
    ]


@app.post("/chat")
def chat(body: Question) -> Answer:
    """Pytanie do agenta-doradcy; zwraca odpowiedź LLM."""
    wynik = agent.run([ChatMessage.from_user(body.pytanie)])
    return Answer(odpowiedz=wynik["messages"][-1].text)
