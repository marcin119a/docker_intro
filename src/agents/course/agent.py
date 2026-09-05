from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from agents.course.tools import search_courses
from settings import settings

INSTRUCTIONS = (
    "Jesteś doradcą szkoleniowym. Odpowiadasz po polsku, krótko i konkretnie.\n"
    "Zasady:\n"
    "- Zanim cokolwiek polecisz, wyszukaj szkolenia narzędziem search_courses.\n"
    "- Polecaj wyłącznie szkolenia zwrócone przez narzędzie — nie wymyślaj innych.\n"
    "- Przy każdym poleconym szkoleniu podaj kategorię, liczbę dni i link do PDF.\n"
    "- Jeśli żadne szkolenie nie pasuje do pytania, powiedz to wprost."
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model=settings.openai_model,
        api_key=Secret.from_token(settings.openai_api_key),
    ),
    system_prompt=INSTRUCTIONS,
    tools=[search_courses],
)