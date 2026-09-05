import sys

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole

from agents.course.agent import agent


def main() -> None:
    if len(sys.argv) > 1:
        # Tryb jednorazowy: pytanie przekazane jako argument.
        pytanie = " ".join(sys.argv[1:])
        agent.run([ChatMessage.from_user(pytanie)], streaming_callback=print_streaming_chunk)
        return

    print("Doradca szkoleniowy (wpisz 'exit', aby zakończyć)")
    historia: list[ChatMessage] = []
    while True:
        try:
            pytanie = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pytanie:
            continue
        if pytanie.lower() in {"exit", "quit"}:
            break
        print()
        wynik = agent.run(
            historia + [ChatMessage.from_user(pytanie)],
            streaming_callback=print_streaming_chunk,
        )
        # Agent sam dokłada system prompt przy każdym uruchomieniu — nie trzymamy go w historii.
        historia = [m for m in wynik["messages"] if not m.is_from(ChatRole.SYSTEM)]


if __name__ == "__main__":
    main()