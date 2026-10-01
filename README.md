# Python Agent TUI

Pythonowa warstwa wykonania w terminalu korzystająca z **Harness Router** do podejmowania decyzji o wyborze następnego narzędzia oraz modelu OpenRouter's [`openrouter/free`](https://openrouter.ai/docs/guides/routing/routers/free-router) dla argumentów narzędzi i odpowiedzi.

## Wymagania

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (do zarządzania zależnościami)

## Szybki start

```bash
# Instalacja zależności
uv sync

# Skopiowanie szablonu środowiska
cp .env.example .env

# Ustaw swój klucz API (lub wyeksportuj go w powłoce)
export OPENROUTER_API_KEY=your_key_here

# Uruchom agenta
uv run agent-tui
```

## Demo

Aby uruchomić lokalne demo bez poświadczeń i żądań sieciowych:

```bash
uv run agent-tui --demo
```

## Rozszerzenia i pamięć

Agent obsługuje skille `SKILL.md`, marketplace pluginów (w tym Superpowers), serwery MCP, middleware z hookami, zarządzanie kontekstem i trwałą pamięć SQLite dla workspace.

```text
/plugins
/plugin install superpowers
/skills
/skill superpowers:brainstorming
/mcp
/hooks
/context
/memory
```

Przykładowy serwer MCP i hooki uruchamiasz z katalogu projektu:

```bash
uv run agent-tui --extensions examples/extensions.toml
```

Konfiguracja, komendy i opis przykładów: [dokumentacja rozszerzeń](docs/extensions.md).

## Struktura projektu

- `src/agent_tui/` – Główny kod źródłowy agenta TUI
- `src/agent_tui/openrouter.py` – Integracja z OpenRouter
- `src/agent_tui/routing.py` – Logika wyboru następnego narzędzia przez Harness Router
- `src/agent_tui/tools.py` – Dostępne narzędzia dla agenta
- `src/agent_tui/__main__.py` – Punkt wejściowy

## Licencja

Ten projekt jest open source. Zobacz plik LICENSE dla szczegółów.
