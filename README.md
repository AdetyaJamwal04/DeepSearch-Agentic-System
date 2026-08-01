# 🔍 DeepSearch — Agentic Research System

An autonomous deep research pipeline powered by **LangGraph**, **Google Gemini**, and **Tavily**. Give it a question — it researches, reflects, iterates, and delivers a cited markdown report.

## How It Works

DeepSearch decomposes complex queries into sub-questions, runs multi-round web searches with automatic quality filtering, extracts evidence via LLM, reflects on coverage gaps, and synthesizes everything into a structured research report with inline citations.

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Synthesizer   │──▶ Extracts intent, scope, entities, architecture
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Sub-question Engine  │──▶ Decomposes into independent research threads
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Search Query Gen     │──▶ 2–4 diverse angle-based queries per sub-question
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Search Executor      │──▶ Tavily web search + junk/PDF artifact filtering
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Evidence Extractor   │──▶ LLM extracts factual claims from sources
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Knowledge Store      │──▶ Accumulates evidence across rounds
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Reflection Agent     │──▶ Evaluates coverage, identifies gaps
└─────────┬───────────┘
     ┌────┴────┐
     │ Gaps?   │
     └────┬────┘
    Yes ◄─┘ └─► No
     │          │
     ▼          ▼
  Loop back   ┌─────────────────────┐
  (max 3      │ Report Synthesizer   │──▶ Cited markdown report
   rounds)    └─────────────────────┘
```

## Features

- **Iterative Research Loop** — Up to 3 rounds of search → extract → reflect, narrowing gaps each round
- **Structured Query Analysis** — Classifies intent, scope, and architecture before decomposition
- **Intelligent Decomposition** — Breaks multi-faceted queries into independently researchable sub-questions
- **Diverse Search Angles** — Generates 2–4 angle-based queries per sub-question (not just rephrasings)
- **Quality Filtering** — Removes junk results: short content, PDF extraction artifacts, link-heavy pages
- **Evidence Deduplication** — URL-level dedup within sub-question batches prevents duplicate extraction
- **Gap-Aware Reflection** — Evaluates evidence depth, source diversity, and specificity before deciding to loop
- **Cited Reports** — Final output uses numbered inline citations `[1]`, `[2]` mapped to source URLs
- **API Server** — FastAPI with both blocking (`POST /research`) and streaming (`GET /research/stream`) endpoints
- **Throttled LLM Calls** — Semaphore-based concurrency control to avoid hitting API rate limits

## Setup

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **Tavily API Key** — [app.tavily.com](https://app.tavily.com)
- **Google Gemini API Key** — [aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/AdetyaJamwal04/DeepSearch-Agentic-System.git
cd DeepSearch-Agentic-System

# Install dependencies
uv sync
```

### Configuration

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_tavily_key_here
GEMINI_API_KEY=your_gemini_key_here
```

Optionally set the model (defaults to `gemini-2.5-flash-lite`):

```env
MODEL_NAME=gemini-2.5-flash-lite
```

## Usage

### CLI — Standalone Pipeline

```bash
# Run via LangGraph (recommended)
uv run python graph.py

# Run via imperative loop (alternative)
uv run python main.py
```

Reports are saved to `reports/` with timestamped filenames.

### API Server

```bash
uv run python -m uvicorn api:app --host 127.0.0.1 --port 8080
```

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/research` | Blocking research — returns full report |
| `GET` | `/research/stream` | SSE stream — real-time progress updates |

#### POST /research

```bash
curl -X POST http://127.0.0.1:8080/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Impact of quantum computing on cryptography"}'
```

**Response:**
```json
{
  "query": "Impact of quantum computing on cryptography",
  "report": "# Report Title\n\n## Executive Summary\n..."
}
```

#### GET /research/stream (SSE)

```bash
curl "http://127.0.0.1:8080/research/stream?query=Impact+of+quantum+computing+on+cryptography"
```

Streams Server-Sent Events as each pipeline node completes.

## Project Structure

```
DeepSearch-Agentic-System/
├── agents/
│   ├── query_synthesizer.py      # Stage 1 — Structured query analysis
│   ├── subquestion_generator.py  # Stage 2 — Query decomposition
│   ├── search_query_generator.py # Stage 3 — Diverse search query generation
│   ├── search_executor.py        # Stage 4 — Tavily search + quality filtering
│   ├── content_processor.py      # Stage 5 — LLM evidence extraction
│   ├── knowledge_store.py        # Stage 6 — In-memory evidence accumulator
│   ├── reflection.py             # Stage 7 — Coverage evaluation + gap detection
│   └── report_synthesizer.py     # Stage 8 — Final report generation
├── models/
│   ├── model.py                  # Throttled Gemini LLM client
│   └── web_client.py             # Async Tavily client
├── schemas/
│   └── schema.py                 # Pydantic data models
├── graph.py                      # LangGraph state machine (production entry)
├── main.py                       # Standalone CLI runner (imperative loop)
├── api.py                        # FastAPI server (blocking + SSE endpoints)
├── config.py                     # Environment config loader
└── pyproject.toml                # Project metadata + dependencies
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini (via `langchain-google-genai`) |
| Agent Orchestration | LangGraph |
| Web Search | Tavily API |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic |
| Package Management | uv |

## License

This project is for educational and personal use.
