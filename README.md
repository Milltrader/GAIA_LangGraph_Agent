# GAIA LangGraph Agent

A LangGraph‑based multimodal agent that answers the **GAIA Unit 4** evaluation questions.
It can understand or execute *text, audio, code, spreadsheets and images* and is able to fetch fresh information from the Web.
It was a final projecy for a course from Hugging Face Agents Course (https://huggingface.co/learn/agents-course).

---

##  Key features

| Capability            | Tool                              | What it does                                                                                                   |
| --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Audio**             | `transcribe_audio`                |  Sends the task‑attachment to the Whisper API and returns the first 2 000 chars of transcription               |
| **Images**            | `deep_vision`                     |  A lightweight GPT‑4o‑Vision wrapper that produces a JSON description (caption, objects, colours)              |
| **Spreadsheets**      | `excel_total`                     |  Loads the attached `.xlsx`, sums the *Sales* column and returns the value with two decimals                   |
| **Python**            | `run_python_file` / `python_tool` |  Executes an attached `.py` file or an inline code snippet and surfaces the final `print` or `result` variable |
| **Web search**        | `tavily_search` → `ddg_snippet`   |  Live SERP via Tavily (free tier) with DuckDuckGo fallback / rate‑limit back‑off                               |
| **Wikipedia**         | `wiki_page`                       |  Downloads up to 10 kB of the best matching English page and caches locally                                    |
| **File resolver**     | `resolve_task_file`               |  Maps a *task\_id* to the actual attachment path (e.g. `1f97…_Homework.mp3`)                                   |
| **Answer formatting** |  `give_answer`                    |  Guarantees an **exact‑match**, punctuation‑free final reply                                                   |

---

##  Setup

```bash
# 1 — create & activate virtual‑env
python -m venv .venv
source .venv/bin/activate   #   Windows: .venv\Scripts\activate

# 2 — install requirements
pip install -r requirements.txt

# 3 — set secrets (e.g. in .env or your shell profile)
export OPENAI_API_KEY=sk‑…
export TAVILY_API_KEY=tvly‑…   # optional
```

> **Note** The agent was tested with Python 3.10/3.11 and requires FFmpeg‑free
> execution (audio sent as‑is to Whisper‑API).

---

##  Running

### 1 — Developer test helpers

```python
from agent import debug_single

# one‑shot with an arbitrary question or GAIA task‑id
print(debug_single("Who nominated the dinosaur featured article promoted in Nov 2016?"))
```

### 2 — Gradio evaluation UI

```bash
python app.py      # -> open http://localhost:7860
```

The interface lets you log‑in with a HF account, fetch the official
question set and auto‑submit your answers to the public leaderboard.

---

##  Project layout

```
├── agent.py          # LangGraph graph, tools & system prompt
├── retrieval.py      # Vector‑store indexing (task attachments + questions)
├── app.py            # Gradio web app + submission client
├── utils.py          # Common helpers, logging, constants
└── requirements.txt  # Pinned deps
```

---

##  Extending

* Add a new tool → append to `TOOLS` list and mention it once in `SYSTEM_PROMPT`.
* The prompt enforces: **one tool call per modality** + deterministic `give_answer`.
* Increase `recursion_limit` in `assistant` if you add more hops.

Pull‑requests and suggestions are welcome 🙂

---

##  Licence

MIT – 2025 @ GAIA Course
