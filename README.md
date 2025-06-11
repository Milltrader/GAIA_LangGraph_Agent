# GAIA LangGraph Agent

A LangGraph-based agent that can process various types of files (audio, images, Excel) and answer questions about them using OpenAI's GPT models.

## Features

- Audio transcription using Whisper API
- Image analysis using GPT-4 Vision
- Excel file processing
- Web search capabilities (DuckDuckGo and Tavily)
- Wikipedia integration
- Python code execution

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key  # Optional, for web search
```

## Usage

The agent can be used in two ways:

1. **Direct API**: Import and use the `debug_single` function:
```python
from agent import debug_single

question = "Your question here"
answer = debug_single(question)
```

2. **Gradio Interface**: Run the web interface:
```bash
python app.py
```

## File Structure

- `agent.py`: Core agent implementation with tools and LangGraph setup
- `retrieval.py`: Vector store and document retrieval functionality
- `utils.py`: Utility functions and constants
- `app.py`: Gradio web interface

## Supported File Types

- Audio: MP3 files (transcription)
- Images: PNG files (visual analysis)
- Documents: Excel files (data processing)
- Python: PY files (code execution)
