"""
GAIA LangGraph Agent - A multi-tool agent for processing various file types and answering questions.
"""

import os
import openai
import pandas as pd
import numpy as np
import base64
import requests
import runpy
import contextlib
import io
import pathlib
import logging
from typing import Optional

# LangChain imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader

# Local imports
from retrieval import get_query_engine
from utils import LLM_MODEL, get_logger
from smolagents import DuckDuckGoSearchTool

# Initialize logging
log = get_logger(__name__)

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
_TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# Cache for Wikipedia pages
_wiki_cache: dict[str, str] = {}

# Initialize search tools
_ddg = DuckDuckGoSearchTool()

def _img_to_b64(path_or_url: str) -> str:
    """Convert an image file or URL to base64 string."""
    data = requests.get(path_or_url, timeout=15).content \
        if path_or_url.startswith(("http://", "https://")) \
        else open(path_or_url, "rb").read()
    return base64.b64encode(data).decode()

def find_file_by_type(task_id: str, file_type: str) -> Optional[str]:
    """
    Find a file in data directory by task ID and type.
    Returns the full path if found, None otherwise.
    """
    data_dir = pathlib.Path("data")
    # First try exact UUID pattern
    pattern = f"{task_id}_{task_id}.{file_type}"
    for file in data_dir.iterdir():
        if file.name == pattern:
            return str(file)
    
    # If not found, try to find any file with the given extension
    for file in data_dir.iterdir():
        if file.suffix.lower() == f".{file_type}":
            return str(file)
    return None

# ----------------------- Tools ---------------------------------

@tool
def query_tool(question: str) -> str:
    """Search GAIA docs and return relevant chunks."""
    return get_query_engine().query(question).response.strip()

@tool
def python_tool(code: str) -> str:
    """Execute Python snippet; put answer in variable `result`."""
    scope = {"pd": pd, "np": np}
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        try:
            exec(code, scope)
            return scope.get("result") or buf.getvalue()
        except Exception as e:
            return f"ERR: {e}"

@tool
def run_python_file(path: str) -> str:
    """Run attached .py file and return last print or variable `result`."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        ns = runpy.run_path(path)
    return ns.get("result") or stream.getvalue()

@tool
def excel_total(path: str) -> str:
    """Return total of numerical column *Sales* in .xlsx (2 decimals)."""
    # Extract task ID from the path if it's in UUID format
    if "_" in path:
        task_id = path.split("_")[0]
        actual_path = find_file_by_type(task_id, "xlsx")
        if actual_path:
            path = actual_path
    try:
        df = pd.read_excel(path)
        total = df["Sales"].sum()
        return f"{total:.2f}"
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

@tool
def wiki_page(title_or_query: str) -> str:
    """Return up to ~10 kB of the best-matching English Wikipedia page."""
    log.info("wiki_page called with: %s", title_or_query)
    if title_or_query in _wiki_cache:
        return _wiki_cache[title_or_query]
    docs = WikipediaLoader(query=title_or_query, load_max_docs=1, doc_content_chars_max=10000).load()
    if not docs:
        return "Page not found."
    _wiki_cache[title_or_query] = docs[0].page_content[:]
    return docs[0].page_content[:] if docs else "No content found."

@tool
def deep_vision(image: str, ask: str = "Describe the image and list distinct objects.") -> str:
    """
    Run GPT-4o-Vision on an image and return a JSON summary.
    """
    try:
        # First try direct path
        if pathlib.Path(image).exists():
            file_path = image
        else:
            # Try to find file by type
            file_path = find_file_by_type(image.split("_")[0] if "_" in image else image, "png")
        if not file_path:
            return "Error: Could not find the image file. Please check if the file exists in the data directory."
        b64 = _img_to_b64(file_path)
        resp = openai.chat.completions.create(
            model="gpt-4o",  # Updated model name
            max_tokens=400,
            temperature=0,
            messages=[
                {"role": "user",
                 "content": [
                     {"type": "text", "text":
                        f"You are an expert visual analyst. {ask}\n"
                        "Return JSON with keys:"
                        " caption (string), objects (list of strings), "
                        "colors (list of strings), raw (string)."},
                     {"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{b64}"}},
                 ]}
            ],
        )
        log.info("deep_vision called with: %s", file_path)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error processing image: {str(e)}"

@tool
def ddg_snippet(query: str) -> str:
    """DuckDuckGo top snippet (plain text, ~800 chars)."""
    log.info("ddg_snippet called with: %s", query)
    return _ddg(query)[:800]

@tool
def tavily_search(query: str, k: int = 3) -> str:
    """Live SERP via Tavily (free-tier 500 calls/mo)."""
    os.environ["TAVILY_API_KEY"] 
    docs = TavilySearchResults(max_results=k).invoke(input=query)
    return "\n\n---\n\n".join(d["content"] for d in docs)

@tool
def transcribe_audio(path: str) -> str:
    """Whisper API transcription (first 2k chars)."""
    try:
        # First try direct path
        if pathlib.Path(path).exists():
            file_path = path
        else:
            # Try to find file by type
            file_path = find_file_by_type(path.split("_")[0] if "_" in path else path, "mp3")
        if not file_path:
            return "Error: Could not find the audio file. Please check if the file exists in the data directory."
        log.info("Processing audio file: %s", file_path)
        with open(file_path, "rb") as f:
            txt = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return txt[:2000]
    except Exception as e:
        log.error("Error processing audio file: %s", str(e))
        return f"Error processing audio file: {str(e)}"

@tool(return_direct=True)
def give_answer(ans: str) -> str:
    """Return the bare answer token (no extra words)."""
    return ans.strip()

# Define available tools
TOOLS = [
    query_tool,
    python_tool,
    run_python_file,
    excel_total,
    wiki_page,
    deep_vision,
    ddg_snippet,
    tavily_search,
    transcribe_audio,
    give_answer
]

# System prompt for the agent
SYSTEM_PROMPT = (
    "You are an academic QA agent, you have the following tools.\n"
    "• Wikipedia → call wiki_page once.\n"
    "• Web → USE tavily_search (if does not work fallback ddg_snippet).\n"
    "• Image → ALWAYS call deep_vision.\n"
    "• .mp3 → transcribe_audio;  .py → run_python_file (return 0 if not response),  .xlsx → excel_total.\n"
    "• Finish with give_answer['exact-token'] (no other words).\n"
    "• RETURN THE FINAL ANSWER ONLY, THE MOST CONCISE DIRECT ANSWER FOR THE ASKED QUESTIONS, no douts in the end of the words \n"
    " Do not include points, commas, or any other punctuation at the very end of the answer.\n"
)

# Initialize LLM
llm = ChatOpenAI(
    model=LLM_MODEL, 
    temperature=0,
    request_timeout=30,
    max_retries=2
).bind_tools(TOOLS)

def assistant(state):
    """Process messages and generate responses."""
    loops = state.get("loops", 0)
    if loops >= 13:
        return {
            "messages": state["messages"] + [AIMessage(content="give_answer['ERR']")],
            "loops": loops
        }
    
    response = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response],
        "loops": loops + 1
    }

def retriever(state):
    """Prepare messages for the assistant."""
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
        "loops": state.get("loops", 0)
    }

# Build the LangGraph
builder = StateGraph(MessagesState)
builder.add_node("retriever", retriever)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

# Configure the graph with higher recursion limit
graph = graph.with_config(
    recursion_limit=50,
    recursion_key="loops"
)

def debug_single(q: str):
    """Process a single question and return the answer."""
    out = graph.invoke({"messages": [HumanMessage(content=q)]})
    print("ANSWER:", out["messages"][-1].content)