import os, openai, pandas as pd, numpy as np
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from retrieval import get_query_engine
from utils import LLM_MODEL, get_logger
from smolagents import DuckDuckGoSearchTool
import requests
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_community.document_loaders import WikipediaLoader
# from langgraph import StateGraph, ToolNode, START, tools_condition, MessagesState
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode




log = get_logger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------- tools ---------------------------------
@tool
def query_tool(question: str) -> str:
    """Search GAIA docs and return relevant chunks."""
    return get_query_engine().query(question).response.strip()

@tool
def python_tool(code: str) -> str:
    """Run a Python snippet; result in `result`."""
    scope = {"pd": pd, "np": np}
    try:
        log.info("Executing Python code: %s", code)
        exec(code, scope)
        return str(scope.get("result", ""))
    except Exception as e:
        return f"ERR: {e}"
    






@tool
def wiki_page(title_or_query: str) -> str:
    """
    Fetch the English Wikipedia page whose title best matches the query
    and return the first 2 kB of clean text.
    """
    log.info("wiki_page called with: %s", title_or_query)
    docs = WikipediaLoader(query=title_or_query, load_max_docs=5, doc_content_chars_max=20000).load()
    if not docs:
        return "Page not found."
    return docs[0].page_content[:] if docs else "No content found."

TOOLS = [query_tool, python_tool, wiki_page]

# ----------------------- LLM -----------------------------------
SYSTEM_PROMPT = (
    "You are an academic QA agent. "
    "If the user question requrires the web search, "
    "ALWAYS call the tool named 'wiki_page' first. "
    "Use python_tool for any arithmetic (countrimg, summarizing etc. You can use pandas and numpy)."
    "After calling wiki_page, if the answer requires doing calculations, load the page text into python_tool . "
    "If you retrieve any info from wiki, provide a summary of the page content in your response. "
)

def retriever(state: MessagesState):
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    # state["messages"][0] is the HumanMessage that came from START
    return {"messages": [sys_msg] + state["messages"]}




def build_graph():
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0).bind_tools(TOOLS)

    def assistant(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "retriever")        # ➊
    builder.add_edge("retriever", "assistant")  # ➋
    builder.add_conditional_edges(              # ➌
        "assistant", tools_condition
    )
    builder.add_edge("tools", "assistant")      # ➍ (loop edge)

    return builder.compile()

graph = build_graph()
result = graph.invoke({"messages":[HumanMessage(content=
    "How many studio albums did Mercedes Sosa release between 2000 and 2009?")]})
print(result["messages"][-1].content)   