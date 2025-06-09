import requests
from langchain_core.messages import HumanMessage
from utils import QUESTIONS_URL, get_logger
from agent import react_graph        # <- already built & imported tools/index

log = get_logger(__name__)
log.debug("DEBUG mode active!") 
def fetch_one_question() -> dict:
    task = requests.get(QUESTIONS_URL, timeout=30).json()[0]   # first task
    return {"id": task["task_id"], "q": task["question"]}

if __name__ == "__main__":
    sample = fetch_one_question()
    log.info("Testing on task %s: %s", sample["id"], sample["q"])

    state = react_graph.invoke(
        {"messages": [HumanMessage(content=sample["q"])]}
    )
    final_answer = state["messages"][-1].content
    print(f"\nAnswer → {final_answer}")
