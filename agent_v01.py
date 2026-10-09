"""agent_v01.py — 最小 LangGraph：一个节点调一次 LLM"""
import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class State(TypedDict):
    question: str
    answer: str

def call_llm(state: State):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": state["question"]}],
    )
    return {"answer": resp.choices[0].message.content}

graph = StateGraph(State)
graph.add_node("llm", call_llm)   # 工序：调 LLM
graph.set_entry_point("llm")       # 从这里开工
graph.add_edge("llm", END)         # 干完收工
app = graph.compile()

result = app.invoke({"question": "用一句话解释什么是结构找形", "answer": ""})
print(result["answer"])
