"""agent_v03.py — 双工具 Agent：计算器 + 规范检索(RAG)"""
import os
from typing import TypedDict, Annotated
from unittest import result
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import rag_v03_overlap as rag

load_dotenv()

_docs = rag.load_docs("corpus")
rag.DOC_VECS = rag.embed(_docs)
print(f"RAG 就绪：{len(_docs)} 个片段")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，比如 '2+3*4'。遇到算术问题必须用这个工具，不要心算。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def spec_search(query: str) -> str:
    """在 USACE EM 1110-2-2104《水工结构钢筋混凝土强度设计》规范中检索相关条文。
    当问题涉及规范条文、荷载系数、设计要求时，必须用这个工具查，不要凭记忆回答。"""
    chunks = rag.retrieve(query, _docs, top_k=3)
    return "\n\n".join(f"[条文{i+1}] {c}" for i, c in enumerate(chunks))

tools = [calculator, spec_search]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def should_continue(state: State):
    last = state["messages"][-1]
    return "tool" if last.tool_calls else END

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tool", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
graph.add_edge("tool", "agent")
app = graph.compile()

SYSTEM = ("你是一个结构工程助手。遇到算术问题用 calculator；"
          "遇到规范条文、荷载系数、设计要求问题用 spec_search 查规范；"
          "工具查不到就说不知道，不要编。")

def ask(q):
    result = app.invoke({"messages": [("system", SYSTEM), ("user", q)]})
    for m in result["messages"]:
        print(type(m).__name__, ":", str(m.content)[:80])
        if getattr(m, "tool_calls", None):
            print("   → 调了工具:", m.tool_calls)
    print("Q:", q)
    print("A:", result["messages"][-1].content, "\n")


ask("恒载在正常使用极限状态下的系数是多少？")
ask("把恒载在正常使用极限状态下的系数乘以 3，结果是多少？")


