"""agent_v02.py — ReAct 循环：LLM + 计算器 tool"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，比如 '2+3*4'。遇到算术问题必须用这个工具，不要心算。"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # 教学演示用，生产别这么写
    except Exception as e:
        return f"计算出错：{e}"

tools = [calculator]
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
graph.add_edge("tool", "agent")  # 工具跑完，把结果拿回给 agent
app = graph.compile()

result = app.invoke({"messages": [("user", "12345*6789 等于多少？")]})
print(result["messages"][-1].content)
for m in result["messages"]:
    print(type(m).__name__, ":", str(m.content)[:80])
    if getattr(m, "tool_calls", None):
        print("   → 调了工具:", m.tool_calls)
