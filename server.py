"""server.py -- 把 8/8 规范问答 agent (agent_v04) 包成 HTTP API.

放在 agent_v04.py 同一目录下运行（Windows PowerShell）：
    .\\.venv\\Scripts\\python.exe -m pip install fastapi uvicorn
    .\\.venv\\Scripts\\python.exe -m uvicorn server:app --port 8000

测试（另开一个 PowerShell）：
    Invoke-RestMethod -Uri http://127.0.0.1:8000/chat `
        -Method Post -ContentType "application/json" `
        -Body '{"message": "恒载在正常使用极限状态下的系数是多少？"}'
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from starlette.middleware.base import BaseHTTPMiddleware

import agent_v04 as A

app = FastAPI(title="Spec QA Agent", version="0.1.0")


class CharsetMiddleware(BaseHTTPMiddleware):
    """让 JSON 响应带上 charset=utf-8，避免 Windows PowerShell 5.1 按 Latin-1 解码。"""
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json") and "charset" not in ct:
            resp.headers["content-type"] = "application/json; charset=utf-8"
        return resp


app.add_middleware(CharsetMiddleware)


def _make_tracer():
    """Langfuse trace。keys 走环境变量；没设则关闭 trace，服务不受影响。"""
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        print("[langfuse] keys 未设置，trace 关闭（服务照常运行）", flush=True)
        return None
    # langfuse v4 两步：先显式初始化 client（自动注册），再建 handler（不传 key，自动找到唯一实例）
    Langfuse(
        public_key=pk,
        secret_key=sk,
        host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
    )
    return CallbackHandler()


_tracer = _make_tracer()


class HistoryItem(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryItem] = []
    debug: bool = False  # True 时返回完整消息链，用于看模型每一步的决策


class TraceStep(BaseModel):
    type: str
    content: str
    tool_calls: list[str]


class ChatResponse(BaseModel):
    answer: str
    tools_called: list[str]
    trace: list[TraceStep] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 和 eval_agent.py 完全相同的调用方式：system + 历史 + 当前问题
    messages = [("system", A.SYSTEM)]
    messages += [(h.role, h.content) for h in req.history]
    messages.append(("user", req.message))
    try:
        invoke_config = {"callbacks": [_tracer]} if _tracer else None
        result = A.app.invoke({"messages": messages}, config=invoke_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent 调用失败: {e}")
    msgs = result["messages"]
    answer = msgs[-1].content
    tools_called = [
        tc["name"]
        for m in msgs
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    # 服务端同步打印 trace：uvicorn 控制台直接看到模型决策链
    for m in msgs:
        tcs = [tc["name"] for tc in (getattr(m, "tool_calls", None) or [])]
        print(f"[trace] {type(m).__name__} | tools={tcs} | {str(m.content)[:150]}",
              flush=True)
    trace = None
    if req.debug:
        trace = [
            TraceStep(
                type=type(m).__name__,
                content=str(m.content)[:500],
                tool_calls=[tc["name"] for tc in (getattr(m, "tool_calls", None) or [])],
            )
            for m in msgs
        ]
    return ChatResponse(answer=answer, tools_called=tools_called, trace=trace)
