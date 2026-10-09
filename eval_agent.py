"""eval_agent.py v2 - Agent 评测：工具链（确定性）+ LLM 裁判（语义）"""
import os

from dotenv import load_dotenv
from openai import OpenAI

import agent_v04 as A
from typesafe_sdk import TypeSafeClient, Noul



load_dotenv()
judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# (问题, 期望工具序列, 期望要点)
TESTS = [
    ("恒载在正常使用极限状态下的系数是多少？", ["spec_search"], "2.2"),
    ("123*456等于多少？", ["calculator"], "56088"),
    ("把恒载在正常使用极限状态下的系数乘以3，结果是多少？", ["spec_search", "calculator"], "答案给出系数2.2和计算结果6.6"),
    ("你好，请介绍一下你自己", [], "答案表明自己是结构工程助手"),
    ("这本手册对裂缝控制有什么要求？", ["spec_search"], "答案说明手册中没有裂缝控制的具体要求，诚实表示不知道或未列出即算正确"),
    ("Does this manual cover prestressed concrete?", ["spec_search"], "答案说明手册不涵盖预应力混凝土"),
    ("2.2 加 1.6 等于多少？", ["calculator"], "答案给出结果是3.8"),
    ("规范里有没有提到 2.2 这个数字？", ["spec_search"], "表达出肯定即为正确"),
]


JUDGE_SYSTEM = (
    "你是一个严格的问答评测员。根据期望要点判断模型回答是否正确，"
    "只回答“通过”或“不通过”，不要解释。数字和关键事实必须准确；"
    "诚实表示不知道也算正确；答非所问或编造事实算不通过。"
)

jev = TypeSafeClient()  # 自动读 TYPESAFE_API_KEY

def jev_judge(question, expected, answer):
    resp = jev.system_one(
        state={"question": question, "expected": expected, "model_answer": answer},
        questions={"correct": Noul(instructions="Is the model's answer correct according to the expected key points?")},
    )
    return resp.nouls["correct"].noul  # 比如 0.92


def is_subsequence(sub, seq):
    i = 0
    for x in seq:
        if i < len(sub) and x == sub[i]:
            i += 1
    return i == len(sub)


def run_one(question, expected_tools, expected):
    history = [("system", A.SYSTEM), ("user", question)]
    result = A.app.invoke({"messages": history})
    msgs = result["messages"]
    called = [
        tc["name"]
        for m in msgs
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    thoughts = [
        m.content
        for m in msgs[:-1]
        if type(m).__name__ == "AIMessage" and m.content
    ]
    ans = msgs[-1].content
    tools_ok = is_subsequence(expected_tools, called)
    judge_ok = jev_judge(question, expected, ans)
    return tools_ok and judge_ok, called, thoughts, ans


def run():
    passed = 0
    for i, (q, et, exp) in enumerate(TESTS, 1):
        ok, called, thoughts, ans = run_one(q, et, exp)
        if ok:
            passed += 1
        mark = "✅" if ok else "❌"
        print(mark + " Q" + str(i) + ": " + q + "\n   工具链: " + str(called))
        for t in thoughts:
            print("   thought: " + t[:200])
        print("   A: " + ans[:100])
    print("\n总分：" + str(passed) + "/" + str(len(TESTS)))


if __name__ == "__main__":
    run()
