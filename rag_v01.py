import re
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_docs(path):
    """按句子切分：每个句子一个片段，语义完整"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in re.split(r"(?<=[。？！])", text) if s.strip()]


def retrieve(query, docs, top_k=3):
    """v0.1 关键词检索：按单字统计命中数，取最高的 top_k 个。
    注意：这个实现很糙，中文同义词、换说法就抓瞎——
    这正是 v0.2 要换成向量检索的原因。"""
    keywords = [ch for ch in query
                if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()]
    scored = [(sum(doc.count(k) for k in keywords), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


def answer(question, docs):
    fragments = retrieve(question, docs, top_k=3)
    context = "\n\n---\n\n".join(fragments) if fragments else "（没有找到相关片段）"
    prompt = ("请只根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n"
              f"资料：\n{context}\n\n问题：{question}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    docs = load_docs("spec.txt")
    print(f"已加载 {len(docs)} 个片段")
    while True:
        q = input("\n请输入问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        print("\n回答：\n" + answer(q, docs))
