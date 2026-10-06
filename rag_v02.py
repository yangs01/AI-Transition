from openai import OpenAI
import os
import math
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"


def load_docs(path):
    """按句子切分：每个句子一个片段，语义完整"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [s.strip() for s in re.split(r"(?<=[。？！])", text) if s.strip()]


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


DOC_VECS = None  # 文档向量只算一次，启动时预热


def retrieve(query, docs, top_k=3):
    """v0.2：签名和 v0.1 完全一样，内部换成向量检索。
    调用方 answer() 一行不用改——这就是面向接口编程。”
    """
    q_vec = embed([query])[0]
    scored = [(cosine(q_vec, dv), doc) for dv, doc in zip(DOC_VECS, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]


def answer(question, docs):
    fragments = retrieve(question, docs, top_k=3)
    context = "\n\n---\n\n".join(fragments) if fragments else "（没有找到相关片段）"
    prompt = ("请根据以下资料回答问题。允许把问题中的说法与资料里的同义表述对应起来，"
          "但不要编造资料中没有的事实；如果资料确实没有相关内容，就说不知道。\n\n"
          f"资料：\n{context}\n\n问题：{question}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    docs = load_docs("spec.txt")
    print(f"已加载 {len(docs)} 个片段，正在预热向量…")
    DOC_VECS = embed(docs)  # 只算一次：每问一次就重算=烧钱
    while True:
        q = input("\n请输入问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        print("\n回答：\n" + answer(q, docs))