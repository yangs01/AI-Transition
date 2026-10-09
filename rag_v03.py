from openai import OpenAI
import os
import math
from dotenv import load_dotenv
import re
import glob

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"

def split_sentences(text):
    # 英文按句切：句号/问号/感叹号 + 空格 + 大写字母/数字
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(])', text)
    # 合并短碎片（比如 "3.3.2." 这种编号）
    chunks, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) < 60:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

def load_docs(corpus_dir="corpus"):
    # 读 corpus/ 下所有 txt，切好后拼成一个列表
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.extend(split_sentences(text))
    return docs


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
    docs = load_docs("corpus")
    for i, d in enumerate(docs):
        print(i, len(d), d[:80].replace("\n", " "))
    if not docs:
        raise SystemExit("corpus/ 下没有读到任何 txt，请检查文件夹路径")
    print(f"已加载 {len(docs)} 个片段，正在预热向量…")
    DOC_VECS = embed(docs)  # 只算一次：每问一次就重算=烧钱
    while True:
        q = input("\n请输入问题（输入 q 退出）：").strip()
        if q.lower() == "q":
            break
        print("\n回答：\n" + answer(q, docs))