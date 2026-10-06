from openai import OpenAI
import os
import math
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed(texts):
    """把一串文本变成向量（text-embedding-3-small）"""
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def cosine(a, b):
    """余弦相似度：越接近 1，意思越像"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


if __name__ == "__main__":
    s1 = "rainy day"
    s2 = "windy day"
    s3 = "hot coffee"
    v1, v2, v3 = embed([s1, s2, s3])
    print(f"向量维度：{len(v1)}")
    print(f"s1 vs s2（同义）：{cosine(v1, v2):.3f}")
    print(f"s1 vs s3（无关）：{cosine(v1, v3):.3f}")
