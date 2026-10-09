"""eval.py — RAG v0.3 mini 评测集：一键跑分"""
import rag_v03_overlap as rag

# (问题, 答案必须包含的关键词)
TESTS = [
    ("Does this manual cover prestressed concrete?", ["不"]),
    ("What is the load factor for dead load under Usual serviceability conditions?", ["2.2"]),
    ("这本手册能不能用来设计预应力混凝土结构？", ["不"]),
    ("恒载在正常使用极限状态下的系数是多少？", ["2.2"]),
    ("What is the strength reduction factor for shear?", ["不知道"]),
    ("What do the subscripts U, N, and X stand for?", ["Usual", "Unusual", "Extreme"]),
    ("What does this manual say about crack control in hydraulic structures?", ["不知道"]),
]

def run():
    docs = rag.load_docs("corpus")
    print(f"已加载 {len(docs)} 个片段，正在预热向量…")
    rag.DOC_VECS = rag.embed(docs)
    passed = 0
    for i, (q, keywords) in enumerate(TESTS, 1):
        ans = rag.answer(q, docs)
        ok = all(k.lower() in ans.lower() for k in keywords)
        passed += 1 if ok else 0
        print(f"{'✅' if ok else '❌'} Q{i}: {q}\n   A: {ans[:100]}")
    print(f"\n总分：{passed}/{len(TESTS)}")

if __name__ == "__main__":
    run()
