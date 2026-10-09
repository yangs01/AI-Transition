# Week 2 启动包：RAG 项目开工（2026-10-05 ~ 2026-10-11）

**本周目标**：先还 Week 1 的债，再把关键词检索升级为向量检索，落地真正的 RAG 原型（v0.2）——作品集项目 1 正式开工。

> coaching 时间：周一 10/5 21:08（用户主动要求开始）。本周概念：Embedding（动手）+ Vector Database（理解它解决什么问题）。

---

## Checklist（含验收标准）

### 已完成（10/5）
- [x] Week 2 coaching session：复盘 Week 1 + 本周任务布置
- [x] 李宏毅《生成式 AI 导论 2024》第 6/7/8/9 讲看完（预训练→指令微调→RLHF→AI Agent）
- [x] RLHF 吃透：三步走（人类排名→训练 Reward Model→强化调整），"转正答辩"类比
- [x] model vs system 区分吃透：hello.py 只是调一次模型的脚本；Agent 是把同一模型放进"思考→行动→观察"循环（+工具）的系统
- [x] Agent 面试话术准备好（总-分-总 + "什么时候不用 Agent" 的 caveat）
- [x] `summary.py`：读取本地 txt 并让模型总结，跑通
- [x] 独立排查：API Key 过期 → 更新 key 解决（Q3"作废重建"顺手完成）
- [x] `summary.py` + `spec.txt` push 到 GitHub（`yangs01/AI-Transition`）
- [x] `Note/week-1.md` 笔记文件已建

### 待完成
- [ ] **v0.1**：命令行"规范问答小助手"（关键词检索版）——读 txt → `retrieve(query, docs, top_k)` 关键词检索 → 带片段问模型 → 打印答案；push 到 GitHub
  - 验收：能演示 + 有 commit；`retrieve()` 签名固定，给 Week 2 向量检索留好替换口子
- [ ] **Embedding 实验**：调 `text-embedding-3-small`，规范 txt 做 Chunking（文本切分）→ 每段转向量 → 验证语义相近的句子 Cosine Similarity（余弦相似度）更高
  - 验收：实验脚本 + 观察笔记记进 `Note/week-2.md`
- [ ] **v0.2 向量检索升级**：`retrieve()` 签名不变，内部换成向量检索；同一问题对比关键词版 vs 向量版效果
  - 验收：对比记录 + commit。做完 = 真正的 RAG 原型，作品集项目 1 开工
- [ ] **13 个概念三件套抽查**（Week 1 延续）：不看笔记说出英文全称 + 中文 + 一句话解释；RAG 开工最低限度先保 Token / Embedding / Context Window / Prompt 四个

---

## 本周节奏建议

| 天 | 干什么 |
|---|---|
| 周一 10/5 | coaching + 李宏毅 6~9 讲 + summary.py（已完成） |
| 周二 10/6 | v0.1 关键词检索版 + push（还债收尾） |
| 周三 10/7 | Embedding + Chunking 实验 |
| 周四~六 | v0.2 向量检索升级 + 效果对比 |
| 周日 10/11 | 复盘：更新本文件 + 准备下周问题 |

总量约 15 小时，保底 12 小时算达标。

---

## 已沉淀（可讲的故事 / 面试素材）

1. **RLHF 三步走**：人类排名 → Reward Model 学会"甲方口味" → 模型朝高分方向调整。"没有 RLHF，ChatGPT 只是个知识渊博但不会好好说话的书呆子。"
2. **model vs system**：模型还是那个文字预测模型，变的是外面的脚手架（循环+工具+记忆）。
3. **工程判断力话术**："知道什么时候不用 Agent，和知道怎么做一样重要。"
