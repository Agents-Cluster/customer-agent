# 最小 RAG Demo —— 面试讲解版

单文件、零依赖、端到端跑通：
```bash
python demo/minimal_rag.py
```

## 对着代码讲方案（五层一一对应）

| 方案里的描述 | 代码里的类 | 面试要点 |
|---|---|---|
| 结构化数据优化（MinerU） | `CORPUS_MD` | demo 直接喂结构化 Markdown，真实链路是 PDF→MinerU→这份 md |
| 语义自治切片 | `SemanticChunker` | 标题栈维护 breadcrumb、按句切、15% overlap、动态块长 |
| 混合向量检索 | `DenseRetriever` + `SparseRetriever` | Dense=hash向量余弦（生产换BGE-M3）；Sparse=标准 BM25 |
| 多路召回 | `HyDERetriever` | 三路并行：dense / BM25 / HyDE |
| 结果融合 | `RRFFusion` | RRF = Σ 1/(k+rank)，k=60，不用归一化、抗分数劫持 |
| 重排序裁决 | `Reranker` | breadcrumb命中 + query term 覆盖率 + 词项密度（生产换 bge-reranker） |

## 三个 query 覆盖三种典型场景

1. **"为什么切片不能用固定长度"** —— 语义类，dense/HyDE 把泛化意思相近的 chunk 捞上来
2. **"BM25 的 k1 和 b 参数"** —— 术语类，sparse(BM25) 直接把 #5 顶到第一
3. **"多路召回结果怎么融合"** —— 概念类，需要同时命中"多路召回"和"RRF融合"两个章节

## 生产环境替换清单（面试官追问时主动说）

```
DenseRetriever   →  bge-m3 / text-embedding-3-large
SparseRetriever  →  Milvus BM25 插件 / Elasticsearch
HyDE             →  真调一次 LLM：prompt="请用 200 字回答：{query}"，拿回答去 embed
Reranker         →  bge-reranker-v2-m3（cross-encoder，精度比召回高一个量级）
Chunker          →  直接吃 MinerU 输出的 .md，不用自己解析标题
存储             →  Dense 向量入 Milvus，Sparse 索引入 ES，元数据入 MongoDB
```

## 面试高频追问 & 标准答法

**Q: 为什么要多路召回，单路向量不够吗？**
A: 向量检索对"同义/隐含语义"强，但对精确术语、编号、人名容易漏；BM25 反过来。HyDE 解决"query 太短、语义稀疏"的问题。三路并行 + RRF，召回率比单路高 15~30%。

**Q: RRF 为什么不用加权求和？**
A: 各路分数量纲不一样（cosine 0~1，BM25 可能几十），归一化没做好就会"分数劫持"。RRF 只看排名不看分数，简单且鲁棒，工程上没人想动它。

**Q: Rerank 和召回的本质区别？**
A: 召回是**双塔**（query 和 doc 分别编码，快但粗），Rerank 是**交叉编码**（query+doc 拼一起过模型，慢但准）。所以流程一定是"粗排求全 → 精排求准"。
