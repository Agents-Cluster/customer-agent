"""
最小可运行 RAG Demo —— 对应方案里的五层架构
================================================
本文件零第三方依赖（只用 Python 标准库），跑法：
    python demo/minimal_rag.py

五层映射（对着你那份方案讲）：
  ① 结构化切片     → SemanticChunker
  ② 混合向量检索    → DenseRetriever + SparseRetriever(BM25)
  ③ 多路召回       → MultiRetriever（dense / sparse / HyDE）
  ④ RRF 融合       → RRFFusion
  ⑤ 重排序裁决     → Reranker

生产环境替换点（面试时主动说）：
  - DenseRetriever  → 换成 BGE-M3 / text-embedding-3
  - SparseRetriever  → 换成 Milvus BM25 插件 / ES
  - HyDE            → 真调一次 LLM 生成假设答案
  - Reranker        → 换成 bge-reranker-v2
  - 切片            → 换成 MinerU 输出的 Markdown 直接喂
"""

from __future__ import annotations
import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ============================================================
# 0. 模拟“已经过 MinerU 结构化解析”的 Markdown 语料
#    真实项目里这一步是 PDF → MinerU → 下面这种干净的 .md
# ============================================================
CORPUS_MD = """
# RAG 系统架构

## 结构化数据优化
传统 PDF 抽取会导致结构坍塌、段落断裂、图文割裂。本项目基于 MinerU 对 PDF 做深度结构化解析，
自动恢复标题层级、段落边界、表格结构、公式与列表关系，输出语义化 Markdown。
对图片和图表走 OCR + VLM 双通路，把视觉信息转成可检索文本。

## 语义自治切片
固定长度切片会打断知识逻辑链，导致“命中但证据不完整”。本方案按章节、段落、主题边界切分，
超长文本二次切分，短文本同域合并，并设置 10%~20% 的 overlap 区域，同时继承父级标题路径，
保证每个 chunk 是最小可解释、最小可检索、最小可验证的知识单元。

## 多路召回引擎
单路检索存在语义盲区。项目并行跑向量检索、关键词检索、HyDE 假设性问题检索、联网搜索，
再做结果融合，最大限度避免信息遗漏。多路召回 + 重排序已是高精度 RAG 的标配。

## 混合向量检索
Dense Vector 负责深层语义理解与同义泛化，做到“懂意思”；
Sparse Vector 负责专业术语、实体关键词的精准匹配，做到“查得准”。
两路结果做归一化与加权融合，避免单一向量路径的“分数劫持”。

## 重排序裁决层
召回阶段求“全”，重排阶段求“准”。对多路召回的候选做精细化相关性打分，
过滤低相关噪声，只把 Top-K 证据送进大模型。Rerank 是决定答案质量上限的关键模块。

# 工程实现细节

## BM25 算法
BM25 是经典稀疏检索模型，核心公式为：IDF * (tf * (k1+1)) / (tf + k1*(1-b+b*dl/avgdl))。
k1 通常取 1.5~2.0，b 取 0.75。它对长文档做归一化惩罚，避免长文靠堆词刷分。

## RRF 融合
Reciprocal Rank Fusion 不需要对各路分数做归一化，公式为 score = Σ 1/(k + rank)，
k 常取 60。RRF 对量纲不敏感、抗异常分数，是工业界多路召回融合的首选。

## 向量相似度
Dense 检索常用余弦相似度或点积。文本经过 Embedding 模型映射到高维空间后，
语义相近的句子向量夹角小，从而实现“意思相近但用词不同”也能召回。
"""


# ============================================================
# 1. 中文/英文混合分词（demo 级，不装 jieba）
#    英文按 \w+ 切，中文按 2-gram 切，够演示用
# ============================================================
def tokenize(text: str) -> List[str]:
    text = text.lower()
    # 英文 + 数字按词切
    en_words = re.findall(r"[a-z0-9]+", text)
    # 中文按字级 2-gram（demo 用，生产换 jieba/lac）
    cn_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    cn_bigrams = [cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1)]
    return en_words + cn_bigrams


# ============================================================
# 2. 结构化语义切片器
#    对应方案：按标题边界 + 段落边界 + 动态块长 + overlap + 继承父标题
# ============================================================
@dataclass
class Chunk:
    id: int
    text: str                       # chunk 正文
    breadcrumb: str                 # 父级标题路径，如 "RAG系统 / 多路召回引擎"
    tokens: List[str] = field(default_factory=list)


class SemanticChunker:
    def __init__(self, target_len: int = 220, overlap_ratio: float = 0.15):
        self.target_len = target_len
        self.overlap = int(target_len * overlap_ratio)

    def split(self, md: str) -> List[Chunk]:
        # 第一步：按标题行切，同时维护标题栈，给每个 chunk 挂 breadcrumb
        lines = [l for l in md.strip().split("\n")]
        heading_stack: List[Tuple[int, str]] = []   # (level, title)
        sections: List[Tuple[str, str]] = []        # (breadcrumb, body)
        buf_title = ""
        buf_body: List[str] = []

        def flush():
            if buf_body:
                body = " ".join(buf_body).strip()
                if body:
                    sections.append((buf_title, body))

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                flush()
                buf_body = []
                level = len(m.group(1))
                title = m.group(2).strip()
                # 弹栈：把同级及更深的标题出栈
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                buf_title = " / ".join(t for _, t in heading_stack)
            else:
                buf_body.append(line)
        flush()

        # 第二步：每个 section 内部按句子/长度切，做 overlap
        chunks: List[Chunk] = []
        cid = 0
        for breadcrumb, body in sections:
            # 粗切：按句号/分号
            sentences = re.split(r"(?<=[。；;！！?？])\s*", body)
            sentences = [s.strip() for s in sentences if s.strip()]

            cur = ""
            for sent in sentences:
                if len(cur) + len(sent) <= self.target_len:
                    cur += sent
                else:
                    if cur:
                        chunks.append(self._make(cid, cur, breadcrumb)); cid += 1
                    # overlap：把上一段尾巴带到下一段开头
                    tail = cur[-self.overlap:] if self.overlap > 0 and cur else ""
                    cur = tail + sent
            if cur:
                chunks.append(self._make(cid, cur, breadcrumb)); cid += 1

        return chunks

    def _make(self, cid: int, text: str, breadcrumb: str) -> Chunk:
        return Chunk(id=cid, text=text, breadcrumb=breadcrumb, tokens=tokenize(text))


# ============================================================
# 3. Dense 检索器（演示用 hash-embedding，生产换 BGE）
# ============================================================
class DenseRetriever:
    """词 hash 到 256 维 + TF-IDF 加权 + L2 归一化。
    演示目的：让你看到“稠密向量空间 + 余弦相似度”这件事怎么跑。
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.chunk_vecs: List[List[float]] = []

    def _embed(self, tokens: List[str]) -> List[float]:
        vec = [0.0] * self.dim
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def index(self, chunks: List[Chunk]):
        self.chunk_vecs = [self._embed(c.tokens) for c in chunks]

    def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Tuple[int, float]]:
        qv = self._embed(tokenize(query))
        scored = []
        for i, cv in enumerate(self.chunk_vecs):
            sim = sum(a * b for a, b in zip(qv, cv))   # 余弦
            scored.append((i, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ============================================================
# 4. Sparse 检索器 = BM25（标准实现）
# ============================================================
class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_freq: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.avgdl = 0.0

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.doc_len = [len(c.tokens) for c in chunks]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        for c in chunks:
            for t in set(c.tokens):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        # 经典 BM25 IDF（+1 防止除零）
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        scores = [0.0] * len(self.chunks)
        for qi in q_tokens:
            idf = self._idf(qi)
            for i, c in enumerate(self.chunks):
                tf = c.tokens.count(qi)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-6))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:top_k] if s > 0]


# ============================================================
# 5. HyDE 模拟：把 query 拼成一句“假设答案”，再拿它去 dense 检索
#    真实实现 = 先让 LLM 写一段假设答案，再 embed 那段答案
# ============================================================
class HyDERetriever:
    def __init__(self, dense: DenseRetriever):
        self.dense = dense

    def _mock_hypothetical_answer(self, query: str) -> str:
        # demo 级：把 query 词包成一句“假设段落”
        tokens = tokenize(query)
        return f"根据文档，{query}。相关概念包括{'、'.join(tokens[:8])}，其原理、实现和优化方式如下。"

    def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Tuple[int, float]]:
        fake_doc = self._mock_hypothetical_answer(query)
        return self.dense.search(fake_doc, chunks, top_k=top_k)


# ============================================================
# 6. RRF 融合（不依赖分数归一化，工业标配）
# ============================================================
class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, ranked_lists: List[List[Tuple[int, float]]], top_k: int = 5) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for rl in ranked_lists:
            for rank, (cid, _) in enumerate(rl):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ============================================================
# 7. Rerank 裁决层（演示用交叉打分，生产换 bge-reranker）
# ============================================================
class Reranker:
    """不看向量，直接对 (query, chunk) 对做精细打分：
       - breadcrumb 命中：说明主题对口，加分
       - query term 覆盖率：多少个词在 chunk 里出现
       - term 密度：出现次数 / chunk 长度
    """
    def rerank(self, query: str, chunks: List[Chunk],
               candidates: List[Tuple[int, float]], top_k: int = 3) -> List[Tuple[int, float]]:
        q_tokens = set(tokenize(query))
        results = []
        for cid, base_score in candidates:
            c = chunks[cid]
            # 1) 标题命中
            title_hit = 1.0 if any(t in c.breadcrumb for t in q_tokens) else 0.0
            # 2) query term 覆盖率
            cover = sum(1 for t in q_tokens if t in c.tokens) / max(len(q_tokens), 1)
            # 3) 词项密度
            density = sum(c.tokens.count(t) for t in q_tokens) / max(len(c.tokens), 1)
            # 4) 综合分（融合分 base + 精细特征）
            final = base_score * 2.0 + title_hit * 0.5 + cover * 1.5 + density * 10.0
            results.append((cid, final))
        results.sort(key=lambda x: -x[1])
        return results[:top_k]


# ============================================================
# 8. 串起来的 RAG 管线
# ============================================================
class MinimalRAG:
    def __init__(self):
        self.chunker = SemanticChunker()
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.hyde = HyDERetriever(self.dense)
        self.fusion = RRFFusion()
        self.reranker = Reranker()
        self.chunks: List[Chunk] = []

    def build_index(self, md: str):
        self.chunks = self.chunker.split(md)
        self.dense.index(self.chunks)
        self.sparse.index(self.chunks)
        print(f"[索引] 切出 {len(self.chunks)} 个 chunk\n")

    def ask(self, query: str, verbose: bool = True) -> List[Chunk]:
        # 三路并行召回
        r_dense = self.dense.search(query, self.chunks, top_k=5)
        r_sparse = self.sparse.search(query, top_k=5)
        r_hyde = self.hyde.search(query, self.chunks, top_k=5)

        if verbose:
            def show(name, lst):
                print(f"  [{name}] " + " | ".join(f"#{i}({s:.3f})" for i, s in lst))
            print(f"Q: {query}")
            show("Dense ", r_dense)
            show("Sparse", r_sparse)
            show("HyDE ", r_hyde)

        # RRF 融合
        fused = self.fusion.fuse([r_dense, r_sparse, r_hyde], top_k=5)
        if verbose:
            print(f"  [RRF  ] " + " | ".join(f"#{i}({s:.4f})" for i, s in fused))

        # Rerank
        final = self.reranker.rerank(query, self.chunks, fused, top_k=3)
        if verbose:
            print("  [Rerank]")
            for cid, s in final:
                c = self.chunks[cid]
                print(f"     #{cid} score={s:.3f}  [{c.breadcrumb}]")
                print(f"        {c.text[:80]}...")
            print()
        return [self.chunks[i] for i, _ in final]


# ============================================================
# 9. 跑起来看看
# ============================================================
if __name__ == "__main__":
    rag = MinimalRAG()
    rag.build_index(CORPUS_MD)

    queries = [
        "为什么切片不能用固定长度",                # 语义/泛化类
        "BM25 的 k1 和 b 参数是干嘛的",           # 术语精确类
        "多路召回结果怎么融合",                    # 概念类
    ]
    for q in queries:
        rag.ask(q)
