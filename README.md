# customer-agent

基于 **LangGraph + RAG（检索增强生成）** 的智能客服系统（customer service agent / chatbot），支持产品手册等非结构化 PDF 文档的导入向量化与多轮问答。

核心能力：PDF 文档解析（MinerU）→ BGE-M3 向量嵌入 → Milvus 向量检索 + Neo4j 知识图谱混合检索 → RRF 融合 → BGE 重排序 → LLM 流式回答（SSE）。

> RAG · Retrieval-Augmented Generation · LangGraph Workflow · Vector Search · Knowledge Graph · Hybrid Retrieval · Reranking · FastAPI · Streaming Chatbot

## 架构

系统分为两条 LangGraph 工作流：

### 文档导入流程（`app/import_process/`）

```
PDF 上传 → MinerU 解析转 Markdown → 图片提取摘要 → 文档切块
         → BGE-M3 向量化 → Milvus 向量入库
```

### 查询问答流程（`app/query_process/`）

```
用户提问 → 查询改写 / HyDE 生成 → 商品实体识别确认
        → 向量检索 + 知识图谱检索 → RRF 融合 → BGE 重排序
        → LLM 生成答案 → SSE 流式输出
```

## 技术栈

| 模块 | 选型 |
|------|------|
| 工作流编排 | LangGraph |
| Web 框架 | FastAPI + Uvicorn |
| 向量数据库 | Milvus |
| 知识图谱 | Neo4j |
| 对话历史 | MongoDB |
| 对象存储 | MinIO（开发环境用 moto 内存模拟） |
|  Embedding / Reranker | BGE-M3 / BGE-Reranker |
| PDF 解析 | MinerU (magic-pdf) |
| LLM | OpenAI 兼容接口 |

## 目录结构

```
app/
├── clients/          # Milvus / MongoDB / Neo4j / MinIO 客户端封装
├── conf/             # 各组件配置
├── core/             # Prompt 加载、日志等基础设施
├── import_process/   # 文档导入工作流（节点 + API + 页面）
├── query_process/    # 问答查询工作流（节点 + API + 页面）
├── lm/               # LLM / Embedding / Reranker 调用封装
├── tool/             # 模型下载脚本
└── utils/            # 通用工具函数
prompts/              # 各节点 Prompt 模板
doc/                  # 产品手册 PDF 文档
lessons/              # 教学课件
test/                 # 单元测试与功能测试
```

## 快速启动

详见 [STARTUP.md](./STARTUP.md)。

简要步骤：

```bash
# 1. 启动中间件
docker compose up -d

# 2. 启动对象存储（moto，内存版）
.venv/bin/moto_server -p 9000 &

# 3. 启动文档导入服务（端口 8000）
.venv/bin/python -m app.import_process.api.file_import_service

# 4. 启动问答服务（端口 8001）
.venv/bin/python -m app.query_process.api.query_service
```

- 导入页面：http://127.0.0.1:8000/import.html
- 问答页面：http://127.0.0.1:8001/chat.html
- API 文档：http://127.0.0.1:8000/docs

## 环境要求

- Python >= 3.11
- Docker（用于 Milvus / MongoDB / Neo4j）
- GPU（推荐，用于 Embedding / Reranker 模型推理）
