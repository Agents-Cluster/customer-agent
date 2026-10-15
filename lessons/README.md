# 掌柜智库（RAG Agent）课件包

> 面向前端工程师转 Agent 方向面试的后端/Agent 工程课件，基于 `1.笔记/` 21 份项目文档整理，代码路径已对照项目根目录真实代码核实。

## 课程地图

| 编号 | 课件 | 核心问题 | 对应的笔记 |
|---|---|---|---|
| 01 | [lesson-01-rag-architecture.html](lesson-01-rag-architecture.html) | RAG 是什么？掌柜智库的两条链路如何组织？ | 01、02、03、04、07 |
| 02 | [lesson-02-langgraph-workflow.html](lesson-02-langgraph-workflow.html) | LangGraph 的状态、节点、路由如何工作？ | 04、05、07、08 |
| 03 | [lesson-03-import-document-processing.html](lesson-03-import-document-processing.html) | 导入链路：PDF→MD→图片→切片为什么分多步？ | 05 节点1-4、06 |
| 04 | [lesson-04-vectorization-milvus.html](lesson-04-vectorization-milvus.html) | BGE 双向量、Milvus 混合检索、幂等导入 | 05 节点5-7、07 |
| 05 | [lesson-05-retrieval-recall.html](lesson-05-retrieval-recall.html) | 意图确认、稠密/稀疏检索、HyDE、Web 搜索 MCP | 07、08、09 节点1-4 |
| 06 | [lesson-06-rrf-rerank.html](lesson-06-rrf-rerank.html) | RRF 加权融合公式、Rerank 动态 TopK 断崖 | 09 节点5-6 |
| 07 | [lesson-07-answer-sse.html](lesson-07-answer-sse.html) | Prompt 组装、SSE 流式输出、MongoDB 历史、图片双保险 | 08、09 节点7 |
| 08 | [lesson-08-interview.html](lesson-08-interview.html) | 10 分钟讲项目、高频追问 30 秒话术 | 全部 |

## 使用方式

- 每个课件是**独立 HTML 文件**，浏览器直接打开即可，含可交互 quiz（点击选项查看对错与解释）。
- 推荐顺序 01 → 08，每天一课，每课约 60-90 分钟（讲解 30 / 代码走读 20 / 动手 20 / 复盘 10）。
- 内容标签约定：
  - `[项目事实]` 能在项目根目录代码中找到（路径已核实）；
  - `[通用知识]` 框架/数据库通用原理；
  - `[面试简化]` 为 30 秒表达压缩的说法；
  - `[待验证]` 需运行命令或环境确认。
- 复习闭环：次日开课先做前课 quiz；每 3 天一次混合题；每 7 天一次 10 分钟项目讲解。

## 项目事实速查（全部课件共用）

- 项目：掌柜智库 = 企业级私有知识库智能问答系统（RAG），工程化代码约 5500 行。
- 技术栈：LangGraph（工作流编排）、LangChain（LLM 调用）、PyTorch + FlagEmbedding（BGE-M3 稠密+稀疏向量）、Milvus（向量库）、MinIO（对象存储）、MongoDB（历史/元数据）、FastAPI + SSE（Web 服务）、uv（依赖管理）、loguru（日志）。
- 导入链路 7 节点：`node_entry → node_pdf_to_md → node_md_img → node_document_split → node_item_name_recognition → node_bge_embedding → node_import_milvus`。
- 检索链路 8 节点（含虚拟节点）：`node_item_name_confirm → (条件路由) → node_multi_search → 四路并发(node_search_embedding / node_search_embedding_hyde / node_web_search_mcp / node_query_kg) → node_join → node_rrf → node_rerank → node_answer_output`。
- 真实代码路径（对照核实）：
  - `app/import_process/agent/state.py`（ImportGraphState）
  - `app/import_process/agent/main_graph.py`（导入主图）
  - `app/query_process/agent/state.py`（QueryGraphState）
  - `app/query_process/agent/main_graph.py`（检索主图）
  - `app/query_process/agent/nodes/*.py`（检索 8 节点）
  - `app/import_process/agent/nodes/*.py`（导入 7 节点）
  - `prompts/answer_out.prompt`、`prompts/rewritten_query_and_itemnames.prompt` 等
  - `app/conf/reranker_config.py`、`app/clients/mongo_history_utils.py`、`app/utils/sse_utils.py`
