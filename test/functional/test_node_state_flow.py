"""
节点状态流转测试
================
直接调用每个 node_xxx(state) 函数，验证：
1. 输入 state 的哪些字段被读取
2. 输出 state 更新了哪些字段
3. 边界情况（空输入、缺字段）是否健壮

所有外部依赖（Milvus / LLM / MongoDB / SSE）全部 mock。
"""
import pytest
from unittest.mock import patch, MagicMock


# ============================================================
# 通用 mock：task_utils 的 add_running_task / add_done_task
# ============================================================
@pytest.fixture(autouse=True)
def mock_task_tracker():
    with patch("app.utils.task_utils.add_running_task"), \
         patch("app.utils.task_utils.add_done_task"), \
         patch("app.utils.task_utils.set_task_result"), \
         patch("app.utils.sse_utils.push_to_session"):
        yield


# ============================================================
# 1. node_rrf 状态流转
# ============================================================
class TestNodeRRFState:

    def test_input_output_fields(self):
        """输入: embedding_chunks + hyde_embedding_chunks → 输出: rrf_chunks"""
        from app.query_process.agent.nodes.node_rrf import node_rrf

        state = {
            "session_id": "s1",
            "is_stream": False,
            "embedding_chunks": [
                {"chunk_id": "c1", "content": "A"},
                {"chunk_id": "c2", "content": "B"},
            ],
            "hyde_embedding_chunks": [
                {"chunk_id": "c2", "content": "B"},
                {"chunk_id": "c3", "content": "C"},
            ],
        }
        result = node_rrf(state)

        # 输出只包含 rrf_chunks
        assert "rrf_chunks" in result
        assert len(result["rrf_chunks"]) == 3  # c1, c2, c3 并集

    def test_empty_inputs(self):
        """两路输入都为空时，输出空列表不崩溃"""
        from app.query_process.agent.nodes.node_rrf import node_rrf

        state = {
            "session_id": "s1",
            "is_stream": False,
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
        }
        result = node_rrf(state)
        assert result["rrf_chunks"] == []

    def test_missing_fields(self):
        """state 里完全没有这些字段时不崩溃"""
        from app.query_process.agent.nodes.node_rrf import node_rrf

        state = {"session_id": "s1", "is_stream": False}
        result = node_rrf(state)
        assert result["rrf_chunks"] == []


# ============================================================
# 2. node_rerank 状态流转
# ============================================================
class TestNodeRerankState:

    def test_input_output_fields(self):
        """输入: rrf_chunks + web_search_docs → 输出: reranked_docs"""
        from app.query_process.agent.nodes.node_rerank import node_rerank

        # mock reranker 模型（分数平滑下降，不触发断崖）
        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.9, 0.85, 0.80]

        with patch("app.query_process.agent.nodes.node_rerank.get_reranker_model",
                   return_value=mock_reranker):
            state = {
                "session_id": "s1",
                "is_stream": False,
                "rewritten_query": "什么是 RRF？",
                "rrf_chunks": [
                    {"chunk_id": "c1", "content": "RRF 是倒数排名融合"},
                    {"chunk_id": "c2", "content": "BGE 是重排序模型"},
                ],
                "web_search_docs": [
                    {"title": "Rerank 详解", "snippet": "Rerank 是第二阶段", "url": "http://x.com"},
                ],
            }
            result = node_rerank(state)

        assert "reranked_docs" in result
        assert len(result["reranked_docs"]) == 3

    def test_empty_docs_no_crash(self):
        """没有任何文档时不崩溃，返回空列表"""
        from app.query_process.agent.nodes.node_rerank import node_rerank

        state = {
            "session_id": "s1",
            "is_stream": False,
            "rewritten_query": "测试问题",
            "rrf_chunks": [],
            "web_search_docs": [],
        }
        result = node_rerank(state)
        assert result["reranked_docs"] == []

    def test_no_query_fallback(self):
        """没有 rewritten_query 时跳过重排，返回空"""
        from app.query_process.agent.nodes.node_rerank import node_rerank

        state = {
            "session_id": "s1",
            "is_stream": False,
            "rrf_chunks": [{"chunk_id": "c1", "content": "x"}],
            "web_search_docs": [],
        }
        result = node_rerank(state)
        # 没有 question，step_2 返回空列表
        assert result["reranked_docs"] == []


# ============================================================
# 3. node_answer_output 状态流转
# ============================================================
class TestNodeAnswerOutputState:

    def test_answer_already_exists_short_circuit(self):
        """state 里已有 answer 时，跳过 LLM 直接输出"""
        from app.query_process.agent.nodes.node_answer_output import node_answer_output

        with patch("app.query_process.agent.nodes.node_answer_output.save_chat_message"):
            state = {
                "session_id": "s1",
                "is_stream": False,
                "original_query": "这个多少钱？",
                "rewritten_query": "",
                "item_names": [],
                "history": [],
                "reranked_docs": [],
                "answer": "您是想问 HAK 180 还是 HAK 200？",
            }
            result = node_answer_output(state)

        # answer 保持不变
        assert "HAK 180" in result["answer"]

    def test_no_answer_calls_llm(self):
        """没有 answer 时，调用 LLM 生成"""
        from app.query_process.agent.nodes.node_answer_output import node_answer_output

        # mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这是生成的答案"
        mock_llm.invoke.return_value = mock_response

        with patch("app.query_process.agent.nodes.node_answer_output.get_llm_client",
                   return_value=mock_llm), \
             patch("app.query_process.agent.nodes.node_answer_output.save_chat_message"), \
             patch("app.query_process.agent.nodes.node_answer_output.load_prompt",
                   return_value="mock prompt"):
            state = {
                "session_id": "s1",
                "is_stream": False,
                "original_query": "HAK 180 怎么用？",
                "rewritten_query": "HAK 180 怎么用？",
                "item_names": ["HAK 180"],
                "history": [],
                "reranked_docs": [
                    {"text": "HAK 180 操作步骤", "score": 0.9, "source": "local"}
                ],
                "answer": None,
            }
            result = node_answer_output(state)

        assert result["answer"] == "这是生成的答案"
        assert "prompt" in result

    def test_llm_error_graceful(self):
        """LLM 报错时不崩溃，返回兜底文案"""
        from app.query_process.agent.nodes.node_answer_output import node_answer_output

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API 挂了")

        with patch("app.query_process.agent.nodes.node_answer_output.get_llm_client",
                   return_value=mock_llm), \
             patch("app.query_process.agent.nodes.node_answer_output.load_prompt",
                   return_value="mock prompt"):
            state = {
                "session_id": "s1",
                "is_stream": False,
                "original_query": "测试",
                "rewritten_query": "测试",
                "item_names": [],
                "history": [],
                "reranked_docs": [],
                "answer": None,
            }
            result = node_answer_output(state)

        assert "抱歉" in result["answer"]


# ============================================================
# 4. node_search_embedding 状态流转
# ============================================================
class TestNodeSearchEmbeddingState:

    def test_no_item_names_returns_empty(self):
        """item_names 为空时直接返回空，不连 Milvus"""
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding

        state = {
            "session_id": "s1",
            "is_stream": False,
            "rewritten_query": "随便问问",
            "item_names": [],
        }
        result = node_search_embedding(state)
        assert result["embedding_chunks"] == []

    def test_normal_search_calls_milvus(self):
        """有 item_names 时，调 embedding + Milvus，返回结果"""
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding

        mock_embeddings = {
            "dense": [[0.1, 0.2, 0.3]],
            "sparse": [{"1": 0.5, "2": 0.8}],
        }
        mock_hits = [[
            MagicMock(entity={"chunk_id": "c1", "content": "内容"}, distance=0.9),
        ]]

        with patch("app.query_process.agent.nodes.node_search_embedding.generate_embeddings",
                   return_value=mock_embeddings), \
             patch("app.query_process.agent.nodes.node_search_embedding.create_hybrid_search_requests"), \
             patch("app.query_process.agent.nodes.node_search_embedding.get_milvus_client"), \
             patch("app.query_process.agent.nodes.node_search_embedding.hybrid_search",
                   return_value=mock_hits):
            state = {
                "session_id": "s1",
                "is_stream": False,
                "rewritten_query": "HAK 180 怎么用？",
                "item_names": ["HAK 180 烫金机"],
            }
            result = node_search_embedding(state)

        assert "embedding_chunks" in result
        assert len(result["embedding_chunks"]) == 1


# ============================================================
# 5. 图片提取纯函数
# ============================================================
class TestImageExtraction:

    def test_extract_from_markdown(self):
        """从 Markdown 图片语法提取"""
        from app.query_process.agent.nodes.node_answer_output import _extract_images_from_docs

        docs = [{
            "text": '参考图 ![面板图](http://img.com/panel.jpg) 如下'
        }]
        images = _extract_images_from_docs(docs)
        assert "http://img.com/panel.jpg" in images

    def test_extract_from_url_field(self):
        """从 URL 字段直接提取图片（web 结果）"""
        from app.query_process.agent.nodes.node_answer_output import _extract_images_from_docs

        docs = [{"url": "http://web.com/photo.png", "text": "普通文字"}]
        images = _extract_images_from_docs(docs)
        assert "http://web.com/photo.png" in images

    def test_dedup(self):
        """同一张图不重复出现"""
        from app.query_process.agent.nodes.node_answer_output import _extract_images_from_docs

        docs = [
            {"text": '![a](http://x.com/1.jpg)'},
            {"text": '![b](http://x.com/1.jpg)'},
        ]
        images = _extract_images_from_docs(docs)
        assert images.count("http://x.com/1.jpg") == 1

    def test_empty_input(self):
        from app.query_process.agent.nodes.node_answer_output import _extract_images_from_docs
        assert _extract_images_from_docs([]) == []
        assert _extract_images_from_docs(None) == []
