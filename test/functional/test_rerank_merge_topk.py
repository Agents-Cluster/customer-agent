"""
功能测试：Rerank 节点的文档合并与动态 TopK
- step_1_merge_docs: 本地 + 联网文档统一格式化
- step_3_topk: 基于断崖的动态截断
"""
import pytest
from app.query_process.agent.nodes.node_rerank import step_1_merge_docs, step_3_topk


class TestMergeDocs:

    def test_merge_local_and_web(self):
        """本地 RRF 文档 + 联网文档 合并为统一格式"""
        state = {
            "rrf_chunks": [
                {"chunk_id": "c1", "content": "RRF 是倒数排名融合", "item_name": "算法手册"},
            ],
            "web_search_docs": [
                {"title": "Rerank 详解", "url": "http://web.com/1", "snippet": "Rerank 是第二阶段排序"},
            ]
        }
        items = step_1_merge_docs(state)
        assert len(items) == 2

        # 本地文档
        local = [x for x in items if x["source"] == "local"][0]
        assert local["chunk_id"] == "c1"
        assert local["text"] == "RRF 是倒数排名融合"
        assert local["title"] == "算法手册"

        # 联网文档
        web = [x for x in items if x["source"] == "web"][0]
        assert web["url"] == "http://web.com/1"
        assert web["chunk_id"] is None
        assert "Rerank" in web["text"]

    def test_skip_empty_content(self):
        """无 content 的本地文档被跳过"""
        state = {
            "rrf_chunks": [
                {"chunk_id": "c1", "title": "无内容"},
                {"chunk_id": "c2", "content": "有内容"},
            ],
            "web_search_docs": []
        }
        items = step_1_merge_docs(state)
        assert len(items) == 1
        assert items[0]["chunk_id"] == "c2"

    def test_nested_entity_format(self):
        """兼容嵌套 entity 结构"""
        state = {
            "rrf_chunks": [
                {"entity": {"chunk_id": "c9", "content": "嵌套内容", "item_name": "手册"}},
            ],
            "web_search_docs": []
        }
        items = step_1_merge_docs(state)
        assert len(items) == 1
        assert items[0]["chunk_id"] == "c9"

    def test_empty_inputs(self):
        state = {"rrf_chunks": [], "web_search_docs": []}
        assert step_1_merge_docs(state) == []


class TestDynamicTopK:

    def test_no_gap_returns_all(self):
        """分数平滑下降，取满 max_topk（10）"""
        docs = [{"score": 1.0 - i * 0.01, "chunk_id": f"c{i}"} for i in range(10)]
        result = step_3_topk(docs)
        assert len(result) == 10

    def test_absolute_gap_truncates(self):
        """绝对差距 >= 0.5 时截断"""
        docs = [
            {"score": 0.9, "chunk_id": "c1"},
            {"score": 0.85, "chunk_id": "c2"},
            {"score": 0.30, "chunk_id": "c3"},  # 断崖：0.85 - 0.30 = 0.55 > 0.5
            {"score": 0.20, "chunk_id": "c4"},
        ]
        result = step_3_topk(docs)
        # 应在第3条前截断，只保留前2条
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"

    def test_relative_gap_truncates(self):
        """相对差距 >= 25% 时截断"""
        docs = [
            {"score": 0.80, "chunk_id": "c1"},
            {"score": 0.78, "chunk_id": "c2"},
            {"score": 0.50, "chunk_id": "c3"},  # (0.78-0.50)/0.78 ≈ 36% > 25%
            {"score": 0.40, "chunk_id": "c4"},
        ]
        result = step_3_topk(docs)
        assert len(result) == 2

    def test_small_list_no_truncation(self):
        """少于 min_topk 的列表不截断"""
        docs = [{"score": 0.9, "chunk_id": "c1"}]
        result = step_3_topk(docs)
        assert len(result) == 1

    def test_empty_list(self):
        assert step_3_topk([]) == []
