"""
测试 node_rrf.py 中的核心算法
- reciprocal_rank_fusion: 带权 RRF
- _as_entity_list: 多格式结果统一
"""
import pytest
from app.query_process.agent.nodes.node_rrf import reciprocal_rank_fusion, _as_entity_list


class TestReciprocalRankFusion:

    def test_basic_fusion(self):
        """两路都有的文档，RRF 分数更高"""
        docs_a = [{"chunk_id": "a1", "content": "A"},
                  {"chunk_id": "a2", "content": "B"}]
        docs_b = [{"chunk_id": "a2", "content": "B"},
                  {"chunk_id": "a3", "content": "C"}]
        result = reciprocal_rank_fusion([(docs_a, 1.0), (docs_b, 1.0)], k=60)
        # a2 在两路都出现，应排第一
        top_id = result[0][0]["chunk_id"]
        assert top_id == "a2"

    def test_k_parameter(self):
        """k 越大，排名差异越小（平滑效应）"""
        docs = [{"chunk_id": "x1"}, {"chunk_id": "x2"}]
        result_small_k = reciprocal_rank_fusion([(docs, 1.0)], k=10)
        result_large_k = reciprocal_rank_fusion([(docs, 1.0)], k=100)
        # k=10 时第一名领先更多
        gap_small = result_small_k[0][1] - result_small_k[1][1]
        gap_large = result_large_k[0][1] - result_large_k[1][1]
        assert gap_small > gap_large

    def test_weights(self):
        """权重高的来源排名更靠前"""
        docs_a = [{"chunk_id": "a1"}]  # 权重 0.1
        docs_b = [{"chunk_id": "b1"}]  # 权重 1.0
        result = reciprocal_rank_fusion(
            [(docs_a, 0.1), (docs_b, 1.0)], k=60
        )
        # b1 权重高，虽然只出现一次，分数仍高于 a1?
        # 计算: a1 = 0.1 * 1/61 ≈ 0.0016; b1 = 1.0 * 1/61 ≈ 0.0164
        # b1 分数更高
        top_id = result[0][0]["chunk_id"]
        assert top_id == "b1"

    def test_max_results(self):
        """截断到 top N"""
        docs = [{"chunk_id": f"id_{i}"} for i in range(10)]
        result = reciprocal_rank_fusion([(docs, 1.0)], max_results=3)
        assert len(result) == 3

    def test_missing_id_skipped(self):
        """没有 chunk_id/id 的文档被跳过"""
        docs = [{"content": "no id"}, {"chunk_id": "ok"}]
        result = reciprocal_rank_fusion([(docs, 1.0)])
        assert len(result) == 1
        assert result[0][0]["chunk_id"] == "ok"

    def test_uses_id_when_no_chunk_id(self):
        """没有 chunk_id 时回退用 id 字段"""
        docs = [{"id": "doc_1", "content": "x"}]
        result = reciprocal_rank_fusion([(docs, 1.0)])
        assert len(result) == 1
        assert result[0][0]["id"] == "doc_1"


class TestAsEntityList:

    def test_flat_dict(self):
        """扁平 dict 直接用"""
        docs = [{"chunk_id": "c1", "content": "hello"}]
        result = _as_entity_list(docs)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_nested_entity_dict(self):
        """嵌套 {"entity": {...}, "distance": ...}"""
        docs = [{"entity": {"chunk_id": "c1", "content": "x"},
                 "id": "pk_1", "distance": 0.9}]
        result = _as_entity_list(docs)
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["score"] == 0.9

    def test_none_items_skipped(self):
        """None / 空列表跳过"""
        result = _as_entity_list([None, {}, None])
        assert result == []

    def test_milvus_hit_like_object(self):
        """模拟 Milvus Hit 对象（有 entity 和 id 属性）"""
        class FakeEntity:
            def to_dict(self):
                return {"chunk_id": "c99", "content": "from hit"}

        class FakeHit:
            def __init__(self):
                self.entity = FakeEntity()
                self.id = "pk_99"
                self.distance = 0.95

        result = _as_entity_list([FakeHit()])
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c99"
        assert result[0]["score"] == 0.95

    def test_empty_input(self):
        assert _as_entity_list(None) == []
        assert _as_entity_list([]) == []
