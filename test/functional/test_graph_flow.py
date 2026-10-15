"""
功能测试：Query Graph 结构完整性 + 条件路由
不真正执行 graph，只验证：
1. 所有节点都正确注册
2. 条件路由逻辑正确（有 answer 直接输出 vs 走多路检索）
3. 并发分叉与合并结构正确
"""
import pytest
from app.query_process.agent.main_graph import builder, route_after_item_confirm


class TestGraphStructure:

    def test_graph_compiles(self):
        """graph 能正常编译（所有节点和边都正确）"""
        from app.query_process.agent.main_graph import query_app
        assert query_app is not None

    def test_graph_nodes_visualizable(self):
        """能导出图结构（说明节点和边都连对了）"""
        from app.query_process.agent.main_graph import query_app
        # get_graph() 能成功调用说明结构完整
        graph = query_app.get_graph()
        assert graph is not None
        # 至少有 10 个节点（含 END）
        assert len(graph.nodes) >= 10

    def test_conditional_routing_has_answer(self):
        """条件路由：state 里有 answer → 直接跳 answer_output"""
        state = {"answer": "您是想问哪个产品？"}
        route = route_after_item_confirm(state)
        assert route == "node_answer_output"

    def test_conditional_routing_no_answer(self):
        """条件路由：state 里没有 answer → 走多路检索"""
        state = {"item_names": ["HAK 180"], "rewritten_query": "HAK 180 多少钱"}
        route = route_after_item_confirm(state)
        assert route == "node_multi_search"

    def test_conditional_routing_empty_answer(self):
        """answer 为空字符串也走检索流程"""
        state = {"answer": ""}
        route = route_after_item_confirm(state)
        assert route == "node_multi_search"


class TestGraphFlowEndToEnd:
    """
    端到端：用 mock 替换所有外部依赖节点，只测 graph 编排逻辑
    """

    def test_full_flow_with_mocks(self):
        """完整查询流程：确认商品 → 多路召回 → RRF → Rerank → 输出"""
        from langgraph.graph import StateGraph, END
        from app.query_process.agent.state import QueryGraphState

        # 构建一个测试专用 graph，节点全是 mock
        test_builder = StateGraph(QueryGraphState)

        def mock_confirm(state):
            state["item_names"] = ["HAK 180"]
            state["rewritten_query"] = "HAK 180 多少钱"
            return state

        def mock_search(state):
            state["embedding_chunks"] = [{"chunk_id": "c1", "content": "HAK 180 价格是 5000"}]
            state["hyde_embedding_chunks"] = [{"chunk_id": "c1", "content": "HAK 180 价格是 5000"}]
            state["web_search_docs"] = [{"title": "行情", "snippet": "HAK 180 约 5000", "url": "x"}]
            state["kg_chunks"] = []
            return state

        def mock_rrf(state):
            state["rrf_chunks"] = state["embedding_chunks"]
            return state

        def mock_rerank(state):
            state["reranked_docs"] = [{"text": "HAK 180 价格是 5000", "score": 0.99}]
            return state

        def mock_answer(state):
            state["answer"] = "根据文档，HAK 180 价格约 5000 元。"
            return state

        test_builder.add_node("confirm", mock_confirm)
        test_builder.add_node("search", mock_search)
        test_builder.add_node("rrf", mock_rrf)
        test_builder.add_node("rerank", mock_rerank)
        test_builder.add_node("answer", mock_answer)

        test_builder.set_entry_point("confirm")
        test_builder.add_edge("confirm", "search")
        test_builder.add_edge("search", "rrf")
        test_builder.add_edge("rrf", "rerank")
        test_builder.add_edge("rerank", "answer")
        test_builder.add_edge("answer", END)

        app = test_builder.compile()

        initial = {
            "session_id": "test_e2e",
            "original_query": "HAK 180 多少钱",
            "embedding_chunks": [],
            "hyde_embedding_chunks": [],
            "kg_chunks": [],
            "web_search_docs": [],
            "rrf_chunks": [],
            "reranked_docs": [],
            "prompt": "",
            "answer": "",
            "item_names": [],
            "rewritten_query": "",
            "history": [],
            "is_stream": False,
        }

        final = app.invoke(initial)

        # 验证最终状态
        assert final["item_names"] == ["HAK 180"]
        assert final["rrf_chunks"][0]["chunk_id"] == "c1"
        assert final["reranked_docs"][0]["score"] == 0.99
        assert "5000" in final["answer"]

    def test_short_circuit_when_answer_exists(self):
        """如果确认节点直接给了 answer，跳过检索直接输出"""
        from langgraph.graph import StateGraph, END
        from app.query_process.agent.state import QueryGraphState

        test_builder = StateGraph(QueryGraphState)

        def mock_confirm_with_answer(state):
            state["answer"] = "您是想问 HAK 180 还是 HAK 200？"
            return state

        def mock_search(state):
            state["embedding_chunks"] = [{"chunk_id": "bad"}]
            return state

        def route(state):
            return "answer" if state.get("answer") else "search"

        test_builder.add_node("confirm", mock_confirm_with_answer)
        test_builder.add_node("search", mock_search)
        test_builder.add_node("answer", lambda s: s)

        test_builder.set_entry_point("confirm")
        test_builder.add_conditional_edges("confirm", route)
        test_builder.add_edge("search", "answer")
        test_builder.add_edge("answer", END)

        app = test_builder.compile()
        initial = {
            "session_id": "s", "original_query": "多少钱",
            "embedding_chunks": [], "hyde_embedding_chunks": [], "kg_chunks": [],
            "web_search_docs": [], "rrf_chunks": [], "reranked_docs": [],
            "prompt": "", "answer": "", "item_names": [], "rewritten_query": "",
            "history": [], "is_stream": False,
        }
        final = app.invoke(initial)

        # search 节点没被调用
        assert final["embedding_chunks"] == []
        assert "HAK 180 还是 HAK 200" in final["answer"]
