"""
功能测试：确认状态分支 (step_6_check_confirmation)
分支 A：有确认商品名 → 继续检索
分支 B：只有候选商品名 → 反问用户
分支 C：无结果 → 拒绝回答
"""
import pytest
from unittest.mock import patch
from app.query_process.agent.nodes.node_item_name_confirm import step_6_check_confirmation


class TestCheckConfirmation:

    def setup_method(self):
        self.base_state = {
            "session_id": "test_sess",
            "original_query": "这个多少钱？",
            "rewritten_query": "HAK 180 烫金机多少钱？",
        }
        self.history = []

    @patch("app.query_process.agent.nodes.node_item_name_confirm.update_message_item_names")
    def test_branch_A_confirmed(self, mock_update):
        """分支 A：有确认商品名，state 正确设置"""
        align_result = {"confirmed_item_names": ["HAK 180 烫金机"], "options": []}
        state = step_6_check_confirmation(
            self.base_state.copy(), align_result, "test_sess", self.history, "HAK 180 烫金机多少钱？"
        )
        assert state["item_names"] == ["HAK 180 烫金机"]
        assert "answer" not in state  # 没有 answer，继续走检索
        assert state["rewritten_query"] == "HAK 180 烫金机多少钱？"

    def test_branch_B_options_only(self):
        """分支 B：只有候选，生成反问 answer"""
        align_result = {"confirmed_item_names": [], "options": ["华为 P60", "华为 P60 Art", "华为 Mate 60"]}
        state = step_6_check_confirmation(
            self.base_state.copy(), align_result, "test_sess", self.history, "华为 P60 多少钱？"
        )
        assert state["item_names"] == []
        assert "您是想问以下哪个产品" in state["answer"]
        assert "华为 P60" in state["answer"]

    def test_branch_C_no_result(self):
        """分支 C：无确认无候选，生成拒绝 answer"""
        align_result = {"confirmed_item_names": [], "options": []}
        state = step_6_check_confirmation(
            self.base_state.copy(), align_result, "test_sess", self.history, "随便问问"
        )
        assert state["item_names"] == []
        assert "抱歉，未找到相关产品" in state["answer"]

    def test_branch_A_clears_existing_answer(self):
        """分支 A：如果之前有 answer（比如反问），确认后要清掉"""
        state = self.base_state.copy()
        state["answer"] = "之前的反问"
        align_result = {"confirmed_item_names": ["HAK 180"], "options": []}
        result = step_6_check_confirmation(state, align_result, "s", [], "q")
        assert "answer" not in result

    def test_empty_align_result_handled(self):
        """align_result 为 None 不崩溃"""
        state = step_6_check_confirmation(
            self.base_state.copy(), None, "s", [], "q"
        )
        assert "抱歉" in state["answer"]
