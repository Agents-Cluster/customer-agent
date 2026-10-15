"""
功能测试：商品名对齐规则 (step_5_align_item_names)
对应 node_item_name_confirm 中的业务规则 A/B/C/D
"""
import pytest
from app.query_process.agent.nodes.node_item_name_confirm import step_5_align_item_names


class TestAlignItemNames:

    def test_rule_A_single_high_confidence(self):
        """规则 A：只有一个高置信度（>0.85）→ 直接确认"""
        query_results = [
            {
                "extracted_name": "HAK 180",
                "matches": [
                    {"item_name": "HAK 180 烫金机", "score": 0.92},
                    {"item_name": "HAK 200", "score": 0.70},
                ]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == ["HAK 180 烫金机"]
        assert result["options"] == []

    def test_rule_B_multiple_high_exact_match(self):
        """规则 B：多个高置信度，优先精确匹配 extracted_name"""
        query_results = [
            {
                "extracted_name": "HAK 180",
                "matches": [
                    {"item_name": "HAK 180", "score": 0.95},
                    {"item_name": "HAK 180 Pro", "score": 0.90},
                ]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == ["HAK 180"]

    def test_rule_B_multiple_high_no_exact_match(self):
        """规则 B：多个高置信度，没有精确匹配 → 取最高分"""
        query_results = [
            {
                "extracted_name": "烫金机",
                "matches": [
                    {"item_name": "HAK 180 Pro", "score": 0.91},
                    {"item_name": "HAK 180", "score": 0.88},
                ]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == ["HAK 180 Pro"]

    def test_rule_C_mid_confidence_options(self):
        """规则 C：没有高置信度，但有中置信度（0.6~0.85）→ 加入候选"""
        query_results = [
            {
                "extracted_name": "华为手机",
                "matches": [
                    {"item_name": "华为 P60", "score": 0.75},
                    {"item_name": "华为 P60 Art", "score": 0.68},
                    {"item_name": "华为 Mate 60", "score": 0.62},
                ]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == []
        assert "华为 P60" in result["options"]
        assert len(result["options"]) == 3

    def test_rule_D_low_confidence_nothing(self):
        """规则 D：所有分数 < 0.6 → 什么都不返回"""
        query_results = [
            {
                "extracted_name": "小米",
                "matches": [
                    {"item_name": "红米", "score": 0.50},
                    {"item_name": "小米手环", "score": 0.30},
                ]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == []
        assert result["options"] == []

    def test_no_matches(self):
        """没有匹配结果"""
        query_results = [
            {"extracted_name": "不存在的产品", "matches": []}
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"] == []
        assert result["options"] == []

    def test_multiple_extracted_names(self):
        """多个提取的商品名，各自独立对齐"""
        query_results = [
            {
                "extracted_name": "HAK 180",
                "matches": [{"item_name": "HAK 180 烫金机", "score": 0.93}]
            },
            {
                "extracted_name": "万用表",
                "matches": [{"item_name": "万用表 RS-12", "score": 0.65}]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert "HAK 180 烫金机" in result["confirmed_item_names"]
        assert "万用表 RS-12" in result["options"]

    def test_deduplication(self):
        """结果去重"""
        query_results = [
            {
                "extracted_name": "A",
                "matches": [{"item_name": "同一个产品", "score": 0.90}]
            },
            {
                "extracted_name": "B",
                "matches": [{"item_name": "同一个产品", "score": 0.91}]
            }
        ]
        result = step_5_align_item_names(query_results)
        assert result["confirmed_item_names"].count("同一个产品") == 1
