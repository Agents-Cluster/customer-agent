"""
测试 format_utils.py
JSON 格式化工具
"""
import json
import pytest
from app.utils.format_utils import format_state, format_json


class TestFormatJson:

    def test_dict(self):
        data = {"name": "测试", "value": 123}
        result = format_json(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_keep_chinese(self):
        """默认 ensure_ascii=False，中文不转义"""
        result = format_json({"msg": "你好世界"})
        assert "你好世界" in result
        assert "\\u" not in result

    def test_escape_chinese(self):
        """ensure_ascii=True 时中文被转义"""
        result = format_json({"msg": "你好"}, ensure_ascii=True)
        assert "\\u" in result

    def test_list(self):
        data = [1, 2, 3, "a"]
        result = format_json(data)
        assert json.loads(result) == data

    def test_custom_indent(self):
        result = format_json({"a": 1}, indent=2)
        # indent=2 时应该有两个空格的缩进
        lines = result.split("\n")
        assert "  \"a\"" in lines[1]

    def test_nested(self):
        data = {"a": {"b": [1, 2, {"c": "x"}]}}
        result = format_json(data)
        assert json.loads(result) == data


class TestFormatState:

    def test_basic_state(self):
        state = {"task_id": "001", "pdf_path": "test.pdf"}
        result = format_state(state)
        parsed = json.loads(result)
        assert parsed["task_id"] == "001"
        assert parsed["pdf_path"] == "test.pdf"

    def test_state_keeps_chinese(self):
        state = {"file_title": "操作手册.pdf", "item_name": "烫金机"}
        result = format_state(state)
        assert "烫金机" in result
