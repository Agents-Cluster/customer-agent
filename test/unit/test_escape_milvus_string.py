"""
测试 escape_milvus_string.py
Milvus filter 表达式的字符串转义
"""
import pytest
from app.utils.escape_milvus_string_utils import escape_milvus_string


class TestEscapeMilvusString:

    def test_backslash(self):
        """反斜杠要双写"""
        assert escape_milvus_string(r"C:\new\test") == r"C:\\new\\test"

    def test_double_quote(self):
        """双引号要转义"""
        s = '他说"你好"'
        assert escape_milvus_string(s) == r'他说\"你好\"'

    def test_newline_tab_carriage(self):
        """换行/回车/制表符统一替换为空格"""
        s = "line1\nline2\rline3\tline4"
        result = escape_milvus_string(s)
        assert "\n" not in result
        assert "\r" not in result
        assert "\t" not in result
        assert result == "line1 line2 line3 line4"

    def test_none_input(self):
        """None 输入返回空字符串"""
        assert escape_milvus_string(None) == ""

    def test_non_string_input(self):
        """非字符串输入自动转 str"""
        assert escape_milvus_string(123) == "123"
        assert escape_milvus_string(3.14) == "3.14"

    def test_normal_string_unchanged(self):
        """普通无特殊字符的字符串保持不变"""
        s = "HAK 180 烫金机"
        assert escape_milvus_string(s) == s

    def test_complex_mixed(self):
        """混合特殊字符"""
        s = r'路径: "C:\test" 换行\n结束'
        result = escape_milvus_string(s)
        # 反斜杠都双写
        assert r'\\' in result
        # 双引号转义
        assert r'\"' in result
        # 没有真换行
        assert "\n" not in result
