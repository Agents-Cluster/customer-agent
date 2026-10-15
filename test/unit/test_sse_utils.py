"""
测试 sse_utils.py
SSE 队列管理
"""
import json
import pytest
from app.utils import sse_utils


@pytest.fixture(autouse=True)
def clean_sse_store():
    sse_utils._session_stream.clear()
    yield
    sse_utils._session_stream.clear()


class TestSSEQueue:

    def test_create_queue(self):
        q = sse_utils.create_sse_queue("session_1")
        assert q is not None
        assert "session_1" in sse_utils._session_stream

    def test_get_queue(self):
        sse_utils.create_sse_queue("session_1")
        q = sse_utils.get_sse_queue("session_1")
        assert q is not None

    def test_get_nonexistent_queue(self):
        assert sse_utils.get_sse_queue("not_exist") is None

    def test_remove_queue(self):
        sse_utils.create_sse_queue("session_1")
        sse_utils.remove_sse_queue("session_1")
        assert sse_utils.get_sse_queue("session_1") is None

    def test_push_to_session(self):
        sse_utils.create_sse_queue("session_1")
        sse_utils.push_to_session("session_1", "delta", {"text": "hello"})
        q = sse_utils.get_sse_queue("session_1")
        msg = q.get_nowait()
        assert msg["event"] == "delta"
        assert msg["data"] == {"text": "hello"}

    def test_push_to_nonexistent_session_no_crash(self):
        """推送到不存在的 session 不应崩溃"""
        sse_utils.push_to_session("ghost", "delta", {"x": 1})  # 不抛异常即可

    def test_sse_pack_format(self):
        """_sse_pack 输出符合 SSE 协议格式"""
        result = sse_utils._sse_pack("delta", {"text": "你好"})
        assert result.startswith("event: delta\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        # data 部分是合法 JSON
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        payload = json.loads(data_line[6:])
        assert payload == {"text": "你好"}
