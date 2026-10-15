"""
测试 rate_limit_utils.py
滑动窗口限流
"""
from collections import deque
import time
from unittest.mock import patch
import pytest
from app.utils.rate_limit_utils import apply_api_rate_limit


class TestRateLimit:

    def test_under_limit_no_sleep(self):
        """窗口内请求数未超限时不 sleep"""
        q = deque()
        with patch("app.utils.rate_limit_utils.time.sleep") as mock_sleep:
            apply_api_rate_limit(q, max_requests=5, window_seconds=60)
            mock_sleep.assert_not_called()
        assert len(q) == 1

    def test_over_limit_triggers_sleep(self):
        """超过限额时应触发 sleep"""
        q = deque([time.time() - 10, time.time() - 20])  # 已有 2 个请求
        with patch("app.utils.rate_limit_utils.time.sleep") as mock_sleep:
            apply_api_rate_limit(q, max_requests=2, window_seconds=60)
            # 应该 sleep 了（因为窗口内已有 2 个，再加就超）
            mock_sleep.assert_called_once()
            sleep_arg = mock_sleep.call_args[0][0]
            assert sleep_arg > 0

    def test_old_requests_expire(self):
        """窗口外的旧请求应被清理，不占名额"""
        old_time = time.time() - 120  # 2 分钟前，超出 60 秒窗口
        q = deque([old_time, old_time])
        with patch("app.utils.rate_limit_utils.time.sleep") as mock_sleep:
            apply_api_rate_limit(q, max_requests=2, window_seconds=60)
            # 旧请求被清掉，没超限额，不 sleep
            mock_sleep.assert_not_called()
        # 队列里只剩当前请求 + 可能还有清理后的
        assert len(q) == 1

    def test_queue_appends_current_time(self):
        """每次调用后当前时间戳应入队"""
        q = deque()
        before = time.time()
        apply_api_rate_limit(q, max_requests=10, window_seconds=60)
        after = time.time()
        assert len(q) == 1
        assert before <= q[0] <= after

    def test_custom_window(self):
        """自定义窗口时长生效"""
        # 10 秒前的请求，在 5 秒窗口里算过期
        old = time.time() - 10
        q = deque([old, old])
        with patch("app.utils.rate_limit_utils.time.sleep") as mock_sleep:
            apply_api_rate_limit(q, max_requests=2, window_seconds=5)
            mock_sleep.assert_not_called()
