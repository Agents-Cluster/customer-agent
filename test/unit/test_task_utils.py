"""
测试 task_utils.py
内存态任务状态机
"""
import pytest
from app.utils import task_utils


@pytest.fixture(autouse=True)
def clean_task_store():
    """每个测试前后清空任务存储，避免相互污染"""
    task_utils._tasks_running_list.clear()
    task_utils._tasks_done_list.clear()
    task_utils._tasks_status.clear()
    task_utils._tasks_result.clear()
    yield
    task_utils._tasks_running_list.clear()
    task_utils._tasks_done_list.clear()
    task_utils._tasks_status.clear()
    task_utils._tasks_result.clear()


class TestTaskLifecycle:

    def test_add_running(self):
        task_utils.add_running_task("task_1", "node_entry")
        assert task_utils.get_running_task_list("task_1") == ["检查文件"]

    def test_add_running_no_duplicate(self):
        """同一节点重复 add 不重复追加"""
        task_utils.add_running_task("task_1", "node_entry")
        task_utils.add_running_task("task_1", "node_entry")
        assert task_utils.get_running_task_list("task_1") == ["检查文件"]

    def test_add_done_removes_running(self):
        """add_done 后节点从 running 移到 done"""
        task_utils.add_running_task("task_1", "node_entry")
        task_utils.add_done_task("task_1", "node_entry")
        assert task_utils.get_running_task_list("task_1") == []
        assert task_utils.get_done_task_list("task_1") == ["检查文件"]

    def test_unknown_node_name_kept_as_is(self):
        """未映射的节点名原样返回"""
        task_utils.add_running_task("task_1", "my_custom_node")
        assert task_utils.get_running_task_list("task_1") == ["my_custom_node"]

    def test_status_update(self):
        task_utils.update_task_status("task_1", "processing")
        assert task_utils.get_task_status("task_1") == "processing"

    def test_result_set_and_get(self):
        task_utils.set_task_result("task_1", "answer", "这是答案")
        assert task_utils.get_task_result("task_1", "answer") == "这是答案"

    def test_result_default(self):
        assert task_utils.get_task_result("task_999", "answer", "默认值") == "默认值"

    def test_clear_task(self):
        task_utils.add_running_task("task_1", "node_entry")
        task_utils.update_task_status("task_1", "completed")
        task_utils.set_task_result("task_1", "answer", "x")
        task_utils.clear_task("task_1")

        assert task_utils.get_task_status("task_1") == ""
        assert task_utils.get_running_task_list("task_1") == []
        assert task_utils.get_done_task_list("task_1") == []

    def test_multiple_nodes_order(self):
        """多个节点按添加顺序排列"""
        task_utils.add_running_task("t1", "node_pdf_to_md")
        task_utils.add_running_task("t1", "node_document_split")
        task_utils.add_running_task("t1", "node_bge_embedding")
        running = task_utils.get_running_task_list("t1")
        assert running == ["PDF转Markdown", "文档切分", "向量生成"]
