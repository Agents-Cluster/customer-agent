"""
测试 normalize_sparse_vector.py
L2 归一化的边界与正确性
"""
import math
import numpy as np
import pytest
from app.utils.normalize_sparse_vector import normalize_sparse_vector


class TestNormalizeSparseVector:

    def test_normal_vector(self):
        """普通稀疏向量：归一化后 L2 范数应为 1"""
        vec = {1: 3.0, 2: 4.0}
        result = normalize_sparse_vector(vec)
        # 3-4-5 三角形，归一化后应为 0.6 和 0.8
        assert math.isclose(result[1], 0.6, rel_tol=1e-6)
        assert math.isclose(result[2], 0.8, rel_tol=1e-6)

    def test_empty_vector(self):
        """空向量直接返回原对象"""
        assert normalize_sparse_vector({}) == {}

    def test_zero_vector(self):
        """全零向量：范数接近 0 时应原样返回，不除零"""
        vec = {1: 0.0, 2: 0.0}
        result = normalize_sparse_vector(vec)
        assert result == vec

    def test_single_dimension(self):
        """单维度向量：归一化后应为 1"""
        vec = {5: 10.0}
        result = normalize_sparse_vector(vec)
        assert math.isclose(result[5], 1.0, rel_tol=1e-6)

    def test_negative_values(self):
        """负值也能正确归一化（符号保留）"""
        vec = {1: -3.0, 2: -4.0}
        result = normalize_sparse_vector(vec)
        assert math.isclose(result[1], -0.6, rel_tol=1e-6)
        assert math.isclose(result[2], -0.8, rel_tol=1e-6)

    def test_output_is_new_dict(self):
        """返回的是新 dict，不修改原对象"""
        vec = {1: 3.0, 2: 4.0}
        result = normalize_sparse_vector(vec)
        assert result is not vec
        # 原对象不变
        assert vec[1] == 3.0
        assert vec[2] == 4.0

    def test_large_vector(self):
        """大向量：归一化后整体范数为 1"""
        vec = {i: float(i) for i in range(1, 101)}
        result = normalize_sparse_vector(vec)
        norm = math.sqrt(sum(v * v for v in result.values()))
        assert math.isclose(norm, 1.0, rel_tol=1e-6)
