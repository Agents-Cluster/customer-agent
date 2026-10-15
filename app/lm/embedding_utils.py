import requests
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None
# Cloudflare API 客户端（轻量，无需单例）
_cf_session = None

def _get_cf_session():
    global _cf_session
    if _cf_session is None:
        _cf_session = requests.Session()
        _cf_session.trust_env = False  # 绕过系统代理
    return _cf_session

def _is_cf_enabled():
    return bool(embedding_config.cf_account_id and embedding_config.cf_api_token)

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象。
    优先走 Cloudflare Workers AI API（配了 CF_ACCOUNT_ID 时），否则加载本地模型。
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    if _is_cf_enabled():
        # Cloudflare 模式：返回一个轻量代理对象，encode_documents 走 API
        logger.info(f"使用 Cloudflare Workers AI embedding: @cf/baai/bge-m3 (account={embedding_config.cf_account_id[:8]}...)")
        _bge_m3_ef = _CFBGEM3Client()
        return _bge_m3_ef

    # 本地模型模式
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False
    logger.info("开始初始化本地BGE-M3模型", extra={"model_name": model_name, "device": device})
    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        normalize_embeddings=True
    )
    logger.success("本地BGE-M3模型初始化成功")
    return _bge_m3_ef


class _CFBGEM3Client:
    """Cloudflare Workers AI bge-m3 客户端，接口对齐 BGEM3EmbeddingFunction.encode_documents"""
    def encode_documents(self, texts):
        s = _get_cf_session()
        url = f"https://api.cloudflare.com/client/v4/accounts/{embedding_config.cf_account_id}/ai/run/@cf/baai/bge-m3"
        r = s.post(url,
            headers={"Authorization": f"Bearer {embedding_config.cf_api_token}", "Content-Type": "application/json"},
            json={"text": texts}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare embedding 失败: {data}")
        dense = data["result"]["data"]
        # Cloudflare 不返回 sparse 向量，返回空 dict 列表占位
        sparse = [{} for _ in dense]
        return {"dense": dense, "sparse": sparse}


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        # Cloudflare 模式下 sparse 已是 dict 列表（空 dict），直接透传
        if isinstance(embeddings["sparse"], list) and len(embeddings["sparse"]) > 0 and isinstance(embeddings["sparse"][0], dict):
            processed_sparse = embeddings["sparse"]
        else:
            for i in range(len(texts)):
                sparse_indices = embeddings["sparse"].indices[
                    embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
                ].tolist()
                sparse_data = embeddings["sparse"].data[
                    embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
                ].tolist()
                sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
                processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        # Cloudflare 返回的已是 list，本地 numpy array 需要 tolist()
        dense_out = [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings["dense"]]
        result = {
            "dense": dense_out,
            "sparse": processed_sparse
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性。
"""