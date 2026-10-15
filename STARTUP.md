# 项目启动命令

> 以下命令均假设在项目根目录下执行。

## 1. 启动 Docker 中间件（Milvus / MongoDB / Neo4j）

```bash
docker compose up -d
```

验证：
- Milvus: http://127.0.0.1:19530
- MongoDB: 127.0.0.1:27017
- Neo4j: http://127.0.0.1:7474

## 2. 启动 S3 兼容服务（MinIO 替代，用 moto）

```bash
.venv/bin/moto_server -p 9000 &
```

然后创建 bucket（首次）：
```bash
.venv/bin/python -c "
from minio import Minio
c = Minio('127.0.0.1:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
c.make_bucket('knowledge-base-files')
print('bucket created')
"
```

## 3. 启动导入服务（8000 端口，PDF 上传 → 向量入库）

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  .venv/bin/python -m app.import_process.api.file_import_service
```

访问：
- Swagger: http://127.0.0.1:8000/docs
- 上传页面: http://127.0.0.1:8000/import.html

## 4. 启动查询服务（8001 端口，RAG 问答）

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  .venv/bin/python -m app.query_process.api.query_service
```

访问：
- 问答页面: http://127.0.0.1:8001/chat.html

## 5. 启动 Attu（Milvus 可视化 GUI，可选）

```bash
docker run -d --name attu -p 3000:3000 docker.1ms.run/zilliz/attu:v2.4
```

访问：http://127.0.0.1:3000
- Milvus Address 填：`host.docker.internal:19530`

## 快速一键启动（新开终端窗口跑）

```bash
# 中间件
docker compose up -d

# S3
.venv/bin/moto_server -p 9000 &

# 导入服务（后台）
env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python -m app.import_process.api.file_import_service &

# 查询服务（后台）
env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python -m app.query_process.api.query_service &
```

## 注意事项

- `env -u HTTP_PROXY ...` 是为了绕过系统代理（127.0.0.1:10808），否则本地服务之间会连不通
- moto_server 是内存存储，重启后 bucket 和上传的文件会丢，需要重新创建 bucket 和重新上传 PDF
- 任务队列在内存里，重启 import_service 后需要重新上传 PDF
