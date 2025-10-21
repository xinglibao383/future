#!/bin/bash

set -e

echo "📦 创建数据目录..."
mkdir -p ~/milvus-docker/milvus
mkdir -p ~/ollama/logs

# ========== Milvus 启动部分 ==========
# etcd
if docker ps -a --format '{{.Names}}' | grep -q '^etcd$'; then
  if ! docker ps --format '{{.Names}}' | grep -q '^etcd$'; then
    echo "🔁 etcd 容器已存在但未运行，尝试启动..."
    docker start etcd
  else
    echo "✅ etcd 容器正在运行，跳过启动。"
  fi
else
  echo "🚀 启动 etcd 容器..."
  docker run -d --name etcd \
    --network host \
    quay.io/coreos/etcd:v3.5.5 \
    etcd -advertise-client-urls http://localhost:2379 \
         -listen-client-urls http://0.0.0.0:2379 \
         -listen-peer-urls http://0.0.0.0:2380
fi

# MinIO
if docker ps -a --format '{{.Names}}' | grep -q '^minio$'; then
  if ! docker ps --format '{{.Names}}' | grep -q '^minio$'; then
    echo "🔁 MinIO 容器已存在但未运行，尝试启动..."
    docker start minio
  else
    echo "✅ MinIO 容器正在运行，跳过启动。"
  fi
else
  echo "🚀 启动 MinIO 容器..."
  docker run -d --name minio \
    --network host \
    -e "MINIO_ACCESS_KEY=minioadmin" \
    -e "MINIO_SECRET_KEY=minioadmin" \
    quay.io/minio/minio server /data
fi

# Milvus
if docker ps -a --format '{{.Names}}' | grep -q '^milvus-standalone$'; then
  if ! docker ps --format '{{.Names}}' | grep -q '^milvus-standalone$'; then
    echo "🔁 Milvus 容器已存在但未运行，尝试启动..."
    docker start milvus-standalone
  else
    echo "✅ Milvus 容器正在运行，跳过启动。"
  fi
else
  echo "🚀 启动 Milvus Standalone 容器..."
  docker run -d --name milvus-standalone \
    --network host \
    -v ~/milvus-docker/milvus:/var/lib/milvus \
    -e ETCD_ENDPOINTS="localhost:2379" \
    -e MINIO_ADDRESS="localhost:9000" \
    -e MINIO_ACCESS_KEY="minioadmin" \
    -e MINIO_SECRET_KEY="minioadmin" \
    milvusdb/milvus:v2.4.4
fi

echo "🎉 Milvus 所有服务已就绪。"

# ========== Ollama 启动部分 ==========
if ! pgrep -f "ollama serve" >/dev/null; then
  echo "🚀 启动 Ollama 后台服务..."
  nohup ~/ollama/bin/ollama serve > ~/ollama/logs/ollama.log 2>&1 &
  sleep 3
else
  echo "✅ Ollama 服务已在运行中。"
fi

# 检查 Ollama 是否可用
if curl -s http://localhost:11434/api/tags >/dev/null; then
  echo "✅ Ollama API 正常运行。"
else
  echo "❌ Ollama API 启动失败，请检查 ~/ollama/logs/ollama.log"
fi

# ========== 下载 & 加载 Qwen 模型 ==========
MODEL_NAME="qwen:7b"

echo "📥 检查模型 $MODEL_NAME 是否存在..."
if ~/ollama/bin/ollama list | grep -q "$MODEL_NAME"; then
  echo "✅ 模型 $MODEL_NAME 已存在，跳过下载。"
else
  echo "🚀 下载并加载 Qwen 模型 $MODEL_NAME..."
  ~/ollama/bin/ollama pull "$MODEL_NAME"
fi

echo "🎉 全部服务已就绪：Milvus + Ollama + Qwen"