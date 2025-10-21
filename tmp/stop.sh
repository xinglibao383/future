#!/bin/bash

set -e

echo "🛑 正在关闭所有组件..."

# 停止 milvus-standalone 容器
if docker ps --format '{{.Names}}' | grep -q '^milvus-standalone$'; then
  echo "⛔ 停止 milvus-standalone..."
  docker stop milvus-standalone
else
  echo "✅ milvus-standalone 已停止或未运行。"
fi

# 停止 etcd 容器
if docker ps --format '{{.Names}}' | grep -q '^etcd$'; then
  echo "⛔ 停止 etcd..."
  docker stop etcd
else
  echo "✅ etcd 已停止或未运行。"
fi

# 停止 minio 容器
if docker ps --format '{{.Names}}' | grep -q '^minio$'; then
  echo "⛔ 停止 minio..."
  docker stop minio
else
  echo "✅ minio 已停止或未运行。"
fi

# 杀掉 ollama serve 进程
if pgrep -f "ollama serve" >/dev/null; then
  echo "⛔ 停止 Ollama 后台服务..."
  pkill -f "ollama serve"
else
  echo "✅ Ollama 服务未运行或已关闭。"
fi

echo "🎉 所有组件已成功关闭。"