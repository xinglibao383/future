import ollama

# 检查本地可用的模型
print(ollama.list())  # 查看已下载的模型

# 调用 Qwen 进行文本生成
response = ollama.generate(
    model="qwen:7b",  # 替换成你的模型版本（如 qwen:14b）
    prompt="请用中文解释人工智能的基本概念。",
)

print(response["response"])