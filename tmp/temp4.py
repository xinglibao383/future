import warnings
warnings.filterwarnings("ignore")

import torchvision
torchvision.disable_beta_transforms_warning()

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import torch
import requests
import json

# ========== 参数配置 ==========
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "doc_demo"
EMBEDDING_DIM = 1024  # bge-large-zh 模型输出维度

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen:7b"

# ========== 初始化向量模型 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("/home/xinglibao/workspace/future/embedding/bge-large-zh-v1.5")

def embed_text(text):
    prompt = "为这个句子生成表示以用于检索: " + text
    embedding = embedding_model.encode(prompt, normalize_embeddings=True)
    return embedding

# ========== 连接 Milvus ==========
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# ========== 创建或加载 Collection ==========
if COLLECTION_NAME not in utility.list_collections():
    print(f"Collection {COLLECTION_NAME} 不存在，开始创建...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]
    schema = CollectionSchema(fields, description="Demo collection")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
else:
    collection = Collection(name=COLLECTION_NAME)
collection.load()

# ========== 插入示例文档 ==========
docs = [
    "注册会计师考试包括《会计》《审计》《税法》《财务成本管理》《经济法》《公司战略与风险管理》六门科目。",
    "考试时间一般安排在每年的十月。",
    "考生需要通过所有科目才能获得全科合格证书。",
    "一般程序员喜欢吃火锅。",
]

# Milvus插入格式要求: 每个字段的列表数据
embeddings = [embed_text(d) for d in docs]

# 判断是否已插入过数据，避免重复插入
if collection.num_entities == 0:
    collection.insert([embeddings, docs])
    collection.flush()
    print("✅ 向量已写入 Milvus！")
else:
    print("数据已存在，跳过插入。")

# ========== 定义检索函数 ==========
def retrieve_documents(query, top_k=3):
    query_vector = embed_text(query)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    retrieved_texts = [hit.entity.get("text") for hit in results[0]]
    return retrieved_texts

# ========== 调用 Ollama Qwen 模型生成回答 ==========
def generate_answer(query, contexts):
    # 拼接上下文（你可以根据需求设计拼接格式）
    context_text = "\n".join(contexts)
    prompt = f"基于以下内容尽可能详细地回答问题，同时进行一些拓展：\n{context_text}\n问题：{query}\n回答："

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_API_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=30)
        resp.raise_for_status()
        resp_json = resp.json()
        # Ollama 返回的格式可能是 { "results": [ { "text": "回答内容" } ] }
        return resp_json.get("response", "").strip()
    except Exception as e:
        return f"生成回答时出错: {e}"

# ========== RAG流程 ==========
def rag_pipeline(query):
    retrieved_docs = retrieve_documents(query)
    print("检索到的相关文档：")
    print(retrieved_docs)

    answer = generate_answer(query, retrieved_docs)
    return answer

# ========== 主流程 ==========
if __name__ == "__main__":
    user_query = "注册会计师考什么内容？"
    final_answer = rag_pipeline(user_query)
    print("\n🤖 生成的答案：")
    print(final_answer)